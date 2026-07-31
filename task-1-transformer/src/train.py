import argparse
from io import BytesIO
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .model import TransformerClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "google-bert/bert-base-chinese"

# Prefer common macOS Chinese fonts while keeping a portable fallback.
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Arial Unicode MS",
    "Heiti TC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class SentimentDataset(Dataset):
    def __init__(self, path, max_samples=None):
        df = pd.read_parquet(path)
        if max_samples is not None:
            df = df.iloc[:max_samples]

        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


class SentimentCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class AttentionEvolutionRecorder:
    """Track one query token's attention distribution during training."""

    def __init__(
        self,
        tokenizer,
        text,
        query_token,
        layer_index,
        max_length,
        device,
    ):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].to(device)
        self.tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0].tolist()
        )
        matching_indices = [
            index for index, token in enumerate(self.tokens) if token == query_token
        ]
        if not matching_indices:
            raise ValueError(
                f"query token {query_token!r} not found in tokens: {self.tokens}"
            )

        self.text = text
        self.query_token = query_token
        self.query_index = matching_indices[0]
        self.layer_index = layer_index
        self.resolved_layer = None
        self.snapshots = []

    @torch.no_grad()
    def capture(self, model, epoch):
        was_training = model.training
        model.eval()
        try:
            probabilities = torch.softmax(model(self.input_ids), dim=-1)[0]
            weights, valid_mask, resolved_layer = model.get_attention_weights(
                self.input_ids,
                layer_index=self.layer_index,
            )
            valid_length = int(valid_mask[0].sum().item())
            query_weights = weights[
                0,
                :,
                self.query_index,
                :valid_length,
            ]
            probabilities = probabilities.detach().cpu()
            predicted_label = int(probabilities.argmax().item())

            self.resolved_layer = resolved_layer
            self.snapshots.append(
                {
                    "epoch": epoch,
                    "predicted_label": predicted_label,
                    "class_probabilities": probabilities.tolist(),
                    "query_attention_by_head": (
                        query_weights.detach().cpu().tolist()
                    ),
                }
            )
        finally:
            model.train(was_training)

    def save_json(self, output_path):
        data = {
            "text": self.text,
            "tokens": self.tokens,
            "query_token": self.query_token,
            "query_index": self.query_index,
            "layer": self.resolved_layer + 1,
            "axis_meaning": {
                "rows": "training epochs (0 means initialization)",
                "columns": "key tokens attended to by the query token",
            },
            "snapshots": self.snapshots,
        }
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs,
    learning_rate,
    weight_decay,
    grad_clip,
    checkpoint_path,
    attention_recorder=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "gradient_step": [],
        "gradient_norm": [],
    }

    best_validation_loss = float("inf")
    global_step = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

            # The returned value is the global norm before clipping.
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )
            optimizer.step()

            global_step += 1
            history["gradient_step"].append(global_step)
            history["gradient_norm"].append(
                float(gradient_norm.detach().cpu())
            )

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved best checkpoint -> {checkpoint_path}")

        if attention_recorder is not None:
            attention_recorder.capture(model, epoch)

    return history


def plot_loss(history, output_path):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axis.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_gradient_norm(history, grad_clip, output_path):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        history["gradient_step"],
        history["gradient_norm"],
        linewidth=1,
        label="global norm before clipping",
    )
    axis.axhline(
        grad_clip,
        color="tab:red",
        linestyle="--",
        label=f"clip threshold={grad_clip:g}",
    )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Global gradient norm")
    axis.set_title("Gradient norm during training")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _attention_evolution_matrix(recorder, head_index):
    n_heads = len(recorder.snapshots[0]["query_attention_by_head"])
    if not 0 <= head_index < n_heads:
        raise IndexError(f"head_index out of range: {head_index}")

    return torch.tensor(
        [
            snapshot["query_attention_by_head"][head_index]
            for snapshot in recorder.snapshots
        ]
    ).numpy()


def plot_attention_evolution(recorder, head_index, output_path):
    matrix = _attention_evolution_matrix(recorder, head_index)
    epoch_labels = [
        "init" if snapshot["epoch"] == 0 else str(snapshot["epoch"])
        for snapshot in recorder.snapshots
    ]

    width = max(9.0, min(15.0, len(recorder.tokens) * 0.5))
    height = max(4.0, len(epoch_labels) * 0.75)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, cmap="magma", aspect="auto", vmin=0)
    axis.set_xticks(
        range(len(recorder.tokens)),
        labels=recorder.tokens,
        rotation=90,
    )
    axis.set_yticks(range(len(epoch_labels)), labels=epoch_labels)
    axis.set_xlabel("Key token")
    axis.set_ylabel("Epoch")
    axis.set_title(
        f"How query token '{recorder.query_token}' changes its attention\n"
        f"layer {recorder.resolved_layer + 1}, head {head_index + 1}"
    )
    figure.colorbar(image, ax=axis, label="Attention weight")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_attention_evolution_all_heads(recorder, output_path):
    n_heads = len(recorder.snapshots[0]["query_attention_by_head"])
    n_columns = min(2, n_heads)
    n_rows = (n_heads + n_columns - 1) // n_columns
    matrices = [
        _attention_evolution_matrix(recorder, head_index)
        for head_index in range(n_heads)
    ]
    maximum = max(float(matrix.max()) for matrix in matrices)
    epoch_labels = [
        "init" if snapshot["epoch"] == 0 else str(snapshot["epoch"])
        for snapshot in recorder.snapshots
    ]

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(max(12, len(recorder.tokens) * 0.7), 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = [axis for row in axes for axis in row]
    images = []
    for head_index, matrix in enumerate(matrices):
        axis = flat_axes[head_index]
        image = axis.imshow(
            matrix,
            cmap="magma",
            aspect="auto",
            vmin=0,
            vmax=maximum,
        )
        images.append(image)
        axis.set_xticks(
            range(len(recorder.tokens)),
            labels=recorder.tokens,
            rotation=90,
        )
        axis.set_yticks(range(len(epoch_labels)), labels=epoch_labels)
        axis.set_title(f"Head {head_index + 1}")
        axis.set_xlabel("Key token")
        axis.set_ylabel("Epoch")

    for axis in flat_axes[n_heads:]:
        axis.remove()

    figure.suptitle(
        f"Attention evolution for query token '{recorder.query_token}' "
        f"- layer {recorder.resolved_layer + 1}"
    )
    figure.colorbar(
        images[0],
        ax=flat_axes[:n_heads],
        label="Attention weight",
        shrink=0.8,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def create_attention_evolution_gif(
    recorder,
    head_index,
    output_path,
    duration_ms=900,
):
    matrix = _attention_evolution_matrix(recorder, head_index)
    maximum = float(matrix.max())
    frames = []

    for row_index, snapshot in enumerate(recorder.snapshots):
        figure, axis = plt.subplots(
            figsize=(max(9, len(recorder.tokens) * 0.5), 3.2)
        )
        image = axis.imshow(
            matrix[row_index : row_index + 1],
            cmap="magma",
            aspect="auto",
            vmin=0,
            vmax=maximum,
        )
        axis.set_xticks(
            range(len(recorder.tokens)),
            labels=recorder.tokens,
            rotation=90,
        )
        axis.set_yticks([0], labels=[recorder.query_token])
        axis.set_xlabel("Key token")
        axis.set_ylabel("Query token")
        stage = "initialization" if snapshot["epoch"] == 0 else f"epoch {snapshot['epoch']}"
        confidence = max(snapshot["class_probabilities"])
        axis.set_title(
            f"Attention evolution - {stage}\n"
            f"layer {recorder.resolved_layer + 1}, head {head_index + 1}, "
            f"prediction={snapshot['predicted_label']}, confidence={confidence:.3f}"
        )
        figure.colorbar(image, ax=axis, label="Attention weight")
        figure.tight_layout()

        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=130)
        plt.close(figure)
        buffer.seek(0)
        with Image.open(buffer) as frame:
            frames.append(frame.convert("RGB").copy())

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


@torch.no_grad()
def collect_attention_weights(model, input_ids, layer_index=-1):
    model.eval()
    weights, valid_mask, resolved_layer = model.get_attention_weights(
        input_ids,
        layer_index=layer_index,
    )
    return (
        weights.detach().cpu(),
        valid_mask.detach().cpu(),
        resolved_layer,
    )


@torch.no_grad()
def plot_attention(
    model,
    tokenizer,
    text,
    device,
    layer_index,
    head_index,
    max_length,
    output_path,
    data_output_path,
):
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    probabilities = torch.softmax(model(input_ids), dim=-1)[0].detach().cpu()
    predicted_label = int(probabilities.argmax().item())
    weights, valid_mask, resolved_layer = collect_attention_weights(
        model,
        input_ids,
        layer_index=layer_index,
    )

    if not 0 <= head_index < weights.size(1):
        raise IndexError(f"head_index out of range: {head_index}")

    valid_length = int(valid_mask[0].sum().item())
    matrix = weights[0, head_index, :valid_length, :valid_length].numpy()
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0, :valid_length].detach().cpu().tolist()
    )

    size = max(7.0, min(14.0, valid_length * 0.45))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="magma", aspect="auto")
    axis.set_xticks(range(valid_length), labels=tokens, rotation=90)
    axis.set_yticks(range(valid_length), labels=tokens)
    axis.set_xlabel("Key token")
    axis.set_ylabel("Query token")
    axis.set_title(
        f"Attention heatmap - layer {resolved_layer + 1}, head {head_index + 1}\n"
        f"prediction={predicted_label}, confidence={probabilities[predicted_label]:.3f}"
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    attention_data = {
        "text": text,
        "tokens": tokens,
        "layer": resolved_layer + 1,
        "head": head_index + 1,
        "predicted_label": predicted_label,
        "class_probabilities": probabilities.tolist(),
        "axis_meaning": {
            "rows": "query tokens",
            "columns": "key tokens",
        },
        "attention_weights": matrix.tolist(),
    }
    data_output_path.write_text(
        json.dumps(attention_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Transformer classifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-data", default="data/train.parquet")
    parser.add_argument("--validation-data", default="data/validation.parquet")
    parser.add_argument("--checkpoint", default="ckpt/best.pt")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--attention-layer", type=int, default=-1)
    parser.add_argument("--attention-head", type=int, default=0)
    parser.add_argument("--attention-query-token", default="差")
    parser.add_argument("--attention-max-tokens", type=int, default=32)
    parser.add_argument(
        "--attention-text",
        default="这家酒店位置很好，但是房间太脏，服务也很差。",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data_path = resolve_project_path(args.train_data)
    validation_data_path = resolve_project_path(args.validation_data)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = SentimentCollator(tokenizer, args.max_length)
    train_dataset = SentimentDataset(
        train_data_path,
        max_samples=args.max_train_samples,
    )
    validation_dataset = SentimentDataset(
        validation_data_path,
        max_samples=args.max_validation_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = get_device()
    model = TransformerClassifier(max_len=args.max_length).to(device)
    print(
        f"device={device}, train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}"
    )

    attention_recorder = AttentionEvolutionRecorder(
        tokenizer=tokenizer,
        text=args.attention_text,
        query_token=args.attention_query_token,
        layer_index=args.attention_layer,
        max_length=min(args.max_length, args.attention_max_tokens),
        device=device,
    )
    attention_recorder.capture(model, epoch=0)

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        checkpoint_path=checkpoint_path,
        attention_recorder=attention_recorder,
    )

    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_loss(history, output_dir / "loss_curve.png")
    plot_gradient_norm(
        history,
        args.grad_clip,
        output_dir / "gradient_norm.png",
    )
    attention_recorder.save_json(output_dir / "attention_evolution.json")
    plot_attention_evolution(
        attention_recorder,
        args.attention_head,
        output_dir / "attention_evolution.png",
    )
    plot_attention_evolution_all_heads(
        attention_recorder,
        output_dir / "attention_evolution_all_heads.png",
    )
    create_attention_evolution_gif(
        attention_recorder,
        args.attention_head,
        output_dir / "attention_evolution.gif",
    )

    best_state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.to(device).eval()
    plot_attention(
        model=model,
        tokenizer=tokenizer,
        text=args.attention_text,
        device=device,
        layer_index=args.attention_layer,
        head_index=args.attention_head,
        max_length=min(args.max_length, args.attention_max_tokens),
        output_path=output_dir / "attention_heatmap.png",
        data_output_path=output_dir / "attention_data.json",
    )

    print(f"history -> {history_path}")
    print(f"plots   -> {output_dir}")


if __name__ == "__main__":
    main()
