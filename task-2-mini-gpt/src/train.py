"""Training interfaces for the Task 2 mini-GPT language model."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from .model import MiniGPT, MiniGPTConfig
from .tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    """Hyperparameters and artifact paths used by the training pipeline."""

    train_path: Path = Path("data/train.txt")
    dev_path: Path = Path("data/dev.txt")
    tokenizer_path: Path = Path("ckpt/tokenizer.json")
    checkpoint_path: Path = Path("ckpt/best.pt")
    history_path: Path = Path("ckpt/training_history.json")
    curve_path: Path = Path("ckpt/training_curves.png")
    batch_size: int = 8
    epochs: int = 60
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0
    min_epochs: int = 1
    log_interval: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str | None = None


@dataclass
class EpochMetrics:
    """Aggregated measurements for one training or evaluation epoch."""

    loss: float
    perplexity: float
    learning_rate: float | None = None
    grad_norm: float | None = None
    step: int | None = None


@dataclass
class TrainingHistory:
    """Loss and perplexity values collected across all epochs."""

    steps: list[int] = field(default_factory=list)
    batch_loss: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_perplexity: list[float] = field(default_factory=list)
    dev_loss: list[float] = field(default_factory=list)
    dev_perplexity: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    stopped_early: bool = False


# %%
a = [1]
a[:3]
# %%


@dataclass
class EarlyStoppingState:
    """Best validation result and the number of consecutive stale epochs."""

    best_dev_loss: float
    best_epoch: int
    stale_epochs: int = 0


train_config: TrainConfig = TrainConfig()


class NextTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create fixed-length input/target windows from one token ID sequence."""

    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        stride: int | None = None,
    ) -> None:
        """Store token IDs and configure the window length and step size."""
        self.token_ids = token_ids
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size

    def __len__(self) -> int:
        """Return the number of available next-token training windows."""
        return (len(self.token_ids) - self.block_size - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return input IDs and their one-position-shifted target IDs."""
        start = index * self.stride
        if start >= len(self.token_ids) or start + 1 >= len(self.token_ids):
            raise ValueError("dataset out of Bound")
        input_ids = self.token_ids[start : start + self.block_size]
        target_ids = self.token_ids[start + 1 : start + 1 + self.block_size]

        return Tensor(input_ids).long(), Tensor(target_ids).long()


def load_token_ids(
    text_path: str | Path,
    tokenizer: BPETokenizer,
) -> list[int]:
    """Read one UTF-8 corpus and encode it into a flat token ID sequence."""
    bpe = tokenizer
    train_text = Path(text_path).read_text("utf-8")

    return bpe.encode(train_text)


def build_dataloaders(
    train_ids: list[int],
    dev_ids: list[int],
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    """Construct shuffled training and deterministic validation loaders."""
    train_dataset = NextTokenDataset(train_ids, block_size, stride=8)
    validation_dataset = NextTokenDataset(dev_ids, block_size, stride=8)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, validation_loader


def compute_next_token_loss(
    model: MiniGPT,
    input_ids: Tensor,
    target_ids: Tensor,
) -> Tensor:
    """Compute cross-entropy between token logits and shifted targets."""

    loss_fn = torch.nn.CrossEntropyLoss()
    output = model(input_ids)

    loss_item = loss_fn(
        output.reshape(-1, output.shape[-1]), target_ids.reshape(-1, 1).squeeze(-1)
    )

    return loss_item


def build_optimizer(
    model: MiniGPT,
    config: TrainConfig,
) -> AdamW:
    """Create the AdamW optimizer for all trainable model parameters."""
    optimizer = AdamW(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
    )
    return optimizer


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    min_learning_rate: float,
) -> LRScheduler:
    """Create the cosine learning-rate scheduler used during training."""
    warmup_ratio = 0.05
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    cosine_steps = max(1, total_steps - warmup_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_learning_rate,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def train_one_epoch(
    model: MiniGPT,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: AdamW,
    scheduler: LRScheduler,
    device: str,
    grad_clip: float,
) -> EpochMetrics:
    """Run one training epoch and return aggregated measurements."""
    model = model.to(device)
    model.train()
    total_loss = 0
    total_token = 0
    total_grad_norm = 0
    total_round = 0

    for batch in dataloader:
        train_data, target_data = batch
        train_data = train_data.to(device)
        target_data = target_data.to(device)

        optimizer.zero_grad()
        loss = compute_next_token_loss(model, train_data, target_data)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=grad_clip,
        )
        optimizer.step()
        scheduler.step()
        total_loss += loss.float().item() * train_data.numel()
        total_token += train_data.numel()
        total_grad_norm += grad_norm.detach().item()
        total_round += 1

    metric = EpochMetrics(
        total_loss / total_token,
        math.exp(total_loss / total_token),
        scheduler.get_last_lr()[0],
        total_grad_norm / total_round,
        total_round,
    )
    return metric


@torch.no_grad()
def evaluate(
    model: MiniGPT,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: str,
) -> EpochMetrics:
    """Measure validation loss and perplexity without updating parameters."""
    model = model.to(device)
    model.eval()
    total_loss = 0
    total_token = 0

    for batch in dataloader:
        dev_train_data, target_data = batch
        dev_train_data = dev_train_data.to(device)
        target_data = target_data.to(device)

        loss = compute_next_token_loss(model, dev_train_data, target_data)

        total_loss += loss.float().item() * dev_train_data.numel()
        total_token += dev_train_data.numel()

    metric = EpochMetrics(
        total_loss / total_token, math.exp(total_loss / total_token), None, None
    )
    return metric


def update_early_stopping(
    state: EarlyStoppingState,
    dev_loss: float,
    epoch: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
) -> tuple[EarlyStoppingState, bool, bool]:
    """Return updated state plus ``improved`` and ``should_stop`` flags."""

    if state.best_dev_loss - min_delta > dev_loss:
        state.best_dev_loss = dev_loss
        state.best_epoch = epoch
        state.stale_epochs = 0
        return state, True, False
    elif epoch + 1 < min_epochs:
        state.stale_epochs += 1
        return state, False, False
    else:
        state.stale_epochs += 1
        if state.stale_epochs >= patience:
            return state, False, True
        else:
            return state, False, False


def save_checkpoint(
    path: str | Path,
    model: MiniGPT,
    model_config: MiniGPTConfig,
    epoch: int,
    dev_metrics: EpochMetrics,
) -> None:
    """Persist model state, construction config, and best validation metadata."""

    path = Path(path)
    PARENT = path.parent

    PARENT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    model_config_path = PARENT / "model_config.json"
    best_dev_metric = PARENT / "best_validation_metric.json"

    with open(model_config_path, "w") as f:
        json.dump(dataclasses.asdict(model_config), f)
    with open(best_dev_metric, "w") as f:
        metric_dict = dataclasses.asdict(dev_metrics)
        metric_dict["epoch"] = epoch
        json.dump(metric_dict, f)


def save_training_history(
    history: TrainingHistory,
    path: str | Path,
) -> None:
    """Persist collected step-level and epoch-level metrics as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(dataclasses.asdict(history), f)


def plot_training_history(
    history: TrainingHistory,
    path: str | Path,
) -> None:
    """Plot loss, perplexity, learning rate, and gradient-norm curves."""
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    epoch_count = max(
        len(history.train_loss),
        len(history.dev_loss),
        len(history.train_perplexity),
        len(history.dev_perplexity),
    )
    if epoch_count == 0:
        raise ValueError("training history does not contain any epoch metrics")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    use_steps = len(history.steps) == epoch_count
    epoch_x = history.steps if use_steps else list(range(1, epoch_count + 1))
    x_label = "Optimizer step" if use_steps else "Epoch"

    figure = Figure(figsize=(12, 8), constrained_layout=True, facecolor="#f5f1e8")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.suptitle("Mini-GPT Training Dashboard", fontsize=16, fontweight="bold")

    colors = {
        "train": "#16425b",
        "dev": "#c44536",
        "learning_rate": "#d28b26",
        "grad_norm": "#2a7f62",
        "best": "#6b705c",
    }

    def x_values(values: list[float]) -> list[int]:
        if len(values) == epoch_count:
            return epoch_x
        return list(range(1, len(values) + 1))

    def prepare_axis(axis: Axes, title: str, ylabel: str) -> None:
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(x_label)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_facecolor("#fffdf8")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    loss_axis, perplexity_axis, lr_axis, grad_axis = axes.flat

    prepare_axis(loss_axis, "Cross-Entropy Loss", "Loss")
    loss_axis.plot(
        x_values(history.train_loss),
        history.train_loss,
        color=colors["train"],
        marker="o",
        markersize=4,
        label="Train",
    )
    loss_axis.plot(
        x_values(history.dev_loss),
        history.dev_loss,
        color=colors["dev"],
        marker="o",
        markersize=4,
        label="Dev",
    )
    loss_axis.legend(frameon=False)

    prepare_axis(perplexity_axis, "Perplexity", "Perplexity")
    perplexity_axis.plot(
        x_values(history.train_perplexity),
        history.train_perplexity,
        color=colors["train"],
        marker="o",
        markersize=4,
        label="Train",
    )
    perplexity_axis.plot(
        x_values(history.dev_perplexity),
        history.dev_perplexity,
        color=colors["dev"],
        marker="o",
        markersize=4,
        label="Dev",
    )
    perplexity_axis.legend(frameon=False)

    prepare_axis(lr_axis, "Cosine Learning Rate", "Learning rate")
    lr_axis.plot(
        x_values(history.learning_rate),
        history.learning_rate,
        color=colors["learning_rate"],
        marker="o",
        markersize=4,
    )
    lr_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    prepare_axis(grad_axis, "Average Gradient Norm", "Gradient norm")
    grad_axis.plot(
        x_values(history.grad_norm),
        history.grad_norm,
        color=colors["grad_norm"],
        marker="o",
        markersize=4,
    )

    if history.best_epoch is not None and 0 <= history.best_epoch < epoch_count:
        best_x = epoch_x[history.best_epoch]
        for axis in (loss_axis, perplexity_axis):
            axis.axvline(
                best_x,
                color=colors["best"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
        loss_axis.text(
            best_x,
            0.02,
            " best",
            transform=loss_axis.get_xaxis_transform(),
            color=colors["best"],
            fontsize=9,
            va="bottom",
        )

    figure.savefig(path, dpi=160, facecolor=figure.get_facecolor())


def train_model(
    model: MiniGPT,
    model_config: MiniGPTConfig,
    train_config: TrainConfig,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    dev_loader: DataLoader[tuple[Tensor, Tensor]],
) -> TrainingHistory:
    """Coordinate optimization, scheduling, early stopping, and best checkpointing."""
    state = EarlyStoppingState(float("inf"), -1)
    training_history = TrainingHistory()
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(
        optimizer,
        train_config.epochs * len(train_loader),
        train_config.min_learning_rate,
    )
    for epoch in range(train_config.epochs):
        metric: EpochMetrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            train_config.device,
            train_config.grad_clip,
        )

        training_history.steps.append(metric.step)
        training_history.train_loss.append(metric.loss)
        training_history.learning_rate.append(metric.learning_rate)
        training_history.grad_norm.append(metric.grad_norm)
        training_history.train_perplexity.append(metric.perplexity)

        dev_metric = evaluate(model, dev_loader, train_config.device)

        training_history.dev_loss.append(dev_metric.loss)
        training_history.dev_perplexity.append(dev_metric.perplexity)

        state, improved, should_stop = update_early_stopping(
            state,
            dev_metric.loss,
            epoch,
            train_config.early_stopping_patience,
            train_config.early_stopping_min_delta,
            train_config.min_epochs,
        )
        training_history.stopped_early = should_stop
        training_history.best_epoch = state.best_epoch
        if improved:
            save_checkpoint(
                train_config.checkpoint_path, model, model_config, epoch, dev_metric
            )

        if should_stop:
            break
    training_history.steps = torch.cumsum(
        torch.tensor(training_history.steps, dtype=torch.long), dim=0
    ).tolist()
    save_training_history(training_history, train_config.history_path)
    plot_training_history(training_history, train_config.curve_path)

    return training_history


def main() -> None:
    """Load artifacts, construct the model and data pipeline, and start training."""
    bpe = BPETokenizer().from_pretrained(train_config.tokenizer_path)

    token_ids = load_token_ids(train_config.train_path, bpe)
    dev_ids = load_token_ids(train_config.dev_path, bpe)


    model_config = MiniGPTConfig(bpe.vocab_size)
    train_loader, dev_loader = build_dataloaders(token_ids, dev_ids, model_config.block_size, train_config.batch_size)

    model = MiniGPT(model_config)

    history = train_model(model, model_config, train_config, train_loader, dev_loader)



if __name__ == "__main__":
    main()

# %%
