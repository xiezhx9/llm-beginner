from torch import nn
import torch
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer

from .block import TransformerBlock
from .SinusoidalPE import SinusoidalPE



# %%

def load_for_eval(
    ckpt_path: str,
):  # -> tuple[TransformerClassifier, Callable[[str], torch.Tensor]]:

    model = TransformerClassifier()

    token_fn = lambda x: model.tokenizer(
        x,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length = model.max_len
    )["input_ids"].squeeze(0)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, token_fn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        d_model=64,
        n_heads=4,
        ff_dim=128,
        n_layers=2,
        n_class=2,
        dropout=0.1,
        max_len=64,
    ):
        super().__init__()

        model_name = "google-bert/bert-base-chinese"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.embedding = nn.Embedding(
            num_embeddings=len(self.tokenizer),
            embedding_dim=d_model,
            padding_idx=self.tokenizer.pad_token_id,
        )

        self.max_len = max_len

        self.pe = SinusoidalPE(d_model, max_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, dropout)
                for i in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, n_class)

    def _valid_token_mask(self, input_ids, lengths=None):
        if lengths is not None:
            return (
                torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
                < lengths[:, None]
            )
        return input_ids != self.tokenizer.pad_token_id

    def get_attention_weights(self, input_ids, layer_index=-1, lengths=None):
        """Return the real softmax weights from one encoder attention layer."""
        valid_mask = self._valid_token_mask(input_ids, lengths)
        padding_mask = ~valid_mask[:, None, None, :]
        token_vec = self.pe(self.embedding(input_ids))

        if layer_index < 0:
            layer_index += len(self.blocks)
        if not 0 <= layer_index < len(self.blocks):
            raise IndexError(f"layer_index out of range: {layer_index}")

        for index, block in enumerate(self.blocks):
            if index == layer_index:
                _, weights = block.attn(
                    block.layer_norm1(token_vec),
                    padding_mask,
                    return_weights=True,
                )
                return weights, valid_mask, layer_index
            token_vec = block(token_vec, padding_mask)

        raise RuntimeError("attention layer was not reached")

    def forward(self, input_ids, lengths: torch.Tensor | None = None):
        # batch = self.tokenizer(
        #     x,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt",
        #     max_length=self.max_len
        # )

        # input_ids = batch["input_ids"]
        # attention_mask:torch.Tensor = batch["attention_mask"]

        attention_mask = self._valid_token_mask(input_ids, lengths)

        token_vec = self.embedding(input_ids)

        token_vec = self.pe(token_vec)

        for block in self.blocks:
            token_vec = block(token_vec, ~attention_mask[:, None, None, :].bool())

        token_vec = self.norm(token_vec)

        # pooled_vec = token_vec[:, 0, :]
        attention_mask = attention_mask.unsqueeze(-1).float()
        pooled_vec = (attention_mask * token_vec).sum(dim=-2) / attention_mask.sum(
            dim=-2
        ).clamp_min(1)

        result = self.classifier(pooled_vec)

        return result
