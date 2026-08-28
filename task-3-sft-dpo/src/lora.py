"""Hand-written LoRA interfaces used by Task 3.

The core algorithms are intentionally left unimplemented for the exercise.
"""

# %%
from __future__ import annotations
import copy
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import json
from torch import Tensor, nn
import torch


# %%


class LoRALinear(nn.Module):
    """Wrap one frozen ``nn.Linear`` with a trainable low-rank branch."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """Create the frozen base branch and trainable A/B parameters.

        Expected parameter shapes:
        - A: ``[r, in_features]``
        - B: ``[out_features, r]``
        - scaling: ``alpha / r``
        """
        super().__init__()
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.A = nn.Linear(self.base_layer.in_features, r, bias=False, dtype=self.base_layer.weight.dtype)
        self.B = nn.Linear(r, self.base_layer.out_features, bias=False, dtype=self.base_layer.weight.dtype)
        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_normal_(self.A.weight)

    @property
    def delta_weight(self) -> Tensor:
        """Return the scaled weight update with shape ``[out, in]``."""

        return self.alpha / self.r * (self.B.weight @ self.A.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``base(x) + scaling * B(A(x))``."""
        return self.base_layer(x) + self.alpha / self.r * self.B(
            self.A(self.dropout(x))
        )

    def merged_linear(self) -> nn.Linear:
        """Return a plain Linear whose weight includes the LoRA update."""

        with torch.no_grad():
            temp = copy.deepcopy(self.base_layer)
            temp.weight.add_(self.delta_weight)

        return temp


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace matching Linear children with ``LoRALinear`` wrappers.

    ``target_modules`` matches leaf names such as ``q_proj`` and ``v_proj``.
    The function must freeze the original model and leave only LoRA parameters
    trainable. It must mutate and return ``model`` for eval compatibility.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = model
            *path, leaf = name.split(".")
            if leaf in target_modules:
                for p in path:
                    parent = getattr(parent, p)

                setattr(parent, leaf, LoRALinear(module, r, alpha, dropout))

    return model


def iter_lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    """Yield ``(qualified_name, module)`` for every injected adapter."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every ``LoRALinear`` with its merged plain Linear layer."""
    for name, module in list(iter_lora_modules(model)):
        *paths, leaf = name.split(".")
        parent = model
        for path in paths:
            parent = getattr(parent, path)
        setattr(parent, leaf, module.merged_linear())
    return model


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Return only LoRA trainable tensors for a compact checkpoint."""
    state_dict = {}
    for name, module in iter_lora_modules(model):
        state_dict[f"{name}.A.weight"] = module.A.weight.detach().cpu()
        state_dict[f"{name}.B.weight"] = module.B.weight.detach().cpu()
    return state_dict


def save_lora_adapter(
    model: nn.Module,
    output_dir: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save adapter tensors and enough metadata to reconstruct injection."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lora_dict = lora_state_dict(model)

    torch.save(lora_dict, output_path / "lora_dict.pt")

    metadata = dict(metadata) if metadata else {}

    (output_path / "metadata.json").write_text(json.dumps(metadata))


def load_lora_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    strict: bool = True,
) -> nn.Module:
    """Load adapter tensors into an already injected model."""

    adapter_path = Path(adapter_dir)

    metadata = json.loads((adapter_path / "metadata.json").read_text())
    lora_dict = torch.load(adapter_path / "lora_dict.pt", weights_only=True)

    expected_keys = set(lora_state_dict(model))
    loaded_keys = set(lora_dict)

    if strict and expected_keys != loaded_keys:
        raise RuntimeError("incorrect keys")

    for name, module in iter_lora_modules(model):
        if f"{name}.A.weight" not in lora_dict or f"{name}.B.weight" not in lora_dict:
            if strict:
                raise RuntimeError("incorrect keys")
            else:
                continue
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.A.weight.copy_(lora_dict[f"{name}.A.weight"])
                module.B.weight.copy_(lora_dict[f"{name}.B.weight"])

    return model
