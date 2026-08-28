"""Response log-probability and DPO loss interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DPOLossOutput:
    """Scalar DPO loss and per-sample reward diagnostics."""

    loss: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    reward_margin: Tensor


def sequence_log_probs(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    average_log_prob: bool = False,
) -> Tensor:
    """Return one response log-probability per batch item.

    ``logits`` has shape ``[B, T, V]`` and ``labels`` has shape ``[B, T]``.
    The implementation must apply the causal one-token shift and ignore masked
    positions.
    """
    import torch

    T = labels.shape[-1]

    softmax_logits = torch.log_softmax(logits[:, :-1, :], dim=-1)

    # labels[:, 0:T-1] = labels[:, 1:]
    # labels[:, T-1:] = ignore_index

    per_token_logps = softmax_logits.gather(
        dim=-1,
        index=labels[:, 1:]
        .unsqueeze(-1)
        .clamp(min=0),  # [B, T, 1]，clamp 防止 -100 越界
    ).squeeze(-1)

    per_token_logps = per_token_logps * (labels[:, 1:] != ignore_index)

    logps_sum = torch.sum(per_token_logps, dim=-1)

    if average_log_prob:
        logps_sum = logps_sum / (labels[:, 1:] != ignore_index).sum(dim=-1).clamp(1)

    return logps_sum


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
) -> DPOLossOutput:
    """Compute DPO loss and chosen/rejected implicit rewards."""

    diff_policy = policy_chosen_logps - policy_rejected_logps

    diff_refer = reference_chosen_logps - reference_rejected_logps

    import torch

    loss = -torch.nn.functional.logsigmoid(beta * (diff_policy - diff_refer))

    return DPOLossOutput(
        loss=loss.mean(),
        chosen_rewards=beta * (policy_chosen_logps - reference_chosen_logps),
        rejected_rewards=beta * (policy_rejected_logps - reference_rejected_logps),
        reward_margin=beta * (policy_chosen_logps - reference_chosen_logps)
        - beta * (policy_rejected_logps - reference_rejected_logps),
    )
