"""Stable CRE probability, NLL, residual and pair-ranking losses."""

import torch
import torch.nn.functional as F


def condition_scores_from_logits(base_logits):
    base = base_logits.float() if base_logits.dtype in (torch.float16, torch.bfloat16) else base_logits
    s_c = torch.logsumexp(base[:, 2:4], dim=1) - torch.logsumexp(base[:, 0:2], dim=1)
    s_h = base[:, 1] - base[:, 0]
    s_a = base[:, 3] - base[:, 2]
    return s_c, s_h, s_a


def combine_condition_scores(s_c, s_h, s_a):
    if any(value.dtype in (torch.float16, torch.bfloat16) for value in (s_c, s_h, s_a)):
        s_c, s_h, s_a = (value.float() for value in (s_c, s_h, s_a))
    logp = torch.stack([
        F.logsigmoid(-s_c) + F.logsigmoid(-s_h),
        F.logsigmoid(-s_c) + F.logsigmoid(s_h),
        F.logsigmoid(s_c) + F.logsigmoid(-s_a),
        F.logsigmoid(s_c) + F.logsigmoid(s_a),
    ], dim=-1)
    return logp


def condition_probabilities(s_c, s_h, s_a):
    return combine_condition_scores(s_c, s_h, s_a).exp()


def condition_nll(logp, labels):
    return F.nll_loss(logp, labels)


def residual_regularization(delta, conditional):
    if conditional:
        return delta.float().square().mean()
    centered = delta.float() - delta.float().mean(dim=1, keepdim=True)
    return centered.square().mean()


def pair_ranking_loss(logp, pair_indices, pair_types, margin=0.0, temperature=1.0):
    if not pair_indices:
        return logp.sum() * 0.0
    type_losses = []
    directions = {"H-P": (0, 1), "P-G": (1, 2), "G-U": (2, 3)}
    for pair_type in ("H-P", "P-G", "G-U"):
        selected = [(a, b) for (a, b), kind in zip(pair_indices, pair_types) if kind == pair_type]
        if not selected:
            continue
        a_class, b_class = directions[pair_type]
        values = []
        for a_index, b_index in selected:
            d_a = (logp[a_index, a_class] - logp[a_index, b_class]) / temperature
            d_b = (logp[b_index, b_class] - logp[b_index, a_class]) / temperature
            values.append(F.softplus(-(d_a + d_b) + margin))
        type_losses.append(torch.stack(values).mean())
    return torch.stack(type_losses).mean() if type_losses else logp.sum() * 0.0
