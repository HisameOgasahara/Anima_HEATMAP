"""모델의 Q/K를 변경하지 않고 attention 확률을 계산한다."""

import math

import torch
import numpy as np
from .profiling import record_count, profile_attention, transfer_to_cpu


@profile_attention
def compute_attention(query, key, *, query_indices=None, key_indices=None,
                      heads="mean", query_chunk=128, key_chunk=256,
                      mask=None, scale=None):
    """[B,H,Q,D], [B,H,K,D] -> CPU float32 [B,H 또는 1,Q선택,K선택].

    key 선택은 전체 key에 대한 softmax 이후에 적용한다.
    mask는 SDPA와 동일하게 True가 허용인 bool 또는 additive mask이다.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Q/K는 [batch, heads, tokens, head_dim]이어야 합니다.")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K의 batch, heads, head_dim이 일치해야 합니다.")
    if query_chunk <= 0 or key_chunk <= 0 or key.shape[2] == 0:
        raise ValueError("chunk 크기와 key 길이는 양수여야 합니다.")
    qids = list(range(query.shape[2])) if query_indices is None else list(query_indices)
    kids = list(range(key.shape[2])) if key_indices is None else list(key_indices)
    hids = list(range(query.shape[1])) if isinstance(heads, str) else list(heads)
    if isinstance(heads, str) and heads not in ("mean", "all"):
        raise ValueError("heads는 mean, all 또는 head 번호 목록입니다.")
    for ids, count in ((qids, query.shape[2]), (kids, key.shape[2]), (hids, query.shape[1])):
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= count:
            raise ValueError("선택 인덱스가 비었거나 중복되었거나 범위를 벗어났습니다.")
    factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    output = torch.empty(query.shape[0], 1 if heads == "mean" else len(hids),
                         len(qids), len(kids), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        selected_query = query.detach() if hids == list(range(query.shape[1])) else query.detach()[:, hids]
        k = (key.detach() if hids == list(range(key.shape[1])) else key.detach()[:, hids]).float()
        expanded_mask = None
        if mask is not None:
            expanded_mask = torch.broadcast_to(mask.to(query.device),
                (query.shape[0], query.shape[1], query.shape[2], key.shape[2]))[:, hids]

        def scores(q, positions, keys):
            logits = (q @ k[:, :, keys].transpose(-1, -2)) * factor
            if expanded_mask is not None:
                m = expanded_mask[:, :, positions][:, :, :, keys]
                logits = logits.masked_fill(~m, -torch.inf) if m.dtype == torch.bool else logits + m
            return logits

        def compute_full_keys(q, positions):
            logits = scores(q, positions, slice(None))
            blocked = torch.isneginf(logits).all(dim=-1, keepdim=True)
            probabilities = torch.softmax(logits, dim=-1)
            probabilities.masked_fill_(blocked, 0)
            if kids != list(range(k.shape[2])):
                probabilities = probabilities[:, :, :, kids]
            if heads == "mean":
                probabilities = probabilities.mean(dim=1, keepdim=True)
            return transfer_to_cpu(probabilities)

        split_keys = False
        for start in range(0, len(qids), query_chunk):
            positions = qids[start:start + query_chunk]
            q = selected_query[:, :, positions].float()
            if not split_keys:
                try:
                    values = compute_full_keys(q, positions)
                except torch.OutOfMemoryError:
                    record_count("oom_fallback")
                    # Retry with bounded key chunks for the rest of this record.
                    split_keys = True
                else:
                    output[:, :, start:start + len(positions)] = values
                    continue
            denominator = torch.full(q.shape[:-1], -torch.inf, device=q.device)
            for offset in range(0, k.shape[2], key_chunk):
                logits = scores(q, positions, slice(offset, offset + key_chunk))
                denominator = torch.logaddexp(denominator, torch.logsumexp(logits, dim=-1))
            for offset in range(0, len(kids), key_chunk):
                selected = kids[offset:offset + key_chunk]
                logits = scores(q, positions, selected)
                probabilities = torch.exp(logits - denominator[..., None])
                probabilities = torch.where(torch.isneginf(denominator[..., None]), 0, probabilities)
                if heads == "mean":
                    probabilities = probabilities.mean(dim=1, keepdim=True)
                output[:, :, start:start + len(positions), offset:offset + len(selected)] = transfer_to_cpu(probabilities)
    if not np.isfinite(output.numpy()).all():
        raise ValueError("attention에 NaN/Inf가 있습니다. Q/K와 mask를 확인하세요.")
    return output

