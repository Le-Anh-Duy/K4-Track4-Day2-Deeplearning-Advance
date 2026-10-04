"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", smoothing: float = 0.1, gamma: float = 2.0, alpha=None, weight=None):
    """kind: "ce" | "ls" (label smoothing) | "focal" | "ce_weighted" (cần `weight`)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing)
    if kind == "focal":
        return FocalLoss(gamma, alpha)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần weight (xem class_weights)")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """CE với nhãn q'(k) = (1 - eps) * 1[k == y] + eps / K.

    Dùng sẵn tham số label_smoothing của F.cross_entropy (đúng công thức trên); eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), trung bình theo batch. gamma = 0 -> CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp_t = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - logp_t.exp()).pow(self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của TRAIN, chuẩn hoá về trung bình 1 (tổng = số lớp).

    beta = 0: w_c ∝ 1 / n_c.   beta > 0: class-balanced w_c ∝ (1 - beta) / (1 - beta^n_c).
    """
    n = torch.as_tensor(np.asarray(counts, dtype=np.float64))
    w = 1.0 / n if beta == 0 else (1 - beta) / (1 - beta ** n)
    return (w / w.mean()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch: lam ~ Beta(alpha, alpha), cặp với hoán vị ngẫu nhiên của chính batch.

    cutmix: dán hộp từ x[perm] vào x; lam được tính lại theo DIỆN TÍCH THỰC của hộp sau khi cắt biên.
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x = x.clone()
        x[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam * L(logits, y_a) + (1 - lam) * L(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
