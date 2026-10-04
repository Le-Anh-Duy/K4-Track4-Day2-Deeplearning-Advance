"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở chế độ eval, không gradient. Nhiệt độ T khớp trên VAL rồi mới áp dụng sang test.

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.transforms.v2 import functional as TF


def softmax(logits, axis: int = -1):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


@torch.inference_mode()
def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader (giữ thứ tự file). `view(x)` biến đổi batch trước khi vào model."""
    device = torch.device(device)
    model.eval()
    dtype = next(model.parameters()).dtype
    names, ys, outs = [], [], []
    for x, y, f in loader:
        x = x.to(device, dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
        if view is not None:
            x = view(x)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(x)
        outs.append(out.float().cpu())
        ys.append(y)
        names += list(f)
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; flip=True thêm 5 bản lật (10 view)."""
    crops = list(TF.five_crop(x, [crop, crop]))
    return crops + [view_hflip(c) for c in crops] if flip else crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`.

    Chỉ dùng cho CNN có global pooling; ViT/Swin của timm cố định kích thước đầu vào (lỗi nếu khác 224).
    """
    return [F.interpolate(x, size=(s, s), mode="bilinear", antialias=True, align_corners=False) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax."""
    z = np.stack(logits_per_view)
    if space == "prob":
        return softmax(z).mean(0)
    if space == "logit":
        return softmax(z.mean(0))
    raise ValueError(f"space không hợp lệ: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    return np.mean(np.stack(list_of_probs), axis=0)


def _nll(logits, labels, T):
    p = softmax(logits / T)
    return -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean()


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL, tìm trên log T trong [1/20, 20]. KHÔNG khớp T trên test."""
    z, y = np.asarray(val_logits, dtype=np.float64), np.asarray(val_labels)
    res = minimize_scalar(lambda t: _nll(z, y, np.exp(t)), bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits, T: float):
    """softmax(logits / T). Thứ tự lớp không đổi nên accuracy không đổi."""
    return softmax(np.asarray(logits, dtype=np.float64) / T)


def fuse_conv_bn(model):
    """Gộp mọi cặp (Conv2d, BatchNorm2d) liền kề trong cùng module cha, trả về BẢN SAO đã gộp:
        w' = gamma * w / sqrt(var + eps),   b' = beta + gamma * (b - mean) / sqrt(var + eps)

    BatchNormAct2d của timm (EfficientNet, MobileNetV3) = BN + drop + act: thay bằng drop + act.
    ConvNeXt/ViT/Swin dùng LayerNorm, không có cặp nào: trả về bản sao không đổi (fused = 0).
    In số cặp đã gộp; kiểm tra sai số bằng cách so đầu ra trước/sau (xem test_code.py).
    """
    model = copy.deepcopy(model).eval()
    n = 0
    for parent in model.modules():
        children = list(parent.named_children())
        for (_, conv), (bn_name, bn) in zip(children, children[1:]):
            if type(conv) is nn.Conv2d and isinstance(bn, nn.BatchNorm2d) and conv.out_channels == bn.num_features:
                fused = fuse_conv_bn_eval(conv, bn)
                conv.weight, conv.bias = fused.weight, fused.bias
                tail = [m for m in (getattr(bn, "drop", None), getattr(bn, "act", None)) if m is not None]
                setattr(parent, bn_name, nn.Sequential(*tail) if tail else nn.Identity())
                n += 1
    model.fused_pairs = n
    return model
