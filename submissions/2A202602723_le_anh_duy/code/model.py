"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
from torch.utils.flop_counter import FlopCounterMode

# Ghi lại tag trọng số thực tế bằng weights_tag(model) (timm.list_pretrained("resnet50*") để xem các tag).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}
INITS = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """init: "scratch" (không tải trọng số) | "frozen" (chỉ train head) | "finetune" (train toàn bộ).

    timm tự thay head mới (khởi tạo ngẫu nhiên) theo num_classes.
    """
    if init not in INITS:
        raise ValueError(f"init không hợp lệ: {init}; chọn trong {INITS}")
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def weights_tag(model) -> str:
    """Tên trọng số timm đầy đủ, ví dụ 'resnet50.a1_in1k'."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return f"{cfg.get('architecture', '?')}.{cfg.get('tag', '')}".rstrip(".")


def _head_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head. BN của backbone giữ ở eval: xem set_train_mode."""
    head = _head_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head


def set_train_mode(model, frozen: bool) -> None:
    """model.train(), trừ khi backbone đóng băng: khi đó cả backbone ở eval để BatchNorm không
    cập nhật running_mean/var bằng thống kê DeepWeeds (trọng số BN đứng yên, thống kê thì không)."""
    model.train()
    if frozen:
        model.eval()
        model.get_classifier().train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Nhóm tham số như slide trang 52: backbone (có wd), norm/bias (wd=0), head mới (LR riêng).

    Bias của head cũng không có wd. Tham số timm khai báo ở model.no_weight_decay()
    (pos_embed, cls_token, bảng relative position của Swin) được xếp vào nhóm wd=0.
    """
    head = _head_ids(model)
    no_wd_names = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    groups = {"backbone": [], "backbone_no_wd": [], "head": [], "head_no_wd": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1 or any(k in name for k in no_wd_names)
        groups[("head" if id(p) in head else "backbone") + ("_no_wd" if no_wd else "")].append(p)
    lr = {"backbone": lr_backbone, "head": lr_head}
    return [{"name": k, "params": v, "lr": lr[k.split("_")[0]],
             "weight_decay": 0.0 if k.endswith("no_wd") else weight_decay}
            for k, v in groups.items() if v]


def count_params(model) -> float:
    """Số tham số (triệu), kể cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm conv/matmul/attention, FLOPs = 2 x MAC),
    nên GMAC = FLOPs / 2 / 1e9. Bỏ qua phép theo phần tử (BN, activation), lệch vài % so với fvcore.
    """
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
