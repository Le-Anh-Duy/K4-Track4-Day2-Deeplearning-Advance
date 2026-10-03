"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F).

Mọi thí nghiệm đi qua MỘT hàm run(Config(...)); đổi thí nghiệm = đổi Config.
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Chạy lại được khi Colab bị ngắt: mỗi epoch lưu <run_dir>/last.pt; gọi lại run(cfg) sẽ chạy tiếp từ đó.
Run đã xong (có summary.json) thì run(cfg) trả về kết quả cũ, không train lại.
Macro-F1 val dùng để chọn checkpoint được tính bằng eval.compute_metrics của repo gốc.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import random
import sys
import time
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def _repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "eval.py").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py ở các thư mục cha")


sys.path.insert(0, str(_repo_root()))
from eval import compute_metrics, save_predictions  # noqa: E402

from benchmark import latency_report  # noqa: E402
from dataset import CLASS_NAMES, NUM_CLASSES, build_transforms, check_split, load_split, make_loader  # noqa: E402
from inference import predict_logits, softmax  # noqa: E402
from losses import build_criterion, class_weights, mix_batch, mixed_loss  # noqa: E402
from model import build_model, count_gmacs, count_params, param_groups, set_train_mode, weights_tag  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng trong tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    optimizer: str = "adamw"          # adamw | sgd (trục E)
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache: str | None = None          # file .npy từ dataset.build_cache (None = đọc JPEG trực tiếp)
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curves_path(cfg: Config) -> Path:
    name = "_".join(s for s in (cfg.exp_id, cfg.desc) if s)
    return Path(cfg.curves_dir) / (f"{name}.png" if cfg.seed == 0 else f"{name}_seed{cfg.seed}.png")


def set_seed(seed: int) -> None:
    """Cố định random/numpy/torch. cudnn.benchmark=True để nhanh hơn, đổi lại kết quả không tất định
    tuyệt đối (lệch nhỏ giữa hai lần chạy cùng seed); ghi điều này vào báo cáo."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD momentum 0.9) trên các nhóm tham số của model.param_groups."""
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Theo từng bước: warmup tuyến tính warmup_epochs epoch, rồi cosine về 0 ở bước cuối."""
    total = cfg.epochs * steps_per_epoch
    warm = int(cfg.warmup_epochs * steps_per_epoch)

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        t = min((step - warm) / max(1, total - warm), 1.0)
        return 0.5 * (1 + math.cos(math.pi * t))

    return LambdaLR(optimizer, factor)


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W sau mỗi bước tối ưu, trên một bản sao riêng (self.module).

    Buffer số thực (running_mean/var của BN) cũng được trung bình động như trọng số (giống timm ModelEmaV2);
    buffer số nguyên (num_batches_tracked) được chép thẳng. Đánh giá val bằng self.module.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.lerp_(m, 1 - self.decay)
            else:
                e.copy_(m)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch. Trả về {"train_loss": ..., "lrs": [LR head theo từng bước]}."""
    set_train_mode(model, cfg.init == "frozen")
    total, n, lrs = 0.0, 0, []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            logits = model(x)
            loss = mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()}; thử giảm LR hoặc tắt AMP")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        lrs.append(optimizer.param_groups[-1]["lr"])
        total += loss.item() * len(y)
        n += len(y)
    return {"train_loss": total / n, "lrs": lrs}


def evaluate(model, loader, criterion, device, amp: bool = True):
    """Chạy model trên loader ở chế độ eval, không gradient.

    Trả về (filenames, y_true[N], logits[N, 9], loss). `criterion` tính trên CPU; run() dùng CE thường
    cho loss val để mọi thí nghiệm (kể cả focal/weighted) so được với nhau.
    """
    names, y, logits = predict_logits(model, loader, device, amp=amp)
    loss = float(criterion(torch.from_numpy(logits), torch.from_numpy(y)))
    return names, y, logits, loss


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Loss train/val, macro-F1/top-1 val theo epoch, LR theo bước -> một file png."""
    h = pd.DataFrame(history)
    best = h.loc[h["val_macro_f1"].idxmax(), "epoch"]
    fig = Figure(figsize=(16, 4.2))
    ax = fig.subplots(1, 3)
    ax[0].plot(h["epoch"], h["train_loss"], "o-", label="train (loss huấn luyện)")
    ax[0].plot(h["epoch"], h["val_loss"], "o-", label="val (CE)")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    ax[1].plot(h["epoch"], h["val_macro_f1"], "o-", label="macro-F1 val")
    ax[1].plot(h["epoch"], h["val_top1"], "s--", label="top-1 val")
    if "val_macro_f1_raw" in h:
        ax[1].plot(h["epoch"], h["val_macro_f1_raw"], "^:", label="macro-F1 val (không EMA)")
    ax[1].axvline(best, color="grey", ls=":", label=f"checkpoint chọn (epoch {best})")
    ax[1].set(title="Chỉ số val", xlabel="epoch", ylabel="giá trị")
    if lr_steps:
        ax[2].plot(np.arange(1, len(lr_steps) + 1), lr_steps, label="LR head")
    ax[2].set(title="Lịch LR (warmup + cosine)", xlabel="bước", ylabel="LR")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def load_config(runs: str | Path, exp_id: str, seed: int = 0, /, **overrides) -> Config:
    """Đọc lại Config của một lần chạy (runs/<exp_id>/seed<k>/config.json); overrides ghi đè trường
    (ví dụ đường dẫn khi chạy ở phiên Colab khác)."""
    d = json.loads((Path(runs) / exp_id / f"seed{seed}" / "config.json").read_text())
    return Config(**{**d, **overrides})


def load_model(cfg: Config, device="cuda"):
    """Dựng lại kiến trúc (không tải ImageNet) và nạp checkpoint tốt nhất best.pt."""
    model = build_model(cfg.backbone, pretrained=False, drop_rate=cfg.drop_rate, init=cfg.init)
    model.load_state_dict(torch.load(run_dir(cfg) / "best.pt", map_location="cpu"))
    return model.to(device).eval().to(memory_format=torch.channels_last)


def predict_test(cfg: Config, model=None, device="cuda") -> Path:
    """1-view trên TEST, ĐÚNG MỘT LẦN cho mỗi (exp_id, seed): file đã có thì không chạy lại."""
    path = pred_path(cfg, "test")
    if path.exists():
        print(f"{path} đã có: không chạy lại test (quy tắc test một lần mỗi seed).")
        return path
    model = model or load_model(cfg, device)
    _, _, test_df = load_split(cfg.labels_dir, cfg.fold)
    loader = make_loader(test_df, cfg.images_dir, build_transforms(False, cfg.img_size), cfg.batch_size * 2,
                         False, None, cfg.num_workers, cfg.cache)
    names, y, logits = predict_logits(model, loader, device, amp=cfg.amp)
    np.save(run_dir(cfg) / "test_logits.npy", logits)
    return save_predictions(path, names, y, softmax(logits))


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình, lưu checkpoint/log/logit/ảnh, trả về dict tóm tắt."""
    rd = run_dir(cfg)
    done = rd / "summary.json"
    if done.exists():
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong, đọc lại {done}")
        summary = json.loads(done.read_text())
        if cfg.save_test_predictions:
            predict_test(cfg)
        return summary
    rd.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    (rd / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                               cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.cache, cfg.seed)
    val_loader = make_loader(val_df, cfg.images_dir, build_transforms(False, cfg.img_size), cfg.batch_size * 2,
                             False, None, cfg.num_workers, cfg.cache)

    model = build_model(cfg.backbone, pretrained=True, drop_rate=cfg.drop_rate, init=cfg.init)
    tag = "scratch (không tiền huấn luyện)" if cfg.init == "scratch" else weights_tag(model)
    model = model.to(device).to(memory_format=torch.channels_last)
    weight = None
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"], minlength=NUM_CLASSES)  # chỉ dùng số liệu TRAIN
        weight = class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                weight=weight).to(device)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history, lr_steps, best_f1, start = [], [], -1.0, 0
    last = rd / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None:
            ema.module.load_state_dict(ck["ema"])
        history, lr_steps, best_f1, start = ck["history"], ck["lr_steps"], ck["best_f1"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"])
        np.random.set_state(ck["rng_np"])
        print(f"[{cfg.exp_id} seed{cfg.seed}] chạy tiếp từ epoch {start + 1}")

    for epoch in range(start, cfg.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        train_time = time.time() - t0
        eval_model = ema.module if ema is not None else model
        _, y, logits, val_loss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
        m = compute_metrics(y, logits.argmax(1), softmax(logits))
        row = {"epoch": epoch + 1, "train_loss": tr["train_loss"], "val_loss": val_loss,
               "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
               "lr_head_end": tr["lrs"][-1], "train_time_s": train_time}
        if ema is not None:
            _, y_raw, logits_raw, _ = evaluate(model, val_loader, val_criterion, device, cfg.amp)
            row["val_macro_f1_raw"] = compute_metrics(y_raw, logits_raw.argmax(1), softmax(logits_raw))["macro_f1"]
        history.append(row)
        lr_steps += tr["lrs"]
        if m["macro_f1"] > best_f1:  # hoà thì giữ epoch sớm hơn
            best_f1 = m["macro_f1"]
            torch.save(eval_model.state_dict(), rd / "best.pt")
            np.save(rd / "val_logits.npy", logits)
        print(f"[{cfg.exp_id} seed{cfg.seed}] epoch {epoch + 1}/{cfg.epochs} "
              f"train_loss {tr['train_loss']:.4f} val_loss {val_loss:.4f} "
              f"F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} ({train_time:.0f}s)")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.module.state_dict() if ema is not None else None,
                    "history": history, "lr_steps": lr_steps, "best_f1": best_f1, "epoch": epoch,
                    "rng_cpu": torch.get_rng_state(), "rng_np": np.random.get_state()}, last)

    del model, optimizer, ema
    gc.collect()
    best_model = load_model(cfg, device)
    val_logits = np.load(rd / "val_logits.npy")
    val_probs = softmax(val_logits)
    save_predictions(pred_path(cfg, "val"), val_df["Filename"], val_df["Label"], val_probs)
    if cfg.save_test_predictions:
        predict_test(cfg, best_model, device)

    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_steps))
    plot_curves(history, curves_path(cfg), f"{cfg.exp_id} · {cfg.backbone} · {cfg.desc} · seed {cfg.seed}", lr_steps)

    h = pd.DataFrame(history)
    best_row = h.loc[h["val_macro_f1"].idxmax()]
    vm = compute_metrics(val_df["Label"].to_numpy(), val_probs.argmax(1), val_probs)
    lat = latency_report(best_model, 1, cfg.img_size, "fp32", "cuda", iters=50)["p50"] if device.type == "cuda" else None
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "desc": cfg.desc, "backbone": cfg.backbone, "weights_tag": tag,
        "params_m": count_params(best_model), "gmacs": count_gmacs(best_model, cfg.img_size),
        "img_size": cfg.img_size, "epochs": cfg.epochs, "best_epoch": int(best_row["epoch"]),
        "val_macro_f1": vm["macro_f1"], "val_top1": vm["top1"], "val_balanced_acc": vm["balanced_acc"],
        "val_ece": vm["ece"], **{f"val_f1_{c}": float(vm["f1"][i]) for i, c in enumerate(CLASS_NAMES)},
        "val_macro_f1_raw_at_best": float(best_row.get("val_macro_f1_raw", np.nan)),
        "train_time_per_epoch_s": float(h["train_time_s"].mean()), "latency_b1_fp32_p50_ms": lat,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "timm": timm.__version__, "python": platform.python_version(),
        "config": asdict(cfg),
    }
    done.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    last.unlink(missing_ok=True)  # đã xong: bỏ checkpoint trung gian, giữ best.pt
    del best_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _cast(raw: str, tp):
    if raw.lower() in ("none", "null"):
        return None
    t = next((a for a in typing.get_args(tp) if a is not type(None)), tp)
    if t is bool:
        if raw.lower() in ("1", "true", "yes"):
            return True
        if raw.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"không hiểu giá trị bool: {raw}")
    return t(raw)


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    hints = typing.get_type_hints(Config)
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or key not in hints:
            raise ValueError(f"'{pair}' không hợp lệ; key phải thuộc {sorted(hints)}")
        out[key] = _cast(raw, hints[key])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
