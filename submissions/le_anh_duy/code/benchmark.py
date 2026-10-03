"""benchmark.py - đo độ trễ suy luận đúng cách (GUIDE.md mục 4.1).

Quy ước đo: warmup 10 lần bỏ đi; torch.cuda.synchronize() trước và sau mỗi lần đo; >= 50 lần;
báo p50/p95/p99. CHỈ đo forward của model trên tensor đã nằm sẵn trên GPU (KHÔNG tính tiền xử lý,
đọc ảnh hay chép CPU -> GPU).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo `fn()` (mili-giây). `sync` = torch.cuda.synchronize trên GPU, None trên CPU."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "mean": float(np.mean(times)), "n": iters}


def latency_report(model, batch_size: int = 1, img_size: int = 224, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, sizes: list[int] | None = None, fused_bn: bool = False) -> dict:
    """Độ trễ một lượt suy luận với đầu vào ngẫu nhiên (batch_size, 3, s, s).

    `model` có thể là một model hoặc list model (ensemble: chạy lần lượt từng model).
    `sizes`: các độ phân giải chạy nối tiếp trong một lượt (TTA nhiều tỉ lệ); mặc định [img_size].
    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half(), làm trên bản sao).
    """
    device = torch.device(device)
    models = model if isinstance(model, (list, tuple)) else [model]
    sizes = sizes or [img_size]
    half = dtype == "fp16"
    if half:
        models = [copy.deepcopy(m).half() for m in models]
    for m in models:
        m.eval()
    xs = [torch.randn(batch_size, 3, s, s, device=device, dtype=torch.float16 if half else torch.float32)
          .contiguous(memory_format=torch.channels_last) for s in sizes]
    cuda = device.type == "cuda"

    def fn():
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=dtype == "amp" and cuda):
            for m in models:
                for x in xs:
                    m(x)

    r = bench(fn, warmup, iters, torch.cuda.synchronize if cuda else None)
    return {"gpu": torch.cuda.get_device_name(device) if cuda else "cpu", "dtype": dtype, "batch": batch_size,
            "img_size": img_size if len(set(sizes)) == 1 and sizes[0] == img_size else "+".join(map(str, sizes)),
            "n_models": len(models), "fused_bn": fused_bn,
            "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"], "iters": iters,
            "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__,
            "preprocessing": "không tính"}


def tta_latency(model, k_views: int, img_size: int = 224, **kw) -> dict:
    """Độ trễ TTA K view đo thật (K lượt forward nối tiếp), kèm tỉ lệ so với K x p50 của 1 view."""
    one = latency_report(model, img_size=img_size, **kw)
    r = latency_report(model, img_size=img_size, sizes=[img_size] * k_views, **kw)
    r["k_views"] = k_views
    r["ratio_vs_k_x_single"] = r["p50"] / (k_views * one["p50"])
    return r
