"""Chạy thử TOÀN BỘ notebook lab_day2.ipynb trên CPU trong < 30 giây (CI trước khi tốn GPU):
dữ liệu giả 9 lớp x 10 ảnh, model timm `test_*` (~0,1-0,4M tham số, KHÔNG tải trọng số), 1 epoch.
Bỏ qua các ô chỉ dành cho Colab (clone, Drive, tải dữ liệu, overfit ResNet-50, tải file zip).

    python smoke_notebook.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
from PIL import Image  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import benchmark  # noqa: E402
import dataset  # noqa: E402
import experiments  # noqa: E402

t0 = time.time()
plt.show = lambda *a, **k: plt.close("all")
_create = timm.create_model
timm.create_model = lambda name, pretrained=False, **kw: _create(name, pretrained=False, **kw)
_lat = benchmark.latency_report


def _fast_latency(model, batch_size=1, img_size=224, dtype="fp32", device=None, warmup=0, iters=0, sizes=None,
                  fused_bn=False):
    # Chỉ kiểm tra luồng dữ liệu: 1 lượt FP32 batch 1 trên CPU (benchmark có test riêng trong test_code.py)
    r = _lat(model, 1, img_size, "fp32", "cpu", warmup=0, iters=1, sizes=sizes, fused_bn=fused_bn)
    return {**r, "dtype": dtype, "batch": batch_size}


benchmark.latency_report = _fast_latency
experiments.BACKBONES = [("B01", "test_resnet", "resnet"), ("B02", "test_convnext", "convnext"),
                         ("B03", "test_vit", "vit"), ("B04", "test_efficientnet", "effnet")]
# Tập con T vẫn phủ mọi nhánh code: frozen, CutMix, CE trọng số, sampler cân bằng, EMA (I06 cần T10)
experiments.TRAINING = [t for t in experiments.TRAINING if t[0] in ("T00", "T02", "T05", "T08", "T09", "T10")]

d = Path(tempfile.mkdtemp())
(d / "images").mkdir()
(d / "labels").mkdir()
rng = np.random.default_rng(0)
rows = []
for c in range(9):
    for i in range(10):
        f = f"c{c}_{i}.jpg"
        Image.fromarray(np.clip(rng.normal(25 * c, 30, (256, 256, 3)), 0, 255).astype(np.uint8)).save(d / "images" / f)
        rows.append((f, c, dataset.CLASS_NAMES[c]))
df = pd.DataFrame(rows, columns=["Filename", "Label", "Species"]).sample(frac=1, random_state=0)
n = len(df)
parts = {"train": df[: int(.6 * n)], "val": df[int(.6 * n): int(.8 * n)], "test": df[int(.8 * n):]}
for k, p in parts.items():
    p.to_csv(d / "labels" / f"{k}_subset0.csv", index=False)
df.sort_values("Label").to_csv(d / "labels" / "labels.csv", index=False)
dataset.TOTAL_IMAGES = n

g = {"display": lambda *a: None, "__name__": "__main__", "subprocess": subprocess}
exec(f"""
import os
REPO_DIR = r"{CODE.parents[2]}"; SUB_DIR = r"{d / 'sub'}"; CODE_DIR = r"{CODE}"; ART = r"{d / 'art'}"
RUNS, PRED, CURVES, EVAL_OUT, FIGS = (f"{{ART}}/{{x}}" for x in ("runs", "predictions", "curves", "eval_out", "figures"))
for x in (RUNS, PRED, CURVES, EVAL_OUT, FIGS): os.makedirs(x, exist_ok=True)
LOCAL = DATA = r"{d}"; LABELS_DIR = r"{d / 'labels'}"; IMAGES_DIR = r"{d / 'images'}"
import sys, math, copy, json, shutil, numpy as np, pandas as pd, torch, timm, torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
""", g)

cells = ["".join(c["source"]) for c in json.load(open(CODE / "lab_day2.ipynb", encoding="utf-8"))["cells"]
         if c["cell_type"] == "code"]
start = next(i for i, s in enumerate(cells) if s.startswith("import dataset"))
for i, src in enumerate(cells[start:], start):
    if "overfit" in src or "files.download" in src:
        continue
    src = (src.replace('DEVICE = "cuda"', 'DEVICE = "cpu"').replace('device="cuda"', 'device="cpu"')
           .replace("num_workers=min(4, os.cpu_count())", "num_workers=0").replace("num_workers=2", "num_workers=0")
           .replace("epochs=12)", "epochs=1, batch_size=16, amp=False)"))
    lines = []
    for ln in src.splitlines():
        m = re.match(r"^(\s*)[!%](.*)$", ln)
        if m and "eval.py" in m.group(2):  # chạy eval.py thật; các lệnh shell khác bỏ qua
            cmd = m.group(2).replace("python ", f'"{sys.executable}" ', 1)
            lines.append(f"{m.group(1)}subprocess.run(rf'''{cmd}''', shell=True, check=True, stdout=subprocess.DEVNULL)")
        else:
            lines.append(f"{m.group(1)}pass" if m else ln)
    print(f"ô {i} ({time.time() - t0:.1f}s)", flush=True)
    exec("\n".join(lines), g)

assert set(pd.read_excel(f"{d}/art/results.xlsx", sheet_name=None)) == \
    {"Summary", "Backbones", "Training", "Inference", "Final", "PerClass", "Latency"}
print(f"SMOKE OK sau {time.time() - t0:.1f}s | F01 = {g['FINAL_KW']['backbone']} + {g['FINAL_METHOD']} | thư mục {d}")
