"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện (giữ đúng bộ khung starter/):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Thêm: build_cache(...) giải mã toàn bộ ảnh một lần vào một file .npy (memmap) trên đĩa cục bộ, để
Colab (2 vCPU) không bị nghẽn ở bước giải mã JPEG. Dataset đọc từ cache nếu được truyền `cache`.
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # cả 6 backbone timm dùng trong lab đều dùng mean/std ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
NATIVE_SIZE = 256
EXPECTED_RATIO = {"train": 0.6, "val": 0.2, "test": 0.2}


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv nguyên bản (S1). Không sửa, lọc hay chia lại."""
    d = Path(labels_dir)
    return tuple(pd.read_csv(d / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README mục 2.1). Vi phạm thì AssertionError."""
    sets = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    for k, v in sets.items():
        assert not v["Filename"].duplicated().any(), f"{k}: có Filename bị trùng"

    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    assert not any(overlap.values()), f"giao giữa các tập phải rỗng: {overlap}"

    union = set().union(*names.values())
    assert len(union) == TOTAL_IMAGES, f"hợp ba tập có {len(union)} ảnh, kỳ vọng {TOTAL_IMAGES}"

    missing = sorted(union - set(os.listdir(images_dir)))
    assert not missing, f"thiếu {len(missing)} ảnh trong {images_dir}, ví dụ {missing[:5]}"

    n = {k: len(v) for k, v in sets.items()}
    ratio = {k: n[k] / len(union) for k in n}
    off = {k: round(ratio[k] - EXPECTED_RATIO[k], 4) for k in n if abs(ratio[k] - EXPECTED_RATIO[k]) > 0.01}
    per_class = pd.DataFrame({k: v["Label"].value_counts() for k, v in sets.items()})
    per_class = per_class.reindex(range(NUM_CLASSES)).fillna(0).astype(int)
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)

    if verbose:
        print("Số ảnh:", n, "| tỉ lệ:", {k: round(r, 4) for k, r in ratio.items()})
        print("Giao:", overlap, "| hợp:", len(union), "| thiếu file: 0")
        print(per_class.to_string())
    if off:
        print(f"CẢNH BÁO: tỉ lệ lệch 60/20/20 hơn 1 điểm %: {off}. Báo giảng viên trước khi chạy tiếp.")
    return {"n": n, "ratio": ratio, "overlap": overlap, "union": len(union), "missing": 0,
            "per_class": per_class}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic", crop: bool = True):
    """Transform cho ảnh PIL hoặc mảng uint8 HWC (từ cache).

    Train: RandomResizedCrop(img_size) + lật ngang + [aug] + Normalize.
      aug: "basic" | "color" (ColorJitter) | "trivial" (TrivialAugmentWide) | "randaug" | "vflip"
      Không lật dọc trong "basic": ảnh chụp từ robot có hướng tương đối cố định; "vflip" để thử riêng.
    Val/test (crop=True): Resize(img_size / 0.875) rồi CenterCrop(img_size); 224 -> resize 256, crop 224.
    Val/test (crop=False): Resize cả ảnh về img_size (dùng cho dò độ phân giải và TTA nhiều crop).
    """
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        size = [v2.Resize(round(img_size / 0.875), antialias=True), v2.CenterCrop(img_size)] if crop \
            else [v2.Resize(img_size, antialias=True)]
        return v2.Compose([v2.ToImage(), *size, *norm])
    extra = {"basic": [], "color": [v2.ColorJitter(0.3, 0.3, 0.3, 0.05)], "trivial": [v2.TrivialAugmentWide()],
             "randaug": [v2.RandAugment()], "vflip": [v2.RandomVerticalFlip()]}
    if aug not in extra:
        raise ValueError(f"aug không hợp lệ: {aug}; chọn trong {sorted(extra)}")
    return v2.Compose([v2.ToImage(), v2.RandomResizedCrop(img_size, antialias=True), v2.RandomHorizontalFlip(),
                       *extra[aug], *norm])


def build_cache(images_dir: str | Path, files, path: str | Path, workers: int = 8) -> Path:
    """Giải mã mọi ảnh trong `files` vào <path> (uint8, N x 256 x 256 x 3, ~3,4 GB cho 17.509 ảnh).

    Chạy một lần; lần sau thấy file đã có thì bỏ qua. Ảnh khác 256x256 (nếu có) được resize về 256.
    """
    path = Path(path)
    names_path = path.with_suffix(".files.txt")
    if path.exists() and names_path.exists():
        return path
    files = list(files)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(files), NATIVE_SIZE, NATIVE_SIZE, 3))

    def load(k):
        img = Image.open(Path(images_dir) / files[k]).convert("RGB")
        if img.size != (NATIVE_SIZE, NATIVE_SIZE):
            img = img.resize((NATIVE_SIZE, NATIVE_SIZE), Image.BILINEAR)
        arr[k] = np.asarray(img)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(load, range(len(files))))
    arr.flush()
    del arr
    names_path.write_text("\n".join(files))
    tmp.replace(path)  # đổi tên cuối cùng = đánh dấu cache đã hoàn chỉnh
    return path


class DeepWeedsDataset(Dataset):
    """Ảnh theo DataFrame (Filename, Label). __getitem__ -> (tensor, nhãn int, tên file)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: str | Path | None = None):
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache = cache
        self._arr = None  # mở lười trong từng worker: pickle một memmap sẽ sao chép cả mảng

    def __len__(self) -> int:
        return len(self.files)

    def _image(self, i: int):
        if self.cache is None:
            return Image.open(self.images_dir / self.files[i]).convert("RGB")
        if self._arr is None:
            self._arr = np.load(self.cache, mmap_mode="r")
            order = Path(self.cache).with_suffix(".files.txt").read_text().split("\n")
            row = {f: k for k, f in enumerate(order)}
            self._rows = [row[f] for f in self.files]
        return np.array(self._arr[self._rows[i]])

    def __getitem__(self, i: int):
        x = self._image(i)
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i], self.files[i]


def _seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                cache: str | Path | None = None, seed: int = 0):
    """DataLoader. train=False giữ đúng thứ tự df (để ghép logit với Filename).

    sampler="balanced": WeightedRandomSampler trọng số 1 / (số ảnh của lớp) (trục D).
    """
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        w = 1.0 / df["Label"].map(df["Label"].value_counts()).to_numpy(dtype=np.float64)
        smp = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(df), replacement=True, generator=g)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform, cache), batch_size=batch_size,
                      shuffle=train and smp is None, sampler=smp, drop_last=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker, generator=g,
                      persistent_workers=num_workers > 0)
