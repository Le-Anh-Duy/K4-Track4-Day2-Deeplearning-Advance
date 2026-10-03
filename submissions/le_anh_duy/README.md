# Lab Day 2 — DeepWeeds · Lê Anh Duy

> TODO trước khi nộp: đổi tên thư mục thành `<mssv>_le_anh_duy` (và `SUB_NAME` trong ô đầu notebook), dán link Colab, điền phiên bản thư viện từ `requirements-colab.txt`.

- Notebook Colab: `code/lab_day2.ipynb` — link: _TODO_
- GPU: Colab T4 · seed: 0 (Bước 1–3), 0/1/2 (T00 và chung kết F01)

## Thứ tự chạy

1. Push repo lên GitHub, mở `code/lab_day2.ipynb` trên Colab (Runtime: T4 GPU), *Run all*.
2. Ô đầu tiên clone repo, cài `timm openpyxl`, chạy CI trên CPU: test gốc của repo, `test_code.py` (focal γ=0 ≡ CE, CutMix, gộp BN, temperature, …) và `smoke_notebook.py` (chạy cả notebook trên dữ liệu giả).
3. Kết quả ghi vào Google Drive `MyDrive/deepweeds_lab/`. Phiên bị ngắt thì chạy lại từ đầu: run đã xong được đọc lại, run đang dở chạy tiếp từ `last.pt`.
4. Ô cuối chép `results.xlsx`, `curves/`, `predictions/`, `figures/` vào thư mục này và tải về `submission.zip`.

Một thí nghiệm lẻ từ dòng lệnh: `python code/train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=... labels_dir=...`

## Code (`code/`)

| File | Nội dung |
|---|---|
| `dataset.py` | đọc fold 0, `check_split` (S1–S4), transform/augmentation, cache ảnh uint8 (memmap), DataLoader + sampler cân bằng |
| `model.py` | backbone timm, đóng băng (BN giữ eval), nhóm tham số (không wd cho norm/bias/pos_embed), params, GMAC (FlopCounterMode) |
| `losses.py` | label smoothing, focal, trọng số lớp, Mixup/CutMix (λ theo diện tích thực) |
| `train.py` | `run(Config)` dùng chung: AMP, warmup + cosine theo bước, EMA, chọn checkpoint theo macro-F1 val, chạy tiếp được |
| `inference.py` | TTA (lật, 5 crop, đa tỉ lệ), gộp prob/logit, ensemble, temperature scaling, gộp Conv+BN |
| `benchmark.py` | độ trễ: warmup 10, `cuda.synchronize`, 100 lần, p50/p95/p99, chỉ tính forward |
| `experiments.py` | danh sách thí nghiệm B/T, `final_predict` (test đúng một lần mỗi seed), bảng Final/PerClass, ghi xlsx |
