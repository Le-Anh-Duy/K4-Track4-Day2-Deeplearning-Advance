# Lab Day 2 — DeepWeeds · Lê Anh Duy

MSSV: 2A202602723

- Notebook: `code/lab_day2.ipynb` · đã chạy trên **Kaggle, GPU Tesla T4** · [link Kaggle](https://www.kaggle.com/code/meowluvmatcha/aitc-lab-cv-2-deepweeds-backbone-training)
- Thư viện: python 3 (Kaggle), torch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.29, numpy 2.1.3, pandas 2.3.3, scipy 1.16.3 (`requirements-colab.txt`)
- Seed: 0 cho Bước 1–3; 0/1/2 cho mốc `T00` và chung kết `F01`

## Kết quả

| | macro-F1 test | top-1 test | ECE test | recall Chinee / Snake |
|---|---|---|---|---|
| **F01** (ConvNeXt-T + TrivialAug + LS + EMA, TTA 3 tỉ lệ + temperature) | **0,9807 ± 0,0018** | **0,9840 ± 0,0012** | 0,0039 | 95,7% / 95,9% |
| Mốc T00 + I00 | 0,9693 ± 0,0028 | 0,9760 ± 0,0016 | 0,0099 | 93,8% / 94,9% |

Tự chấm phần I (`eval.py grade`, `eval_out_copy/grade_I.json`): 20/20 (đề xuất). Phân tích đầy đủ ở [`report.md`](report.md), bảng ở `results.xlsx`.

| Thư mục/file | Nội dung |
|---|---|
| `report.md` | báo cáo kết luận |
| `results.xlsx` | Summary, Backbones, Training, Inference, Final, PerClass, Latency |
| `curves/` | biểu đồ training của mọi run B, T, F (mỗi seed một ảnh) |
| `figures/` | EDA, ảnh augmentation, overfit 1 batch, đánh đổi backbone/suy luận, ma trận nhầm lẫn, ảnh bị đoán sai |
| `predictions/` | dự đoán test (và val) của F01, F01uncal, T00, mỗi seed |
| `eval_out_copy/` | đầu ra `eval.py score/grade` |

## Thứ tự chạy

1. Kaggle (khuyên dùng): *File > Import Notebook* từ GitHub `code/lab_day2.ipynb`, Settings: GPU + Internet On, *Save & Run All (Commit)*. Hoặc Colab (T4 GPU), *Run all*.
2. Ô đầu tiên clone repo, cài `timm openpyxl`, chạy CI trên CPU: test gốc của repo, `test_code.py` (focal γ=0 ≡ CE, CutMix, gộp BN, temperature, …) và `smoke_notebook.py` (chạy cả notebook trên dữ liệu giả).
3. Kết quả ghi vào `/kaggle/working/deepweeds_lab` (Kaggle Output) hoặc Google Drive `MyDrive/deepweeds_lab/` (Colab). Phiên bị ngắt thì chạy lại từ đầu: run đã xong được đọc lại, run đang dở chạy tiếp từ `last.pt`.
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
