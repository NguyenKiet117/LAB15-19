# bc_mlops — Practice LAB Chapter 15–19

Dự án MLOps end-to-end trên dataset Breast Cancer (sklearn), chạy CPU.

## Cấu trúc

```
bc_mlops/
├── data/                  # (trống — data load trực tiếp từ sklearn)
├── src/                   # code train, pipeline, drift
├── app/                   # FastAPI app
├── tests/                 # pytest
├── models/                # model .joblib (KHÔNG commit — xem .gitignore)
├── logs/                  # prediction logs (M4)
├── config/                # config nếu cần
├── .github/workflows/     # ci.yml (M5)
├── ANSWERS.md             # ĐIỀN SỐ LIỆU CỦA BẠN vào đây
└── requirements.txt
```

## Checklist code TỰ VIẾT (theo đặc tả đề)

- [ ] `src/train.py` — M1: load, split (stratify, seed), Pipeline(scaler+clf),
      4 metrics + CV5; M2: bọc MLflow run, chạy ≥3 cấu hình.
      Hỗ trợ `--smoke` cho CI (mẫu nhỏ, không log MLflow).
- [ ] `src/leakage_demo.py` — M1 bước 4: scale toàn bộ X trước split, so sánh.
- [ ] `src/serialize.py` — M3 bước 1: joblib.dump cả pipeline (tên có version),
      load lại + assert khớp dự đoán.
- [ ] `app/main.py` — M3: POST /predict, GET /health, GET /model-info
      + M4 bước 4: log mỗi request (timestamp, features, prediction,
      confidence, model_version) vào logs/predictions.log.
- [ ] `src/drift.py` — M4: hàm psi(expected, actual), 2 kịch bản
      không-drift / có-drift, PSI + ks_2samp, ngưỡng 0.2.
- [ ] `tests/test_model.py` — M5: ≥4 test (shape, nhãn nhị phân,
      proba ∈ [0,1], NaN edge case) + phiên bản viết lại của test vô dụng.

## Chạy

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
mlflow ui                      # http://127.0.0.1:5000
uvicorn app.main:app --reload  # http://127.0.0.1:8000/docs
pytest --cov=src
```

## Nộp

- ANSWERS.md điền đủ số liệu thật
- 3 ảnh: MLflow Compare · Swagger /docs · báo cáo drift (để trong screenshots/)
