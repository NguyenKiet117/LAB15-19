# ANSWERS.md — Practice LAB Chapter 15–19

> Họ tên: Lê Nguyên Kiệt · MSSV: SE173346 · Ngày: 15/07/2026
> Quy tắc tự nhắc: mọi ô ghi `___` phải là SỐ TÔI TỰ CHẠY RA, không chép.

---

## Module 0 — Repo & Git

**Cây thư mục:** (dán output của `tree` hoặc `find . -type d`)

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

**git log --oneline:**

```
(dán vào đây)
```

**Phản biện M0:**
- Vì sao version code + data + model (hậu quả cụ thể nếu chỉ version code): Trong phần mềm thông thường, cùng một code sẽ cho ra cùng một kết quả. Nhưng với mô hình ML thì khác, kết quả của mô hình phụ thuộc vào code, dữ liệu và tham số huấn luyện. Cùng một file train.py nhưng huấn luyện trên dữ liệu tháng 1 và dữ liệu tháng 3 sẽ ra hai mô hình khác nhau. Nếu chỉ version code thôi, khi production bị lỗi checkout về commit cũ, chạy train lại nhưng dữ liệu đã bị ghi đè hoặc cập nhật mới nên không thể tạo ra đúng mô hình đang chạy trên production. Lúc đó sẽ không debug được lỗi cũng như không thể khôi phục về phiên bản cũ.

- Vì sao không commit .joblib/secrets; công cụ thay thế (DVC, Registry, ...): 
    + File mô hình thường rất lớn (dạng binary) và thay đổi sau mỗi lần huấn luyện. Nếu để vào git thông thường, repository sẽ phình to rất nhanh, đồng thời lệnh git diff cũng không xem được sự khác biệt một cách có ý nghĩa.
    + Bí mật (secrets) như API key, mật khẩu database… nếu vô tình commit vào git thì sẽ rò rỉ vĩnh viễn. Dù sau này bạn xóa file đi, lịch sử git vẫn còn lưu lại, ai đó vẫn có thể khai thác được.
    + Giải pháp thay thế phổ biến:
        1. Với dữ liệu và mô hình lớn: Dùng DVC, Git LFS, hoặc MLflow Model Registry.
        2. Với secrets: Dùng biến môi trường, GitHub Secrets, HashiCorp Vault,… thay vì commit trực tiếp vào code.  

---

## Module 1 — Pipeline & Leakage

**DỰ ĐOÁN TRƯỚC KHI CHẠY (bắt buộc, ghi trước khi có kết quả):**
- Tôi đoán ROC-AUC ≈ ___ vì: ...

**Kết quả thật (bản đúng — scale trong Pipeline):**

| Metric | Giá trị của tôi |
|---|---|
| Accuracy | ___ |
| F1 | ___ |
| ROC-AUC | ___ |
| CV 5-fold (mean ± std) | ___ ± ___ |

**Bản leakage (scale toàn bộ X trước split):**
- ROC-AUC leakage: ___
- Chênh lệch so với bản đúng: ___
- Nhận xét của tôi: ...

**Phản biện M1:** (trả lời bằng lời của mình, gắn số ở trên)
- Cơ chế Pipeline chống leakage khi CV: ...
- RandomForest có cần StandardScaler không, vì sao: ...
- Vì sao điểm leakage là điểm giả: ...

---

## Module 2 — MLflow

**3 cấu hình đã chạy:**

| Run | Model / Hyperparams | ROC-AUC | F1 |
|---|---|---|---|
| 1 | ___ | ___ | ___ |
| 2 | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ |

- Tên model đã đăng ký vào Registry: ___
- Run tốt nhất (run_id): ___
- Ảnh Compare: `screenshots/mlflow_compare.png`

**Phản biện M2:**
- Backend store vs artifact store lưu gì, vì sao tách: ...
- `models:/Name/Production` hơn load theo path ở điểm nào (rollback, A/B): ...
- MLflow log gì để reproducibility; thiếu gì vẫn không tái lập được: ...

---

## Module 3 — REST API

- File model đã serialize: `models/bc_pipeline_v___.joblib`
- Assert load lại khớp dự đoán: PASS / FAIL

**Log curl 3 mẫu (kèm nhãn thật):**

| Mẫu | Nhãn thật | Prediction | Confidence |
|---|---|---|---|
| 1 | ___ | ___ | ___ |
| 2 | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ |

- Ảnh /docs: `screenshots/swagger_docs.png`

**Săn lỗi (serve estimator thiếu scaler):**
- Hiện tượng tôi quan sát: ...
- Tên hiện tượng: ...

**Phản biện M3:**
- Training-serving skew là gì, vì sao âm thầm; quy tắc 1 câu: ...
- Chọn kiểu triển khai cho (a) chấm ảnh y khoa ban đêm, (b) chặn gian lận thẻ: ...
- Vì sao POST chứ không GET: ...

---

## Module 4 — Monitoring & Drift

**Bảng PSI tính tay (4 bin):**

| Bin | Expected % | Actual % | A−E | A/E | ln(A/E) | (A−E)×ln(A/E) |
|---|---|---|---|---|---|---|
| 1 | ___ | ___ | ___ | ___ | ___ | ___ |
| 2 | ___ | ___ | ___ | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ | ___ | ___ | ___ |
| 4 | ___ | ___ | ___ | ___ | ___ | ___ |
| **PSI** | | | | | | **___** |

- Hàm PSI code ra: ___ (khớp tính tay: CÓ / KHÔNG)

**DỰ ĐOÁN TRƯỚC KHI CHẠY 2 kịch bản:**
- Không-drift: tôi đoán PSI ≈ ___, KS p-value ___
- Có-drift: tôi đoán PSI ≈ ___, KS p-value ___

**Kết quả thật:**

| Kịch bản | PSI | KS statistic | p-value | Vượt ngưỡng 0.2? |
|---|---|---|---|---|
| Không-drift | ___ | ___ | ___ | ___ |
| Có-drift | ___ | ___ | ___ | ___ |

**3 dòng log dự đoán mẫu:**

```
(dán 3 dòng từ logs/predictions.log)
```

**Phản biện M4:**
- Data drift vs concept drift (ví dụ TMĐT VN của tôi): ...
- "Hỏng trong im lặng"; vì sao theo dõi phân phối đầu ra hữu ích khi nhãn trễ: ...
- Hai ngưỡng khác nhau cho cùng PSI ở hai bài toán: ...
- Alert kêu thì làm gì; retrain theo lịch vs trigger vs rollback: ...

---

## Module 5 — Testing & CI/CD

- Số test pass: ___ / ___ · Coverage: ___%

**Săn lỗi (test vô dụng):**
- 2 vấn đề của test "luôn pass": (1) ... (2) ...
- Cách tôi viết lại: ...

**Phản biện M5:**
- CI bắt lỗi gì; loại "hỏng" nào chỉ monitoring thấy: ...
- CI/CD vs CT; điều gì kích hoạt CT: ...
- Vì sao test phải tách train/test, không chấm trên dữ liệu đã fit: ...

---

## Module 6 — Tổng hợp & Maturity

**Bảng ghép vòng đời MLOps ↔ Module:**

| Giai đoạn vòng đời | Thành phần tôi đã làm | Module |
|---|---|---|
| Versioning | ___ | M0 |
| Training pipeline | ___ | M1 |
| Experiment tracking / Registry | ___ | M2 |
| Deployment | ___ | M3 |
| Monitoring | ___ | M4 |
| Testing / CI | ___ | M5 |

**Tự xếp maturity level:** Level ___ — bằng chứng: ...

**Bước lên level kế tiếp + lý do ưu tiên:** ...

**Tự đánh giá (5–7 câu):** ...

**Phản biện M6:**
- DevOps vs MLOps qua ví dụ của chính bài làm (Testing, Maintenance): ...
- Mắt xích yếu nhất về reproducibility: ...
