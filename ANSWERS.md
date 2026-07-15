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
- Tôi đoán ROC-AUC ≈ 0.98 vì: Breast Cancer là dataset dễ, tuyến tính tách tốt

**Kết quả thật (bản đúng — scale trong Pipeline):**

| Metric | Giá trị của tôi |
|---|---|
| Accuracy | 0.9825 |
| F1 | 0.9861 |
| ROC-AUC | 0.9954 |
| CV 5-fold (mean ± std) | 0.9935 ± 0.0108 |

**Bản leakage (scale toàn bộ X trước split):**
- ROC-AUC leakage: 0.9954
- Chênh lệch so với bản đúng: +0.0000
- Nhận xét của tôi: dataset nhỏ, sạch, phân phối train/test giống nhau nên leakage qua scaler ít lộ; nhưng trên dữ liệu thật (phân phối lệch, có outlier, có time-series) leakage sẽ thổi phồng điểm rõ rệt.

**Phản biện M1:** (trả lời bằng lời của mình, gắn số ở trên)
- Cơ chế Pipeline chống leakage khi CV: Trong cross validation, khi dùng pipeline thì ở mỗi fold, sklearn chỉ fit scaler (tính mean và std) trên phần train của fold đó, sau đó mới dùng scaler này để transform cả train và validation. Nếu bạn scale dữ liệu trước khi đưa vào cross_val_score (không dùng pipeline), mean/std sẽ được tính trên cả validation fold → làm rò rỉ thông tin của tập đánh giá vào bước tiền xử lý, dẫn đến kết quả đánh giá không còn chính xác.
- RandomForest có cần StandardScaler không, vì sao: Không ảnh hưởng nhiều đến chất lượng mô hình. Cây quyết định chia dữ liệu theo ngưỡng trên từng đặc trưng (ví dụ: radius > 14.5?), nên việc scale dữ liệu chỉ thay đổi con số ngưỡng chứ không thay đổi cách chia nhánh, kết quả mô hình gần như giống nhau. Scaler chỉ quan trọng với các mô hình dựa trên khoảng cách hoặc gradient như Logistic Regression, SVM, KNN hay mạng nơ-ron.
- Vì sao điểm leakage là điểm giả: Vì mô hình đã vô tình “nhìn trộm” thông tin của tập test trong bước tiền xử lý nên kết quả đánh giá bị lạc quan hơn thực tế. Khi đưa mô hình ra chạy trên dữ liệu thật (dữ liệu mới hoàn toàn chưa từng thấy), hiệu suất sẽ giảm mạnh xuống mức đúng, khiến bạn không giữ được con số đã hứa với business.

---

## Module 2 — MLflow

**4 cấu hình đã chạy (số từ mlflow.db của tôi):**

| Run | Model / Hyperparams | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| logreg_C1 | LogisticRegression, C=1.0 | 0.9825 | 0.9861 | 0.9954 |
| logreg_C001 | LogisticRegression, C=0.01 | 0.9474 | 0.9595 | 0.9950 |
| rf_200 | RandomForest, n_estimators=200 | 0.9561 | 0.9655 | 0.9932 |
| svc_rbf | SVC kernel=rbf | 0.9825 | 0.9861 | 0.9950 |

- Tên model đã đăng ký vào Registry: `bc_classifier` (version 1, stage **Production**)
- Run tốt nhất: `logreg_C1` — run_id `83490a4ae9de47eea14a385fbee013af` — ROC-AUC = 0.9954
- Ảnh Compare: `screenshots/mlflow_compare.png` 

**Phản biện M2:** 
- **Backend store vs artifact store:** Metric và parameter như roc_auc là dữ liệu nhỏ, cần tra cứu và so sánh nhanh nên MLflow lưu trong backend store (ở đây là file SQLite mlflow.db). Còn thư mục model chứa file lớn nên được lưu riêng trong artifact store (thư mục mlruns/, sau này có thể chuyển sang S3 hoặc GCS). Việc tách riêng giúp hệ thống chạy mượt: dù có hàng nghìn lần chạy thử nghiệm vẫn tra cứu nhanh, trong khi các file model dung lượng lớn vẫn lưu rẻ và không làm nặng database.
- **`models:/bc_classifier/Production` hơn load theo path:** Đường dẫn file model là hard-code vào một version cụ thể nên rất cứng nhắc. Trong khi đó, URI Registry chỉ là con trỏ logic — code serving của bạn không cần thay đổi, chỉ cần thay đổi “Production đang trỏ vào version nào” trên Model Registry. Rollback rất dễ: chỉ cần chuyển Production trỏ về version cũ là xong, không cần sửa code hay redeploy lại. Còn A/B testing thì có thể để hai service cùng lúc trỏ vào hai version khác nhau để so sánh hiệu suất.
- **MLflow log gì để reproducibility:** Các thông tin MLflow lưu lại gồm: parameters, metrics, git commit, môi trường (file conda.yaml hoặc requirements.txt), model artifact, seed… **Thiếu thứ nào thì vẫn không tái lập được kết quả?** nếu thiếu version của dữ liệu thì vẫn không thể tái tạo lại mô hình giống hệt. Vì MLflow không tự snapshot (lưu bản sao) dữ liệu. Dữ liệu thay đổi một chút thôi thì dù code và tham số giống nhau, mô hình vẫn ra khác.

---

## Module 3 — REST API

- File model đã serialize: `models/bc_pipeline_v1.0.0.joblib`
- Assert load lại khớp dự đoán: **PASS** (predict và predict_proba khớp 100%)

**Log curl 3 mẫu (kèm nhãn thật):**

| Mẫu (index) | Nhãn thật | Prediction | Confidence |
|---|---|---|---|
| 0 | 0 (malignant) | 0 (malignant) ✅ | 1.0000 |
| 100 | 0 (malignant) | 0 (malignant) ✅ | 0.9560 |
| 550 | 1 (benign) | 1 (benign) ✅ | 1.0000 |

- Ảnh /docs: `screenshots/swagger_docs.png` 
- Lệnh curl mẫu: chạy `python src/make_curl.py` để in sẵn 3 lệnh.

**Săn lỗi (serve estimator thiếu scaler — `python src/skew_demo.py`):**
- Hiện tượng tôi quan sát: accuracy sập từ **0.9825 → 0.3684**; mô hình dự đoán **114/114 mẫu test đều là class 0** (nhãn thật là 42 class 0 / 72 class 1). Không có exception nào — API vẫn trả kết quả "bình thường".
- Tên hiện tượng: **training-serving skew**.

**Phản biện M3:** 
- **Training-serving skew:** Khi đưa mô hình vào phục vụ (serve), dữ liệu đầu vào không đi qua đúng các bước tiền xử lý như lúc huấn luyện (ví dụ: lúc train dùng dữ liệu đã scale, nhưng lúc serve lại dùng dữ liệu thô). Lỗi này rất khó phát hiện vì đầu vào vẫn đúng shape và kiểu dữ liệu nên không báo lỗi, nhưng chất lượng dự đoán sẽ kém đi.
     + Quy tắc quan trọng: Luôn serialize (lưu) và deploy toàn bộ pipeline (các bước tiền xử lý + mô hình) như một khối duy nhất.
- **(a) chấm ảnh y khoa hàng loạt ban đêm → batch** (khối lượng lớn, không cần độ trễ thấp, chạy theo lịch); **(b) chặn gian lận thẻ → real-time** (phải quyết định trong mili-giây ngay lúc giao dịch).
- **POST vì:** Đầu vào là 30 số thực được gửi dưới dạng JSON có cấu trúc. Nếu dùng phương thức GET thì phải nhét hết vào query string, dẫn đến URL quá dài, dễ bị giới hạn, lộ thông tin trong URL và log, lại còn khó viết JSON. Vì vậy nên dùng POST để gửi dữ liệu trong body sẽ tốt và an toàn hơn.

---

## Module 4 — Monitoring & Drift

**Bảng PSI tính tay (4 bin):**

| Bin | Expected % | Actual % | A−E | A/E | ln(A/E) | (A−E)×ln(A/E) |
|---|---|---|---|---|---|---|
| 1 | 0.25 | 0.10 | −0.15 | 0.40 | −0.9163 | 0.1374 |
| 2 | 0.25 | 0.20 | −0.05 | 0.80 | −0.2231 | 0.0112 |
| 3 | 0.25 | 0.30 | +0.05 | 1.20 | +0.1823 | 0.0091 |
| 4 | 0.25 | 0.40 | +0.15 | 1.60 | +0.4700 | 0.0705 |
| **PSI** | | | | | | **0.2282** → ≥ 0.2 ⇒ DRIFT |

- Hàm PSI code ra: **0.2282** (khớp tính tay: **CÓ**)

**DỰ ĐOÁN TRƯỚC KHI CHẠY 2 kịch bản:** 
- Không-drift: tôi đoán PSI < 0.1 (cùng phân phối, chỉ lệch do lấy mẫu), KS p-value lớn (> 0.05)
- Có-drift: tôi đoán PSI vượt 0.2 rõ rệt (dịch cả phân phối +1σ), KS p-value ≈ 0

**Kết quả thật (`python src/drift.py`):**

| Kịch bản | PSI | KS statistic | p-value | Vượt ngưỡng 0.2? |
|---|---|---|---|---|
| Không-drift | 0.0395 | 0.0700 | 0.438 | KHÔNG |
| Có-drift | 2.3352 | 0.4992 | 3.4e-34 | **CÓ → ALERT** |

**3 dòng log dự đoán mẫu (`logs/predictions.log`, features rút gọn):**

```
{"timestamp": "2026-07-15T07:56:11.360340", "features": [17.99, 10.38, 122.8, ...], "prediction": 0, "confidence": 0.99999999, "model_version": "1.0.0"}
{"timestamp": "2026-07-15T07:56:11.367587", "features": [13.61, 24.98, 88.05, ...], "prediction": 0, "confidence": 0.95600415, "model_version": "1.0.0"}
{"timestamp": "2026-07-15T07:56:11.370193", "features": [10.86, 21.48, 68.51, ...], "prediction": 1, "confidence": 0.99997963, "model_version": "1.0.0"}
```

**Phản biện M4:** 
- **Data drift vs concept drift:** Data drift là khi đặc điểm đầu vào (X) của dữ liệu thay đổi, nhưng mối quan hệ giữa X và nhãn (y) vẫn giữ nguyên. Ví dụ: mô hình phát hiện review giả trên Shopee, sau đợt sale lớn, review từ tài khoản mới tăng mạnh và ngắn hơn → phân phối đặc trưng thay đổi. Còn Concept drift là khi chính mối quan hệ giữa X và y thay đổi. Ví dụ: kẻ đăng review giả chuyển sang viết review dài, tự nhiên, kèm ảnh → dù đặc trưng giống cũ nhưng nhãn “giả” hay “thật” đã khác. **Data drift phát hiện được KHÔNG cần nhãn** (chỉ so phân phối đầu vào bằng PSI/KS).
- **Hỏng trong im lặng:** Mô hình luôn trả về một dự đoán, không crash và không báo lỗi HTTP (như trong skew_demo M3, accuracy chỉ còn 0.37 nhưng API vẫn trả về mã 200 bình thường). Chất lượng thật chỉ biết được khi có nhãn thật, mà nhãn thường về muộn sau vài tuần hoặc vài tháng. Vì vậy, việc theo dõi phân phối đầu ra (output) rất hữu ích: nếu tỷ lệ dự đoán một lớp nào đó tăng đột ngột (ví dụ nhảy lên 100% class 0), đó là dấu hiệu cảnh báo sớm, giúp phát hiện vấn đề mà không cần chờ nhãn thật.
- **Hai ngưỡng cho cùng PSI:** Dù cùng chỉ số PSI = 0.15, nhưng ngưỡng cảnh báo lại khác nhau tùy mô hình. Ví dụ: với mô hình gợi ý sản phẩm, sai lệch này chỉ làm CTR giảm nhẹ nên có thể đặt ngưỡng cao hơn (0.25). Còn với mô hình duyệt tín dụng, sai lệch đó có thể gây mất tiền và rủi ro pháp lý nên phải đặt ngưỡng nghiêm ngặt hơn (0.1). Chi phí của việc sai khác nhau nên ngưỡng cảnh báo cũng phải khác nhau.
- **Alert kêu thì:** (1) kiểm tra xem alert có phải do lỗi pipeline dữ liệu hoặc logging không, (2) tìm xem đặc trưng nào bị drift và thuộc loại data drift hay concept drift, (3) đánh giá mức độ ảnh hưởng đến hiệu suất mô hình. Sau đó mới quyết định: retrain theo lịch nếu drift xảy ra chậm và theo mùa vụ, retrain theo trigger khi PSI hoặc metric tăng đột ngột vượt ngưỡng, hoặc rollback ngay về version cũ trong Model Registry (như M2) nếu mô hình mới vừa deploy gây vấn đề, trong lúc tiếp tục điều tra.

---

## Module 5 — Testing & CI/CD

- Số test pass: **5 / 5** · Coverage: **32%** trên `src/train_pipeline.py` (module được test); tổng `--cov=src` là 8% vì src/ chứa cả các script demo (skew_demo, drift, make_curl) không thuộc phạm vi unit test.
- Lệnh: `pytest tests/ -v --cov=src --cov-report=term`
- CI: `ci.yml` đã có sẵn; bước `python src/train.py --smoke` nay chạy được (đã tạo `src/train.py` hỗ trợ cờ `--smoke`: 100 mẫu, 1 cấu hình, không log MLflow).

**Săn lỗi (test vô dụng):**

```python
def test_useless():                       # phản ví dụ
    X, y = load_breast_cancer(return_X_y=True)
    pipe = build_pipeline().fit(X, y)     # fit trên TOÀN BỘ dữ liệu
    acc = (pipe.predict(X) == y).mean()   # chấm trên chính dữ liệu đã fit
    assert acc >= 0.5                     # ngưỡng quá thấp
```

- 2 vấn đề: (1) chấm trên dữ liệu đã fit → mô hình học vẹt/overfit vẫn đạt điểm cao, test không phân biệt được model tốt/xấu; (2) ngưỡng 0.5 = mức đoán ngẫu nhiên trên bài nhị phân → gần như không thể fail.
- Cách tôi viết lại: `test_holdout_performance` trong `tests/test_model.py` — split train/test ngay trong test, đánh giá trên tập CHƯA fit, ngưỡng 0.90 sát năng lực thật.

**Phản biện M5:** 
- **CI bắt được:** Lỗi thường gặp khi deploy mô hình gồm: lỗi code (syntax, import sai), pipeline tiền xử lý không chạy được, output trả về sai shape hoặc sai kiểu dữ liệu, và performance của mô hình giảm mạnh dưới ngưỡng trên tập dữ liệu kiểm tra cố định. **CI KHÔNG bắt được:** Mô hình có thể hỏng vì dữ liệu production thay đổi (drift), trong khi CI/CD vẫn chạy test trên dữ liệu cũ (đóng băng) nên tất cả test vẫn xanh, nhưng thực tế mô hình đã kém đi. Loại lỗi này chỉ phát hiện được nhờ M4 - Monitoring (theo dõi PSI và log dự đoán). Đây chính là vòng đời MLOps hoàn chỉnh (Bài 15): CI/CD lo phần build và deploy, Monitoring lo phần vận hành. Khi monitoring phát hiện vấn đề → kích hoạt retrain → quay lại CI/CD để deploy phiên bản mới, tạo thành vòng lặp liên tục.
- **CI/CD vs CT:** 
    + CI/CD được kích hoạt khi có thay đổi code (push code hoặc pull request).
    + Continuous Training (CT) được kích hoạt khi có thay đổi về dữ liệu hoặc hiệu suất (dữ liệu mới theo lịch, hoặc alert drift vượt ngưỡng).
    + Cốt lõi là: trong Machine Learning, mô hình có thể “hỏng” dù không ai sửa code.
- **Tách train/test trong test:** để test đo khả năng khái quát hóa chứ không đo khả năng ghi nhớ; chấm trên dữ liệu đã fit thì test mất chức năng phát hiện regression.

---

## Module 6 — Tổng hợp & Maturity

**Bảng ghép vòng đời MLOps ↔ Module:**

| Giai đoạn vòng đời | Thành phần tôi đã làm | Module |
|---|---|---|
| Versioning | Git (main/develop/feature), .gitignore loại model/secrets; Registry version model | M0, M2 |
| Training pipeline | sklearn Pipeline (scaler+clf) chống leakage, seed cố định | M1 |
| Experiment tracking / Registry | MLflow 4 run, Compare; `bc_classifier` v1 → Production | M2 |
| Deployment | joblib serialize pipeline v1.0.0 + FastAPI /predict, /health, /model-info | M3 |
| Monitoring | PSI (tự viết, ngưỡng 0.2) + KS test; prediction logging 5 trường | M4 |
| Testing / CI | 5 unit test pytest + coverage; GitHub Actions test + smoke-train | M5 |

**Tự xếp maturity level:** Level **1 (một phần)** — bằng chứng: đã có pipeline train tự động hóa được (chạy 1 lệnh), experiment tracking + Registry, CI chạy test tự động mỗi push → vượt Level 0 (notebook rời rạc, thủ công). Chưa đạt Level 2 vì: retrain vẫn do người chạy tay, alert drift (M4) chưa nối vào pipeline retrain tự động, chưa có data versioning.

**Bước lên level kế tiếp + lý do ưu tiên:** nối alert PSI ≥ 0.2 (M4) vào một job retrain tự động (chạy lại train_mlflow.py, so với Production hiện tại, tự promote nếu tốt hơn). Ưu tiên vì đây là mắt xích khép kín vòng lặp monitor → retrain → deploy, biến hệ thống từ "phản ứng thủ công" sang "tự phục hồi".

**Tự đánh giá (5–7 câu):** 
- 

**Phản biện M6:** 
- **DevOps vs MLOps qua chính bài làm:** 
    + Testing: DevOps chỉ test code (ví dụ kiểm tra shape, label ở M5). MLOps còn test thêm chất lượng mô hình và dữ liệu (như accuracy trên holdout set phải ≥ 0.90, không có NaN…). Những test này có thể fail dù code không thay đổi, chỉ vì dữ liệu thay đổi.
    + Maintenance: DevOps bảo trì khi có lỗi code. MLOps thì mô hình có thể tự xuống cấp theo thời gian do drift (chỉ cần phân phối dữ liệu dịch chuyển một chút là PSI tăng vọt từ 0.04 lên 2.34). Vì vậy bảo trì trong MLOps là giám sát liên tục + retrain mô hình, chứ không chờ có bug report mới xử lý.
- **Mắt xích yếu nhất về reproducibility: DATA.** Code có git, model có MLflow Registry, params/seed có log — nhưng dataset load trực tiếp `load_breast_cancer()` không có version/hash; ở dự án thật, data trên DB/S3 đổi liên tục mà không snapshot thì mọi thứ khác đầy đủ vẫn không tái lập đúng model. Khắc phục: DVC hoặc lưu hash + snapshot dữ liệu train kèm mỗi run.
