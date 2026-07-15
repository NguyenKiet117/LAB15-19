"""Module 3 - Buoc 3: In san 3 lenh curl tu 3 mau that (kem nhan that)."""
import json
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
for i in [0, 100, 550]:
    print(f"# Mau {i}, nhan that = {y[i]} ({['malignant','benign'][y[i]]})")
    body = json.dumps({"features": X[i].tolist()})
    print(
        "curl -s -X POST http://127.0.0.1:8000/predict "
        f"-H 'Content-Type: application/json' -d '{body}'\n"
    )
