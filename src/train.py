"""Entry train dung chung cho CI (M5) va chay tay.

--smoke: train nhanh tren ~100 mau, 1 cau hinh, KHONG log MLflow —
chi de CI kiem tra pipeline train khong gay.
"""
import argparse

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from train_pipeline import build_pipeline

SEED = 42


def main(smoke: bool):
    X, y = load_breast_cancer(return_X_y=True)
    if smoke:
        X, y = X[:100], y[:100]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pipe = build_pipeline().fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    mode = "SMOKE" if smoke else "FULL"
    print(f"[{mode}] train OK — accuracy = {acc:.4f} tren {len(y_te)} mau test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="train nhanh tren mau nho cho CI")
    args = parser.parse_args()
    main(args.smoke)
