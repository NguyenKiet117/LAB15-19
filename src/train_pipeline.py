"""Module 1: Pipeline chuẩn + thí nghiệm data leakage."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42

def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
    ])

def run_correct():
    X, y, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    cv_auc = cross_val_score(build_pipeline(), X_tr, y_tr,
                             cv=5, scoring="roc_auc")

    print("=== BẢN ĐÚNG (scale bên trong pipeline) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"CV 5-fold ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    return auc

def run_leakage():
    """SĂN LỖI: cố tình scale TOÀN BỘ X trước khi split → leakage."""
    X, y, _ = load_data()
    X_scaled = StandardScaler().fit_transform(X)   # <-- LỖI CỐ Ý
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=SEED
    )
    clf = LogisticRegression(max_iter=5000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    print("\n=== BẢN LEAKAGE (scale trước khi split) ===")
    print(f"ROC-AUC  : {auc:.4f}")
    return auc

if __name__ == "__main__":
    auc_ok = run_correct()
    auc_leak = run_leakage()
    print(f"\nChênh lệch ROC-AUC (leak - đúng): {auc_leak - auc_ok:+.4f}")