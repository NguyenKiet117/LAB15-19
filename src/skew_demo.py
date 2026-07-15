"""Module 3 - Buoc 4 (SAN LOI): luu CHI estimator (bo scaler),
roi phuc vu du lieu tho -> training-serving skew."""
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(max_iter=5000, random_state=SEED)
clf.fit(scaler.transform(X_tr), y_tr)  # train tren du lieu DA scale

joblib.dump(clf, "models/estimator_only.joblib")  # <-- quen scaler
loaded = joblib.load("models/estimator_only.joblib")

acc_raw = accuracy_score(y_te, loaded.predict(X_te))  # serve du lieu THO
acc_ok = accuracy_score(y_te, loaded.predict(scaler.transform(X_te)))

print(f"Accuracy khi serve du lieu tho : {acc_raw:.4f}   <-- SKEW")
print(f"Accuracy khi serve dung (scale): {acc_ok:.4f}")
print("Phan bo du doan (tho) [class0, class1]:",
      np.bincount(loaded.predict(X_te), minlength=2))
print("Phan bo nhan that       [class0, class1]:",
      np.bincount(y_te, minlength=2))
