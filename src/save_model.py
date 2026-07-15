"""Module 3 - Buoc 1: Serialize toan bo pipeline, load lai va assert khop."""
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from train_pipeline import load_data, build_pipeline

SEED = 42

X, y, _ = load_data()
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
pipe = build_pipeline().fit(X_tr, y_tr)

PATH = "models/bc_pipeline_v1.0.0.joblib"
joblib.dump(pipe, PATH)

loaded = joblib.load(PATH)
assert np.array_equal(pipe.predict(X_te), loaded.predict(X_te)), "Mismatch!"
assert np.allclose(pipe.predict_proba(X_te), loaded.predict_proba(X_te))
print(f"Saved & verified: {PATH}")
print(f"Sanity check accuracy: {(loaded.predict(X_te) == y_te).mean():.4f}")
