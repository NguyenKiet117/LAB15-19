"""Module 5: unit tests cho pipeline (pytest)."""
import os
import sys

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pipeline import build_pipeline  # noqa: E402

SEED = 42


@pytest.fixture(scope="module")
def data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


@pytest.fixture(scope="module")
def fitted(data):
    X_tr, _, y_tr, _ = data
    return build_pipeline().fit(X_tr, y_tr)


def test_output_shape(fitted, data):
    _, X_te, _, _ = data
    assert fitted.predict(X_te).shape == (X_te.shape[0],)


def test_binary_labels(fitted, data):
    _, X_te, _, _ = data
    assert set(np.unique(fitted.predict(X_te))) <= {0, 1}


def test_proba_in_range(fitted, data):
    _, X_te, _, _ = data
    proba = fitted.predict_proba(X_te)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_nan_input_raises(fitted, data):
    """Edge case: dau vao chua NaN phai bi chan, khong tra ket qua rac."""
    _, X_te, _, _ = data
    x_bad = X_te[:1].copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        fitted.predict(x_bad)


def test_holdout_performance(data):
    """Test dung cach: danh gia tren tap CHUA fit, nguong thuc te 0.90.

    (Ban viet lai cua test 'vo dung' — xem ANSWERS.md M5.)
    """
    X_tr, X_te, y_tr, y_te = data
    pipe = build_pipeline().fit(X_tr, y_tr)
    acc = (pipe.predict(X_te) == y_te).mean()
    assert acc >= 0.90, f"Accuracy holdout {acc:.3f} < 0.90"
