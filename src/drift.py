"""Module 4: PSI tu viet + kiem chung tinh tay + mo phong 2 kich ban drift."""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer

PSI_THRESHOLD = 0.2


def psi(expected, actual, n_bins=10):
    """PSI voi bin chia theo quantile cua phan phoi expected."""
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    a_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6  # tranh chia 0 / log 0
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


# ---- Buoc 2: kiem chung ham code khop voi bang tinh tay (4 bin) ----
e = np.array([0.25, 0.25, 0.25, 0.25])
a = np.array([0.10, 0.20, 0.30, 0.40])
psi_hand = float(np.sum((a - e) * np.log(a / e)))
print(f"PSI kiem chung theo bang tinh tay (4 bin): {psi_hand:.4f}")
print("(doi chieu voi so ban tinh tay tren giay — phai khop)\n")

# ---- Buoc 3: mo phong 2 kich ban production ----
rng = np.random.default_rng(42)
X, _ = load_breast_cancer(return_X_y=True)
feat = X[:, 0]  # 'mean radius'
train_ref = feat

prod_no_drift = rng.choice(feat, size=200, replace=True)            # KB1
prod_drift = rng.choice(feat, size=200, replace=True) + feat.std()  # KB2: +1 sigma

for name, prod in [("KHONG drift", prod_no_drift), ("CO drift", prod_drift)]:
    p = psi(train_ref, prod)
    ks_stat, ks_p = ks_2samp(train_ref, prod)
    alert = "ALERT (PSI >= 0.2)" if p >= PSI_THRESHOLD else "on dinh"
    print(f"[{name}] PSI = {p:.4f} -> {alert}")
    print(f"[{name}] KS stat = {ks_stat:.4f}, p-value = {ks_p:.3e}\n")
