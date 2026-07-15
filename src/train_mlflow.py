"""Module 2: Log ≥3 cấu hình vào MLflow, đăng ký model tốt nhất."""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42
mlflow.set_experiment("bc_mlops")

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

configs = [
    ("logreg_C1",   LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)),
    ("logreg_C001", LogisticRegression(C=0.01, max_iter=5000, random_state=SEED)),
    ("rf_200",      RandomForestClassifier(n_estimators=200, random_state=SEED)),
    ("svc_rbf",     SVC(kernel="rbf", probability=True, random_state=SEED)),
]

best = {"auc": -1, "run_id": None, "name": None}

for name, est in configs:
    with mlflow.start_run(run_name=name) as run:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = pipe.predict(X_te)

        metrics = {
            "accuracy": accuracy_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
        }
        mlflow.log_params(est.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print(f"{name}: {metrics}")

        if metrics["roc_auc"] > best["auc"]:
            best = {"auc": metrics["roc_auc"],
                    "run_id": run.info.run_id, "name": name}

# Đăng ký model tốt nhất vào Registry
result = mlflow.register_model(
    model_uri=f"runs:/{best['run_id']}/model",
    name="bc_classifier",
)
print(f"\nBest: {best['name']} (AUC={best['auc']:.4f})")
print(f"Registered: bc_classifier version {result.version}")