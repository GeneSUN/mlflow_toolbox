import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


def classification_metrics(df: pd.DataFrame) -> dict:
    y_true = df["y_test"]
    y_pred = df["y_pred"]
    y_proba = df.get("y_proba", None)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["log_loss"] = log_loss(y_true, y_proba)

    return metrics


if __name__ == "__main__":
    mlflow.set_experiment("mlops_demo_experiment")
    with mlflow.start_run(run_name="evaluate_model") as run:
        mlflow.set_tag("mlflow.runName", "evaluate_model")
        df = pd.read_csv("data/predictions/test_predictions.csv")
        metrics = classification_metrics(df)
        mlflow.log_metrics(metrics)
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
