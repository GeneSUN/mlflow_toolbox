import os
from dataclasses import dataclass
from typing import Tuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@dataclass
class TrainerConfig:
    experiment_name: str = "mlops_demo_experiment"
    registered_model_name: str = "mlops_demo_classifier"
    test_size: float = 0.2
    random_state: int = 42
    n_samples: int = 1000
    n_features: int = 10
    n_informative: int = 6
    n_redundant: int = 2
    n_estimators: int = 100
    max_depth: int = 5


class Trainer:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()
        self._ensure_dirs()

    @staticmethod
    def _ensure_dirs():
        os.makedirs("data/training", exist_ok=True)
        os.makedirs("data/predictions", exist_ok=True)

    def generate_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            random_state=self.config.random_state,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        cols = [f"feature_{i}" for i in range(X.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=cols)
        X_test_df = pd.DataFrame(X_test, columns=cols)
        y_train_s = pd.Series(y_train, name="target")
        y_test_s = pd.Series(y_test, name="target")

        # Save training and test data for later use (drift / debugging)
        train_df = X_train_df.copy()
        train_df["target"] = y_train_s
        test_df = X_test_df.copy()
        test_df["target"] = y_test_s

        train_df.to_csv("data/training/train.csv", index=False)
        test_df.to_csv("data/training/test.csv", index=False)

        return X_train_df, y_train_s, X_test_df, y_test_s

    def train(self) -> None:
        mlflow.set_experiment(self.config.experiment_name)

        X_train, y_train, X_test, y_test = self.generate_data()

        with mlflow.start_run(run_name="train_model") as run:
            mlflow.set_tag("mlflow.runName", "train_model")

            # Enable autologging for sklearn models
            mlflow.sklearn.autolog(log_models=False)

            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            )
            model.fit(X_train, y_train)

            # Simple evaluation inside training run (train-time validation)
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("val_accuracy", acc)

            # Save predictions for offline evaluation step
            preds_df = X_test.copy()
            preds_df["y_pred"] = y_pred
            preds_df["y_proba"] = y_proba
            preds_df["y_test"] = y_test.values
            preds_path = "data/predictions/test_predictions.csv"
            preds_df.to_csv(preds_path, index=False)
            mlflow.log_artifact(preds_path, artifact_path="predictions")

            """ Register and promote the model to Production stage 
            # Log the trained model and register it
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=self.config.registered_model_name,
            )

            # Promote the latest version to Production for inference
            client = MlflowClient()
            latest_versions = client.get_latest_versions(
                name=self.config.registered_model_name, stages=["None"]
            )
            if latest_versions:
                version = latest_versions[0].version
                client.transition_model_version_stage(
                    name=self.config.registered_model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(
                    f"Registered model '{self.config.registered_model_name}' "
                    f"version {version} promoted to Production."
                )
            else:
                print("No model versions found to promote.")
            """
            print(f"Training run completed. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
