import numpy as np
import pandas as pd
import mlflow


class InferenceWrapper:
    def __init__(
        self,
        model_uri: str = "models:/mlops_demo_classifier/Production",
    ):
        # This loads the model from the MLflow Model Registry
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df)
        # For sklearn classifiers logged with mlflow.sklearn.log_model,
        # predict() typically returns class labels. We can use predict_proba
        # if we want probabilities. To keep the example simple, we assume
        # the wrapped model exposes predict_proba via underlying model.
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(df)[:, 1]
        # Fallback: treat predictions as 0/1 and cast to float
        return preds.astype(float)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        proba = self.predict_proba(df)
        y_pred = (proba >= threshold).astype(int)
        result = df.copy()
        result["y_proba"] = proba
        result["y_pred"] = y_pred
        return result


if __name__ == "__main__":
    # Simple demo: load some test data and run inference
    from pathlib import Path

    test_path = Path("data/training/test.csv")
    if not test_path.exists():
        raise FileNotFoundError(
            "data/training/test.csv not found. Run modeling/train_model.py first."
        )

    test_df = pd.read_csv(test_path)
    features = [c for c in test_df.columns if c != "target"]
    X_new = test_df[features].head(5)

    wrapper = InferenceWrapper()
    preds = wrapper.predict(X_new)
    print("Inference results:")
    print(preds)
