import pandas as pd
import numpy as np
import mlflow


def compute_mean_std(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    stats = []
    for col in feature_cols:
        stats.append(
            {
                "feature": col,
                "mean": df[col].mean(),
                "std": df[col].std(),
            }
        )
    return pd.DataFrame(stats)


if __name__ == "__main__":
    mlflow.set_experiment("mlops_demo_experiment")
    with mlflow.start_run(run_name="drift_monitor") as run:
        mlflow.set_tag("mlflow.runName", "drift_monitor")

        train_df = pd.read_csv("data/training/train.csv")
        test_df = pd.read_csv("data/training/test.csv")

        feature_cols = [c for c in train_df.columns if c != "target"]

        train_stats = compute_mean_std(train_df, feature_cols)
        test_stats = compute_mean_std(test_df, feature_cols)

        merged = train_stats.merge(
            test_stats, on="feature", suffixes=("_train", "_test")
        )
        merged["mean_diff"] = (merged["mean_test"] - merged["mean_train"]).abs()
        merged["std_ratio"] = merged["std_test"] / (merged["std_train"] + 1e-8)

        # Log summary drift metrics
        mean_mean_diff = merged["mean_diff"].mean()
        mean_std_ratio = merged["std_ratio"].mean()

        mlflow.log_metric("drift_mean_mean_diff", float(mean_mean_diff))
        mlflow.log_metric("drift_mean_std_ratio", float(mean_std_ratio))

        print("Drift summary:")
        print(merged)
        print(f"Average |mean difference|: {mean_mean_diff:.4f}")
        print(f"Average std ratio: {mean_std_ratio:.4f}")
