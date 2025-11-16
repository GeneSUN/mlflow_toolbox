
```
MLflow Tracking
   ├── Experiments
   │       └── Runs
   │             └── Artifacts / Metrics / Params
   │
MLflow Models
   └── Model Registry
           ├── Registered Models
           │       └── Model Versions
           └── Stages (None, Staging, Production, Archived)
```


```
Experiment A
   ├── Run 1
   │     ├── metrics
   │     ├── parameters
   │     ├── artifacts (plots, models, files)
   │
   ├── Run 2
   │     ├── metrics
   │     ├── params
   │     ├── artifacts
   │
   └── Run 3 ...

Model Registry
   ├── "fraud_detector" (registered model)
   │          ├── Version 1 (from a specific run)
   │          ├── Version 2 (from another run)
   │          └── Version 3 ...
   └── "wifi_score_predictor"
```

# 1️⃣ MLflow Tracking
## 1.1. MLflow Experiment

A folder / logical container that groups related ML runs.

## How to create an experiment

```
import mlflow
mlflow.create_experiment("wifi_score_experiment")
mlflow.set_experiment("wifi_score_experiment")
```

# 1.2. MLflow Run

A single execution of training code, One run contains:
- metrics (mlflow.log_metric)
- parameters (mlflow.log_param)
- artifacts (images, models, files)

```
with mlflow.start_run(run_name="baseline_RF"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", 0.12)
    mlflow.log_artifact("feature_importance.png")
```

# 1.3. MLflow Artifacts
Inside each run, you can save:

- model files
- plots (matplotlib)
- processed datasets
- logs


# 2. MLflow Models
```
artifacts/
    model/
        MLmodel
        conda.yml
        model.pkl

mlflow.sklearn.log_model(model, artifact_path="model")

result = mlflow.sklearn.log_model(model, "model")
mlflow.register_model(result.model_uri, "wifi_score_predictor")

```


# End-to-End Example

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. Create or set experiment
mlflow.set_experiment("wifi_score_prediction")

with mlflow.start_run(run_name="rf_baseline") as run:

    # 2. Log params & metrics
    mlflow.log_param("n_estimators", 200)

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    mlflow.log_metric("r2", score)

    # 3. Log model
    mlflow.sklearn.log_model(model, "model")

# 4. Register model in registry
mlflow.register_model(
    f"runs:/{run.info.run_id}/model",
    "wifi_score_predictor"
)
```






