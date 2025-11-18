# MLflow Structure Overview

```
MLflow Tracking
   ├── Experiments
   │       └── Run 1
   │       │     └── Artifacts / Metrics / Params
   │       ├── Run 2
   │       │     └── Artifacts / Metrics / Params
MLflow Models
   └── Model Registry
           ├── Registered Models
           │       └── Model Versions
           │          ├── Version 1 (from a specific run)
           │          ├── Version 2 (from another run)
           │          └── Version 3 ...
           └── Stages (None, Staging, Production, Archived)
```

<img width="1255" height="828" alt="Untitled" src="https://github.com/user-attachments/assets/e2ca4fd0-fbb8-45b7-a333-0772ca313178" />


## 1️⃣ MLflow Tracking
### 1.1. MLflow Experiment
```python
import mlflow
mlflow.create_experiment("wifi_score_experiment")
mlflow.set_experiment("wifi_score_experiment")
```



### 1.2. MLflow Run

A single execution of training code, One run contains:
- metrics (mlflow.log_metric)
- parameters (mlflow.log_param)
- artifacts (images, models, files)

**metrics/parameters**
```
with mlflow.start_run(run_name="baseline_RF"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", 0.12)
    mlflow.log_artifact("feature_importance.png")
```
**Artifacts**
in additition to metrics, you can save Artifacts:
- model files
- plots (matplotlib)
- processed datasets
- logs


```python
    # Save CSV
    save_path = f"{RAW_PATH}/data.csv"
    df.to_csv(save_path)

    # Log as MLflow artifact
    mlflow.log_artifact(save_path)
```



## 2. MLflow Models
```
artifacts/
    model/
        MLmodel
        conda.yml
        model.pkl
```


```python
mlflow.sklearn.log_model(model, artifact_path="model")

result = mlflow.sklearn.log_model(model, "model")
mlflow.register_model(result.model_uri, "wifi_score_predictor")

```
