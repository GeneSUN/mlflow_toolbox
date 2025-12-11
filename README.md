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

```
mlops_demo_project/
│
├── config/
│   ├── train.yaml
│   ├── eval.yaml
│   ├── inference.yaml
│   └── monitoring.yaml
│
├── modeling/
│   ├── trainer.py
│   ├── data_loader.py
│   └── preprocess.py
│
├── evaluation/
│   ├── evaluate_model.py
│   ├── compare_runs.py
│   ├── validate_schema.py
│   └── tune.py
│
├── inference/
│   ├── inference_wrapper.py
│   └── api_server.py  # FastAPI server (optional)
│
├── monitoring/
│   ├── drift_monitor.py
│   ├── prediction_monitor.py
│   └── data_quality.py
│
├── pipelines/
│   ├── train_pipeline.py
│
├── tests/
│   ├── ...
│
│
├── requirements.txt
├── Dockerfile
└── README.md

```
<img width="1141" height="719" alt="Untitled" src="https://github.com/user-attachments/assets/b26ec057-4c40-4302-8f48-2c1bedf6a547" />



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
```python
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
