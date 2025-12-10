# Typical ML Project Structure with MLflow

This article explains the basic script structure for an ML project using MLflow. It shows how training, evaluation, tuning, and model selection are organized into separate, easy-to-manage components.


For simple notebook, check the [example](https://colab.research.google.com/drive/1sNPQNRpDo3Pg6JAw6hmBNRAjdMFvalbh#scrollTo=JZfA5OQFI_Bc)

## Table of Contents

- [Typical ML Project Structure with MLflow](#typical-ml-project-structure-with-mlflow)
  - [1. Training script (train.py)](#1-training-script-trainpy)
  - [2. Evaluation script (evaluatepy)](#2-evaluation-script-evaluatepy)
  - [3. Hyperparameter tuning / advanced evaluation (tunepy)](#3-hyperparameter-tuning--advanced-evaluation-tunepy)
  - [4. Model comparison and selection](#4-model-comparison-and-selection)


## 1. Training script (`train.py`)

- Load and preprocess data.
- Define and train the model.
  - Wrap the whole training loop in:
  
    ```python
    with mlflow.start_run():
        # log_params, log_metrics, log_artifacts, log_model ...
    ```
- Log to MLflow:
  - Parameters (hyperparameters, data config, etc.).
  - Metrics (training / validation loss, basic scores).
  - Artifacts (model files, plots, config).

The model itself (and optionally register it in the Model Registry and set a stage like Staging or Production).

Optionally save predictions (on validation / test data) for later use by an evaluation script.


## 2. Evaluation script (evaluate.py)

```python
    with mlflow.start_run(run_name="evaluate_model") as run:
        mlflow.set_tag("mlflow.runName", "evaluate_model")
        df = pd.read_csv("data/predictions/test_predictions.csv")
        metrics = classification_metrics(df)
        mlflow.log_metrics(metrics)
```

1. Load Either the saved predictions or
2. Compute and Log evaluation metrics (e.g., AUC, F1, RMSE), artifacts (plots, reports) to MLflow as another run.

For Simplicity, you can embed evaluation within trainer.py.
- [example](https://github.com/GeneSUN/NetSignalOutlierPipeline/tree/main/src/modeling/global_autoencoder#high-level-architecture)

## 3. Hyperparameter tuning / advanced evaluation (tune.py)


```python
    with mlflow.start_run(run_name="Main_HPO_Run") as main_run:
        with mlflow.start_run(nested=True):
```

1. Define search space
2. Create objective function, embedding nested run:
  - Trains a model (possibly a lightweight version),
  - Evaluates a set of hyperparameters on a validation set,
  - Logs everything to MLflow inside its own mlflow.start_run(),
3. Use a search method (grid search, random search, Optuna, Hyperopt, etc.) to call the objective function on many hyperparameter combinations.
  - ```python from hyperopt import fmin, tpe, hp, STATUS_OK, Trials```, ```from sklearn.model_selection import cross_val_score```


- full script template: https://github.com/GeneSUN/mlflow_toolbox/blob/main/mlops_demo_project/evaluation/tune.py


4. Use the MLflow UI or a small analysis script to:
  - Compare runs by metrics, select the best model configuration.
  - Inspect the relationship between hyperparameters and performance,

<img width="1033" height="669" alt="Screenshot 2025-12-10 at 12 04 01 PM" src="https://github.com/user-attachments/assets/a8b6edf9-de08-431a-877e-6205e36bfb33" />


## 4. Model comparison and selection

**Use MLflow UI to:**
  - Filter runs by tags, parameters, or metrics.
  - Sort by the key metric (e.g., validation AUC).
  - Promote the best model to Staging or Production in the Model Registry.

<img width="1407" height="624" alt="Screenshot 2025-12-10 at 12 07 40 PM" src="https://github.com/user-attachments/assets/a085224d-4a00-4d19-a554-aa36f3840a2f" />


**Optionally, write a small Python script to:**
  - Programmatically query runs via mlflow.search_runs, Select the best run,
  - Automatically register or transition the corresponding model.

https://github.com/GeneSUN/mlflow_toolbox/blob/main/mlops_demo_project/evaluation/compare_runs.py

```python

    ....

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"{metric} {'ASC' if ascending else 'DESC'}"],
    )

```
