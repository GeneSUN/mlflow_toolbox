## Typical ML Project Structure with MLflow

### 1. Training script (`train.py`)

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


### 2. Evaluation script (evaluate.py)


1. Load Either the saved predictions or
2. Compute and Log evaluation metrics (e.g., AUC, F1, RMSE), artifacts (plots, reports) to MLflow as another run.



### 3. Hyperparameter tuning / advanced evaluation (tune.py)

- Define an objective function that:
  - Takes a set of hyperparameters as input,
  - Trains a model (possibly a lightweight version),
  - Evaluates on a validation set,
  - Logs everything to MLflow inside its own mlflow.start_run(),
  - Returns a scalar score to maximize or minimize.

- Use a search method (grid search, random search, Optuna, Hyperopt, etc.) to call the objective function on many hyperparameter combinations.
  - Optionally, create a parent run and log each trial as a nested run under it.

After the search:

- Use the MLflow UI or a small analysis script to:
  - Compare runs by metrics,
  - Inspect the relationship between hyperparameters and performance,
  - Select the best model configuration.

<img width="1033" height="669" alt="Screenshot 2025-12-10 at 12 04 01 PM" src="https://github.com/user-attachments/assets/a8b6edf9-de08-431a-877e-6205e36bfb33" />


### 4. Model comparison and selection

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
