"""
tune.py — Hyperopt + MLflow Hyperparameter Optimization Template

Modify the following sections:
1. load_data()        -> load your dataset
2. search_space       -> define hyperparameters
3. objective()        -> how to train model + compute metric
"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import mlflow
import mlflow.pyfunc
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from typing import Any, Dict, Tuple


# =========================================================
# 1. Load Data (REPLACE THIS WITH YOUR OWN LOGIC)
# =========================================================
def load_data() -> Tuple[Any, Any, Any, Any]:
    """
    Load your dataset and return (X_train, X_test, y_train, y_test).
    Replace the code below with your own data loading logic.
    """
    raise NotImplementedError("TODO: Implement load_data(). Example: read CSV, parquet, or Spark dataframe.")


# =========================================================
# 2. Define Search Space (REPLACE WITH YOUR PARAMS)
# =========================================================
def get_search_space() -> Dict[str, Any]:
    """
    Define your Hyperopt parameter search space.

    Returns:
        A dictionary specifying hyperparameter distributions.
    """
    return {
        # EXAMPLES — Replace these with your model parameters
        "lr": hp.loguniform("lr", -5, 0),               # learning rate
        "hidden_dim": hp.choice("hidden_dim", [32, 64, 128]),
        "dropout": hp.uniform("dropout", 0.0, 0.5),

        # Add more params as needed
    }


# =========================================================
# 3. Objective Function (THE MAIN PART YOU CUSTOMIZE)
# =========================================================
def objective(params: Dict[str, Any]):
    """
    Objective function for hyperparameter tuning.
    - Train a model with given params
    - Evaluate with CV or a validation set
    - Log everything to MLflow

    Returns:
        dict {"loss": ..., "status": STATUS_OK}
    """

    # Enable autolog if applicable (optional)
    # mlflow.sklearn.autolog()
    # mlflow.xgboost.autolog()
    # mlflow.lightgbm.autolog()
    # mlflow.pytorch.autolog()
    # (Pick the one you need)

    with mlflow.start_run(nested=True):

        mlflow.log_params(params)

        # -----------------------------------------------------
        # TODO: TRAIN YOUR MODEL HERE
        #
        # Example (sklearn):
        #
        # model = MyModel(**params)
        # model.fit(X_train, y_train)
        # preds = model.predict(X_test)
        #
        # Replace everything below with your real logic.
        # -----------------------------------------------------

        raise NotImplementedError("TODO: Train your model and compute metric.")

        # EXAMPLE for CV metric:
        # scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1")
        # f1_mean = scores.mean()
        # loss = 1 - f1_mean

        # mlflow.log_metric("f1_cv", f1_mean)
        # mlflow.log_metric("loss", loss)

        # return {"loss": loss, "status": STATUS_OK}


# =========================================================
# 4. Run Optimization
# =========================================================
def run_tuning(max_evals: int = 20):
    """
    Runs the Hyperopt optimization with MLflow tracking.
    """
    # Load your dataset
    X_train, X_test, y_train, y_test = load_data()

    search_space = get_search_space()
    trials = Trials()

    mlflow.set_experiment("Hyperopt_Tuning_Experiment")

    with mlflow.start_run(run_name="Main_HPO_Run") as main_run:

        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
        )

        mlflow.log_param("best_hyperparams", best_params)

        print("\n==============================")
        print(" Best Hyperparameters Found ")
        print("==============================")
        print(best_params)


# =========================================================
# 5. Run as Script
# =========================================================
if __name__ == "__main__":
    run_tuning(max_evals=30)
