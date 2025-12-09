import mlflow


def find_best_run(
    experiment_name: str = "mlops_demo_experiment",
    metric: str = "metrics.val_accuracy",
    ascending: bool = False,
):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"{metric} {'ASC' if ascending else 'DESC'}"],
    )

    if runs.empty:
        print("No runs found.")
        return None

    best_run = runs.iloc[0]
    print("Best run:")
    print(f"  run_id: {best_run.run_id}")
    print(f"  {metric}: {best_run[metric]}")
    print(f"  run_name: {best_run['tags.mlflow.runName']}")
    return best_run


if __name__ == "__main__":
    find_best_run()
