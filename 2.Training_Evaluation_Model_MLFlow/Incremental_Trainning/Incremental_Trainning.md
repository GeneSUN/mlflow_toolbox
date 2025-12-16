# incremental learning (a.k.a. online learning / continual learning).

## 1. Comprehensive Method: retrain everytime adding new data
step 1. Initial Training (Full Historical Data)

```python
    model.fit(df_hist)
```

step 2. retraining from scratch (merge old + new) 

```python
# Merge existing training data + new data
df_hist = pd.read_csv("data/churn_2_years.csv")
df_all = pd.concat([df_hist, df_new], axis=0)

new_model.fit(df_all)
```

Strength of this method:
Most ML models cannot be fine-tuned using new data unless you retrain from scratch.
- Incremental training requires very specific model properties, Many classical ML models are not designed to continue training.
- You can emphasize recent behavior by assigning higher weight to newer data or downsampling older historical records.
- When data volume is very large and training is efficient, retraining the model from scratch may be simpler and cost-effective.

| Model                                                     | Incremental?         | Why not                                                                 |
| --------------------------------------------------------- | -------------------- | ----------------------------------------------------------------------- |
| **Random Forest**                                         | ❌ No                 | Trees are fixed after training; new data requires rebuilding the forest |
| **Decision Trees**                                        | ❌ No                 | The splits are not updateable once created                              |
| **XGBoost / LightGBM / CatBoost (default)**               | ❌ No                 | Models are additive but not updateable after training ends              |
| **SVM (RBF kernel)**                                      | ❌ No                 | Support vectors must be recomputed globally                             |
| **KNN**                                                   | ⚠️ Yes but expensive | Adding points is trivial but inference becomes slow                     |
| **Most deep learning models trained without checkpoints** | ❌ No                 | You cannot “continue training” without original optimizer state         |



## ✅ 2. Models That CAN Do Incremental Training (Online Learning)


### ✔ A. Models with partial_fit() (True incremental learners)

```python
model.partial_fit(X_new, y_new)
```

| Model                       | Library | Supports incremental?          |
| --------------------------- | ------- | ------------------------------ |
| SGDClassifier               | sklearn | ✔ incremental gradient descent |
| PassiveAggressiveClassifier | sklearn | ✔ online updates               |
| Perceptron                  | sklearn | ✔                              |
| BernoulliNB / MultinomialNB | sklearn | ✔                              |
| MiniBatchKMeans             | sklearn | ✔                              |
| GaussianNB                  | sklearn | ✔                              |

### ✔ B. Deep Learning Models (PyTorch / TensorFlow)

Deep models can be incrementally trained if:
- you saved the model weights
- AND the optimizer state
- AND learning rate schedule if needed

- https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html?utm_source=chatgpt.com
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html
- https://github.com/GeneSUN/NetSignalOutlierPipeline/blob/main/src/modeling/global_autoencoder/readme.md#incremental-trainning

### ✔ C. XGBoost / LightGBM (but only in special modes)

**Option 1 — ```xgb_model=model```**

Continuing boosting adds new trees, does NOT update old trees:

```python xgb.train(params, dtrain_new, xgb_model=old_model)```

**Option 2 — Using process_type='update'**

This updates leaf weights, not structure.


### ✔ D. Online / Streaming algorithms (River library)
River is designed for incremental learning from data streams.

Examples:
- Hoeffding Trees (incremental Decision Trees)
- Naive Bayes
- Logistic Regression
- Online clustering
- Online anomaly detection





