# MLflow Deployment & Inference Guide

This guide explains 
- how to run inference with MLflow using two common wrapper patterns  
  - **external inference wrapper** vs.
  - **MLflow pyfunc wrapper**)
- how to deploy the model for  **batch jobs** or **API services**.  

<img width="1328" height="1037" alt="Untitled" src="https://github.com/user-attachments/assets/cce8b007-1016-49b4-ba69-ad8878f10cf2" />

## 📑 Table of Contents

- [MLflow Deployment & Inference Guide](#mlflow-deployment--inference-guide)  
  - [Inference Wrapper](#inference-wrapper)
    - [Method 1 — Define wrapper after loading the MLflow model](#method-1--define-wrapper-after-loading-the-mlflow-model)
    - [Method 2 - custom python models: Define wrapper during model logging](#method-2---custom-python-models-define-wrapper-during-model-logging)
    - [Use MLflow Model Signature + Input Example](#use-mlflow-model-signature--input-example)
  - [Deployment](#deployment)
    - [Setting up a Batch Inference Job](#setting-up-a-batch-inference-job)
    - [Creating an API Process for Inference](#creating-an-api-process-for-inference)


---

## Inference Wrapper

MLflow uses a **PyFunc interface** so that any model (PyTorch, TensorFlow, sklearn, custom code)  can expose a **standard `predict()` API**.  
An inference wrapper defines **how the model is loaded** and **how predictions are produced**.

There are two main ways to define the wrapper:


## Method 1 — Define wrapper after loading the MLflow model

You keep MLflow’s model simple and define a wrapper outside MLflow:

```python
self.model = mlflow.pyfunc.load_model(model_uri)

def predict(self, df):
    preds = self.model.predict(df)
    ...
```
[**Example:**](https://github.com/GeneSUN/mlflow_toolbox/blob/main/mlops_demo_project/inference/inference_wrapper.py)

```python
class InferenceWrapper:
    def __init__(
        self,
        model_uri: str = "models:/mlops_demo_classifier/Production",
    ):
        # This loads the model from the MLflow Model Registry
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(df)

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
```

- When you expect to modify inference logic frequently.
- When you want to combine MLflow models with:
  - pipelines
  - additional business rules
  - ensemble logic
  - multi-model cascades



### Method 2 - custom python models: Define wrapper during model logging 

MLflow allows you to package inference logic **inside the model** by implementing the  
`PythonModel` interface. This makes the model fully self-contained.

Use this when:
- You want custom inference, Preprocessing, scaling, reshaping, or threshold logic must live **inside** the model.

MLflow has a standard interface, with two functions:
```python
class PythonModel:
    def load_context(self, context):
        ...

    def predict(self, context, model_input):
        ...
```
We implement this interface to wrap our Autoencoder model.

[Example:](https://github.com/GeneSUN/NetSignalOutlierPipeline/blob/main/src/modeling/global_autoencoder/train_autoencoder.py#L72)
```python
mlflow.pyfunc.log_model(
    artifact_path="pyfunc_model",
    python_model=AutoencoderWrapper(),
    artifacts={"autoencoder": "autoencoder.pkl"},
    registered_model_name="Autoencoder_Anomaly_Detection"
)
```

- When you want to control inference logic inside the MLflow model itself (preprocessing, scaling, windowing, postprocessing, thresholds).
- When serving with MLflow Models, MLflow Server, Docker, or Batch Inference.

Reference: https://mlflow.org/docs/latest/ml/model/#deploy-mlflow-models



## Use MLflow Model Signature + Input Example

Signatures define:
- Expected input schema  
- Data types  
- Column ordering  

This ensures **automatic validation** during inference and prevents schema drift.

You add this during logging:

```python
mlflow.pyfunc.log_model(
    python_model=MyModel(),
    signature=signature,
    input_example=example,
)
```

Useful for:
- Production pipelines where schema mismatch is common  
- API services that need strict validation  
- Ensuring consistent inference across jobs, clusters, and teams  



---

# Deployment

Once a model is logged and registered, it can be used in multiple deployment patterns.  
Two common setups are batch inference and online API serving.

---

## Setting up a Batch Inference Job

Batch jobs are ideal for:
- Daily or hourly scoring  
- Re-scoring large datasets (millions of records)  
- Offline analytics or monitoring tasks  

A typical workflow:

1. Load the model from the registry  
2. Load input data from storage (HDFS, S3, BigQuery…)  
3. Apply the inference wrapper  
4. Save outputs to downstream tables or dashboards  

<img width="747" height="451" alt="Screenshot 2025-12-09 at 7 45 29 PM" src="https://github.com/user-attachments/assets/0ced4d5d-5403-4643-aa55-2fd38ce88f88" />

---


## Creating an API Process for Inference

An API is needed when you require:
- Real-time predictions  
- Low-latency scoring  
- Integration with customer-facing services  

A typical workflow:

1. Start a FastAPI / Flask service  
2. Load the MLflow model once at startup  
3. Pass incoming requests through the inference wrapper  
4. Return predictions to clients  
<img width="738" height="239" alt="Screenshot 2025-12-09 at 7 46 05 PM" src="https://github.com/user-attachments/assets/1737be4a-b37b-47bb-8efc-04e130ddcf73" />




