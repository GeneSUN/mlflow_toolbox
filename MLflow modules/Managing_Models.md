## Understanding models in MLflow

### 🎁 MLflow Models = Product Packaging + Instruction Manual

What MLflow Models does is like:
- 🛍️ Packaging the productin a standardized box(MLmodel directory)
- 📄 Labeling the product Model Registry = The Warehouse System
- instructions
  - How to open it/How to use it
  - What input it expects
  - What output it produces(model signature + conda.yaml)

When you run:
```python
mlflow.sklearn.log_model(model, "model")
```
MLflow Models handles:
- create MLmodel file
- save the pickled model
- write conda.yaml
- log flavors
- set inference schema

BUT the model is still just an **artifact inside a run.**


### 🏭 Model Registry = The Warehouse for All Products

The Model Registry manages:

- 📚 Inventory catalog: Keeps track of all products (registered models)
- 🔢 Versioning: Product v1, v2, v3…
- 🏷️ Status labels: Prototype/Testing/Production/Retired
- 👤 Metadata: Who made it/When/Which run it came from

```python
mlflow.register_model("runs:/<run_id>/model", "wifi_scoring_model")
```

Model Registry now:
- creates a Registered Model named wifi_scoring_model
- creates Version 1
- stores metadata
- assigns lifecycle state (None, Staging, etc.)

---

👉 MLflow Models = how the product is prepared, packaged, and made usable. <br>
👉 Model Registry = the warehouse system that stores, tracks, and manages all versions of all products.

### MLflow Model format acts like an adapter

<img width="888" height="485" alt="Screenshot 2025-11-16 at 10 19 43 AM" src="https://github.com/user-attachments/assets/fb7c2b96-4475-4510-a753-b339efcfbf43" />

### MLflow model file


<img width="1187" height="377" alt="Screenshot 2025-11-16 at 10 43 48 AM" src="https://github.com/user-attachments/assets/389c372e-55fb-4ced-984e-63df1e5685b8" />


## Managing Model Signatures and Schemas

MLflow wants to guarantee that the data your model receives during inference (prediction) matches the data it was trained on.

To do this, MLflow stores a signature—a description of the inputs and outputs of the model.

### ⭐ 1. What is a Model Signature?

A model signature is a JSON specification that records:
- the number of input columns
- the names of input columns
- the data types of input columns
- the type/shape of prediction output

```python
signature:
  inputs: 
      - {type: "double"}
      - {type: "double"}
      ...
  outputs:
      - {type: "long"}
```

### How MLflow Creates a Signature Automatically

```
mlflow.sklearn.autolog()
mlflow.tensorflow.autolog()

# MLflow will try to infer the signature automatically when you call:
model.fit(X_train, y_train)

```



