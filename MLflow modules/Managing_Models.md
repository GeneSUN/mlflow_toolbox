
## 🎁 MLflow Models = Product Packaging + Instruction Manual

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


## 🏭 Model Registry = The Warehouse for All Products

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



