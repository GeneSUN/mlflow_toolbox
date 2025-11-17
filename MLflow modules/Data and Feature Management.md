# “Medallion Architecture” (Bronze → Silver → Gold)

<img width="2288" height="1100" alt="image" src="https://github.com/user-attachments/assets/c63258d2-f1c7-4494-8012-ee3b1df6e793" />


1. Store it as-is (raw)
2. Validate it (staged)
3. Transform it for ML (training)

| Folder       | Purpose                                 |
| ------------ | --------------------------------------- |
| **raw**      | Downloaded exactly as received (Bronze) |
| **staged**   | Cleaned + validated (Silver)            | 
| **training** | Feature engineered & model-ready (Gold) |


| **Folder / Layer**  | **Purpose**                                                                                     | **Common Behaviors, Checks, and Metrics**                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **raw** (Bronze)    | Raw data exactly as received from source (API, logs, batch dump). Immutable.                    | - Store unmodified source data<br>- Timestamp + source tracking<br>- Basic ingestion checks (row count, file size)<br>- Detect missing files or incomplete batches<br>- Log freshness (data lag)<br>- Version raw snapshots for reproducibility<br>- No schema enforcement yet                                                                                                                                                                                 |
| **staged** (Silver) | Cleaned, validated, schema-enforced data. Suitable for downstream transformation.               | - **Schema validation** (column names/types)<br>- **Null % checks**, missing value patterns<br>- **Duplicate detection**, deduplication<br>- **Anomaly detection in raw fields** (min/max violations)<br>- **Drift detection** vs historical data<br>- Type coercion (string→numeric)<br>- Validate data ranges and integrity<br>- Early feature consistency checks<br>- Log validation metrics in MLflow                                                      |
| **training** (Gold) | Final ML-ready dataset with clean features, engineered transformations, and consistent formats. | - Feature engineering and transformation<br>- Normalization, scaling, encoding<br>- Time-window feature creation<br>- Feature consistency checks vs feature store<br>- **Compatibility with feature stores** (Feast, Databricks)<br>- Final schema enforcement<br>- Balanced/stratified sampling checks<br>- Train/test split validation<br- Log feature stats + distribution (mean, std, histograms)<br>- Discard rows flagged as low-quality in Silver layer |
