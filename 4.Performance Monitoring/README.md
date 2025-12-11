
# Monitoring with MLflow: a cleaner structure

For any ML project, you can think of **four things to monitor over time**:

1. **Inputs (X)** – explanatory variables / features  
2. **Targets (Yᵗʳᵘᵉ)** – the response variable / label  
3. **Model outputs (Yᵖʳᵉᵈ)** – predictions and residuals  
4. **Infrastructure / system metrics** – latency, cost, errors, etc.

Below is a refined version of what you described, with your examples structured under each type of drift.

[Colab Notebook Example](https://colab.research.google.com/drive/1sNPQNRpDo3Pg6JAw6hmBNRAjdMFvalbh#scrollTo=seuQanAMo2Dl)


---

## 1. Feature drift (input / data drift)

**What it is**

- The distribution of **X** changes over time:
  
$$ P_t(X) \neq P_{t+\Delta}(X) $$

- Even if the model parameters haven’t changed, it will see “new kinds” of inputs it wasn’t trained on.

**Your house price example**

- Features: location, population density, neighborhood income, etc.
- Over several years:
  - Population grows or declines.
  - Government policy changes (e.g., new subway line, school zoning).
  - Previously “poor” regions become gentrified and prices rise.
- The model is still the same, but the **feature distribution** has shifted.  


**Your churn-duration example**

- Feature: `tenure` or “months since subscription”.
- Initially:
  - The longer a customer stays, the higher their probability of churn (a typical increasing hazard function).
- After marketing introduces a **4-month discount**:
  - Many customers cancel right after the discount ends.
  - The distribution of churn by tenure spikes sharply at **month 4**.
- Here, both the **feature distribution** and the relationship between `tenure` and churn change, driven by a new business policy.

In MLflow, you’d typically:

- Log **feature distributions** (summary stats, histograms) regularly as artifacts.
- Log drift metrics (e.g., PSI per feature) as **metrics** for each monitoring run.

---

## 2. Target drift (label / response drift)

**What it is**

- The distribution of **Yᵗʳᵘᵉ** changes over time:
  
$$ P_t(Y) \neq P_{t+\Delta}(Y) $$

- Even if X looks similar, the world’s behavior changes, so the realized labels shift.

**Your inventory forecasting example**

- Task: forecast inventory usage / cargo volume.
- Inputs X: historical demand, promotions, seasonality, maybe macro indicators.
- Then a **tariff increase** happens:
  - Customer demand drops.
  - Actual cargo volume and inventory usage become systematically lower than before,
    **even though X might look similar** (at least for a while).


In MLflow, you’d log:

- Time-stamped evaluation metrics (per day / week).
- Periodic label distribution stats as metrics or artifacts.

---

## 3. Model / concept drift (relationship between X and Y)

Here it’s useful to separate two ideas:

- **Concept drift**: the true relationship between X and Y changes:  
  
$$ P_t(Y|X) \neq P_{t+\Delta}(Y|X) $$

- **Model drift**: the model’s predictions Yᵖʳᵉᵈ (and its error profile) degrade over time because it no longer matches the current concept.

**Your 4G vs 5G usage example**

- You built a **5G data usage forecasting model**.
- Then:
  - New hardware is deployed.
  - Network behavior changes: capacity increases, QoS controls change, 5G becomes less sensitive to load compared with 4G.
- The relationship “load → throughput / usage patterns” changes:
  - Under 4G, performance was strongly tied to the number of users and activity.
  - Under 5G, the same activity level doesn’t degrade performance as much.

Even if feature and target distributions don’t look too crazy, the **mechanism** that maps X to Y has changed → **concept drift**. The old model underfits the new regime.

You detect this by monitoring:

- **Prediction distribution**: is Yᵖʳᵉᵈ systematically higher/lower than before?
- **Residuals** (Yᵗʳᵘᵉ − Yᵖʳᵉᵈ): does the error suddenly become biased or more volatile?
- Performance curves over time: accuracy, RMSE, AUROC, etc., on recent data.

**Anomaly detection / fraud example**

- Unsupervised:
  - You train an anomaly detector on historical “normal” behavior.
  - Fraudsters change their tactics (new transaction patterns, devices, locations).
  - The definition of “normal vs abnormal” changes → concept drift.
- Supervised:
  - Fraud labels may be delayed or biased (only caught fraud becomes Y=1).
  - Fraud patterns evolve (new attack vectors, new collusion strategies).

Monitoring strategy:

- Track the **rate of anomalies** flagged over time.
- Track the **distribution** of anomaly scores.
- For supervised fraud models, track:
  - Positive rate (fraction labeled fraud).
  - Precision/recall over time (once labels become available).

In MLflow, you can log:

- Metrics for **per-batch performance** (e.g., AUROC on a rolling window).
- Histograms / quantiles of anomaly scores as artifacts.
- Flags when performance drops below thresholds.

---

## 4. Infrastructure & operational metrics

Even if the model and data are perfect, the **system can fail operationally**.

**Your real-time anomaly detection example**

- New network performance logs arrive every **5 minutes**.
- Your pipeline must:
  1. Ingest logs.
  2. Run inference / anomaly detection.
  3. Emit alerts.
  - **All within 5 minutes**.
- The key metric is **end-to-end latency**; if it exceeds 5 minutes, alerts become stale.

Operational metrics to monitor:

- **Latency** per component (ingestion, feature computation, model inference).
- **Throughput** (rows / second, requests / second).
- **Resource usage**: CPU, memory, GPU, I/O.
- **Error rates**: failed jobs, timeouts, exceptions.
- **Cost**: e.g., compute-hours per day, number of workers.

In MLflow, you can:

- Log latency and throughput as **metrics** for each batch / run.
- Store system profiles or logs as **artifacts**.
- Version different deployment configs (model version, hardware, parallelism) via **params**.

---

## Evidently

**eval workflow using the Evidently library**

- https://docs.evidentlyai.com/docs/library/evaluations_overview
- https://docs.evidentlyai.com/quickstart_ml
- https://colab.research.google.com/drive/1ui09fKTL7jaMrpEIXBn3l9mKLAVDXGBp

<img width="429" height="1259" alt="Untitled" src="https://github.com/user-attachments/assets/177c31f0-8904-487b-968a-c92da5a7d768" />

**Log in Mlflow**

<img width="1446" height="673" alt="Screenshot 2025-12-11 at 11 40 39 AM" src="https://github.com/user-attachments/assets/090bd600-3e8b-4c2a-a290-4e0fc28ef09d" />


**Example of Data Drift Report**
- https://colab.research.google.com/drive/1ui09fKTL7jaMrpEIXBn3l9mKLAVDXGBp#scrollTo=gELZirCL6ZhO


