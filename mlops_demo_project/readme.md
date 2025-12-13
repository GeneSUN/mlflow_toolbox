# Project Structure
```
mlops_demo_project/
│
├── config/
│   ├── train.yaml
│   ├── eval.yaml
│   ├── inference.yaml
│   └── monitoring.yaml
│
├── modeling/
│   ├── trainer.py
│   ├── data_loader.py
│   └── preprocess.py
│
├── evaluation/
│   ├── evaluate_model.py
│   ├── compare_runs.py
│   ├── validate_schema.py
│   └── tune.py
│
├── inference/
│   ├── inference_wrapper.py
│   └── api_server.py  # FastAPI server (optional)
│
├── monitoring/
│   ├── drift_monitor.py
│   ├── prediction_monitor.py
│   └── data_quality.py
│
├── pipelines/
│   ├── train_pipeline.py
│   ├── evaluate_pipeline.py
│   └── nightly_monitoring.py
│
├── tests/
│   ├── test_trainer.py
│   ├── test_inference.py
│   └── test_monitoring.py
│
├── scripts/
│   ├── run_train.sh
│   ├── run_evaluate.sh
│   └── run_inference.sh
│
├── requirements.txt
├── Dockerfile
└── README.md


```
