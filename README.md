MLOps Fraud Detection Pipeline

An end-to-end MLOps pipeline for credit card fraud detection: experiment tracking, model registry, and a dockerized inference API — from training to a servable, containerized model.

Context

Beyond building an accurate model, this project demonstrates a production-oriented ML workflow: tracked experiments, a versioned model registry, and an API that serves predictions independently of how or where the model was trained — the core practices expected in an MLOps role.

Dataset

Source: Credit Card Fraud Detection (Kaggle) — anonymized European card transactions from September 2013.

Property	Value
Rows	284,807
Fraud cases	492 (0.17%)
Features	Time, V1-V28 (PCA-anonymized), Amount
Target	Class (0 = normal, 1 = fraud)

The extreme class imbalance (1 fraud per ~580 transactions) makes accuracy meaningless as a metric; evaluation focuses on precision, recall, and PR-AUC for the fraud class.

Methodology
Exploration & split — stratified 80/20 train/test split to preserve the 0.17% fraud ratio in both sets.
Preprocessing — Time and Amount standardized (the V1-V28 features are already PCA-scaled); preprocessing and model are combined in a single scikit-learn Pipeline so the exact same transformation is applied at training and inference time.
Modeling — two models compared: Logistic Regression and Random Forest, both with class_weight="balanced".
Tracking — every run logged to MLflow (parameters, metrics, model artifact) against a local SQLite backend.
Model registry — the selected model registered under the name fraud-detector, so the API always resolves the latest version by name rather than a run ID.
Serving — a FastAPI service loading models:/fraud-detector/latest at startup, exposing /health and /predict.
Containerization — both training and serving run as separate Docker Compose services, sharing the MLflow store through mounted volumes, so the model registry uses consistent (Linux) paths regardless of where training was last run.
Results
Model	Precision (fraud)	Recall (fraud)	F1 (fraud)	PR-AUC
Logistic Regression	0.061	0.918	0.114	0.716
Random Forest	0.919	0.806	0.859	0.865

Random Forest was selected as the production model. Logistic Regression's high recall comes at the cost of a 94% false-alarm rate, which would make its alerts unusable by a fraud team in practice. Random Forest's PR-AUC (0.865 vs 0.716) confirms this is a structural difference, not an artifact of the default decision threshold.

API
GET  /health    → {"status": "ok", "model": "fraud-detector"}
POST /predict    → {"fraud_probability": 0.6, "is_fraud": true}

/predict accepts the 30 raw features of a transaction (Time, V1-V28, Amount) and returns a fraud probability plus a boolean decision at the default 0.5 threshold.

Running locally
bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt

python src/02_train_mlflow.py       # trains both models, registers the best one
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs at http://127.0.0.1:5000
uvicorn src.api:app --reload --port 8000            # serve the API
Running with Docker
bash
docker compose run --rm trainer     # train inside the container (writes mlflow.db with container-native paths)
docker compose up --build api       # serve the registered model

Note: training must be run inside the same environment (local or Docker) that will later serve the model. MLflow stores absolute artifact paths at training time; running 02_train_mlflow.py on Windows and then serving from a Linux container produces a No such artifact error, since the paths recorded in mlflow.db don't resolve inside the container. Training via docker compose run trainer keeps paths consistent with the api service.

Key learnings / limitations
scikit-learn's recent model serialization (skops) requires explicitly trusting sklearn.tree._tree.Tree (skops_trusted_types) before a RandomForestClassifier can be logged — a deliberate security guard against untrusted pickle-style files, not a bug.
A Docker bind mount to a file that doesn't yet exist on the host creates a directory instead, which breaks SQLite (unable to open database file). The fix is to create an empty file before mounting it.
Training and serving were split into separate Compose services specifically to avoid host/container path mismatches in the MLflow registry — a concrete illustration of why environment parity matters in MLOps.
Author

Emna Thouabtia — Computer Engineering Student, ENICarthage