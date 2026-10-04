import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, f1_score
)

# --- Données ---
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def build_preprocess():
    # Normalise Time et Amount, laisse V1-V28 inchangées
    return ColumnTransformer(
        [("scale", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
    )

# --- MLflow ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-card-fraud")

def train_and_log(run_name, classifier):
    pipeline = Pipeline([
        ("preprocess", build_preprocess()),
        ("model", classifier),
    ])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", type(classifier).__name__)
        mlflow.log_params(classifier.get_params())
        mlflow.log_param("train_rows", len(X_train))

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "pr_auc": average_precision_score(y_test, y_score),
            "precision_fraud": precision_score(y_test, y_pred),
            "recall_fraud": recall_score(y_test, y_pred),
            "f1_fraud": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
              pipeline,
              name="model",
              input_example=X_test.head(3),
              registered_model_name="fraud-detector" if run_name == "random_forest" else None,
              skops_trusted_types=["sklearn.tree._tree.Tree"],
)
        print(run_name, metrics)

train_and_log(
    "logreg",
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
)
train_and_log(
    "random_forest",
    RandomForestClassifier(n_estimators=100, class_weight="balanced",
                           random_state=42, n_jobs=-1),
)