import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    print("[INFO] MLflow CI Training started")

    # ========================
    # SET TRACKING URI
    # ========================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("LogisticRegression_Basic")

    print(f"[INFO] MLflow Tracking URI: {tracking_uri}")

    # ========================
    # AUTOLOG (WAJIB)
    # ========================
    mlflow.sklearn.autolog()

    # ========================
    # LOAD DATA
    # ========================
    train_df = pd.read_csv("dataset_preprocessing/credit_train_preprocessed.csv")
    test_df  = pd.read_csv("dataset_preprocessing/credit_test_preprocessed.csv")

    target_column = "default"
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test  = test_df.drop(columns=[target_column])
    y_test  = test_df[target_column]

    # ========================
    # TRAINING
    # ========================
    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ========================
    # MANUAL METRICS (PELENGKAP)
    # ========================
    mlflow.log_metric("accuracy_manual", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_manual", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_manual", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score_manual", f1_score(y_test, y_pred))

    print("[SUCCESS] Training selesai & artefak tersimpan di MLflow")

if __name__ == "__main__":
    main()
