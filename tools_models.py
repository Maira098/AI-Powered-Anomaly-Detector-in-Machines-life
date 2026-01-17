# tools_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import joblib
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

# ---------------- LOAD PRETRAINED MODELS ----------------

LSTM_SCALER_PATH = "models/lstm_scaler.pkl"  # StandardScaler for 13 seq features
LSTM_MODEL_PATH = "models/lstm_anomaly_model.h5"  # LSTM model

lstm_scaler = joblib.load(LSTM_SCALER_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)

# ---------------- FEATURE LISTS ----------------

# LSTM seq_feature_cols (13 features, scaled)
LSTM_FEATURE_COLS = [
    "equipment_ID", "alarm",
    "elapsed", "pi", "po", "speed",
    "efficiency", "scrap", "scrap_rate", "tp_sec",
    "hour", "dayofweek", "is_weekend",
]

DEFAULT_THRESHOLD = 0.8


# --------- HELPER: RECREATE FEATURES EXACTLY AS IN COLAB ---------

def _build_feature_frame(raw_data: list[dict]) -> pd.DataFrame:
    """
    Build feature dataframe from raw machine interval data.
    Creates engineered features matching the training process.
    """
    df = pd.DataFrame(raw_data).copy()

    # time parsing
    df["interval_start"] = pd.to_datetime(df["interval_start"], format="mixed", utc=True)
    df["start"] = pd.to_datetime(df["start"], format="mixed", utc=True)
    df["end"] = pd.to_datetime(df["end"], format="mixed", utc=True)

    df = df.drop_duplicates().reset_index(drop=True)

    # label from type
    normal_states = ["production"]
    anomaly_states = ["downtime", "performance_loss"]
    df = df[df["type"].isin(normal_states + anomaly_states)].copy()
    df["label"] = df["type"].apply(lambda x: 1 if x in anomaly_states else 0)

    # basic numeric features
    pi_safe = df["pi"].replace(0, 1)
    df["efficiency"] = df["po"] / pi_safe
    df["scrap"] = df["pi"] - df["po"]
    df["scrap_rate"] = df["scrap"] / pi_safe
    df["tp_sec"] = df["po"] / df["elapsed"].replace(0, 1)

    df["hour"] = df["interval_start"].dt.hour
    df["dayofweek"] = df["interval_start"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # label-encode equipment_ID and alarm
    # LSTM used label-encoded versions
    for col in ["equipment_ID", "alarm"]:
        le = LabelEncoder()
        vals = df[col].astype(str)
        le.fit(vals)
        df[col] = le.transform(vals)

    return df


# ---------------- LSTM TOOL ----------------

def lstm_tool(data):

    df_full = _build_feature_frame(data)

    X = df_full[LSTM_FEATURE_COLS].values.astype("float32")
    X_scaled = lstm_scaler.transform(X)

    # [samples, timesteps, features] with 1-timestep sequences for inference
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    y_proba = lstm_model.predict(X_seq, verbose=0).reshape(-1)

    df_full["anomaly_score"] = y_proba

    # Use the optimal threshold from your analysis
    best_thr = DEFAULT_THRESHOLD
    df_full["prediction"] = (y_proba >= best_thr).astype(int)

    return df_full.to_json(orient="records")


# ---------------- EVALUATION TOOL ----------------

def evaluation_tool(lstm_json):
    """
    Compute confusion matrix, F1 score, ROC-AUC and plots from LSTM JSON output.
    Assumes df has columns: label, prediction, anomaly_score.
    """
    df = pd.read_json(StringIO(lstm_json))

    if not {"label", "prediction", "anomaly_score"}.issubset(df.columns):
        raise ValueError("LSTM JSON must contain 'label', 'prediction', and 'anomaly_score' columns")

    y_true = df["label"]
    y_pred = df["prediction"]
    y_score = df["anomaly_score"]

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    report = classification_report(y_true, y_pred)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM ROC Curve")
    plt.show()

    # Anomaly Score Plot
    plt.figure()
    plt.plot(y_score)
    plt.title("LSTM Anomaly Scores")
    plt.xlabel("Index")
    plt.ylabel("Score")
    plt.show()

    return {
        "confusion_matrix": cm.tolist(),
        "f1_score": f1,
        "roc_auc": roc_auc,
        "classification_report": report,
    }