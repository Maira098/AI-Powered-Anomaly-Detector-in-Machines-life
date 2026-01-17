# src/inference.py
import joblib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# ---------- Load saved models & metadata ----------

# LSTM model and scaler
lstm_scaler = joblib.load("models/lstm_scaler.pkl")
lstm_model = tf.keras.models.load_model("models/lstm_anomaly_model.h5")

# Load LSTM metadata
with open("models/meta_lstm.pkl", "rb") as f:
    meta_lstm = pickle.load(f)
seq_feature_cols = meta_lstm["seq_feature_cols"]
SEQ_LEN = meta_lstm.get("SEQ_LEN", 1)

# LSTM feature columns (13 features)
LSTM_FEATURE_COLS = [
    "equipment_ID", "alarm",
    "elapsed", "pi", "po", "speed",
    "efficiency", "scrap", "scrap_rate", "tp_sec",
    "hour", "dayofweek", "is_weekend",
]

# Default threshold for anomaly detection
DEFAULT_THRESHOLD = 0.8


# ---------- Helper functions ----------

def engineer_features_for_lstm(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    # Parse timestamps
    df["interval_start"] = pd.to_datetime(df["interval_start"], format="mixed", utc=True)
    df["start"] = pd.to_datetime(df["start"], format="mixed", utc=True)
    df["end"] = pd.to_datetime(df["end"], format="mixed", utc=True)

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Filter for relevant types
    normal_states = ["production"]
    anomaly_states = ["downtime", "performance_loss"]
    df = df[df["type"].isin(normal_states + anomaly_states)].copy()

    # Create label (if 'type' column exists)
    if "type" in df.columns:
        df["label"] = df["type"].apply(lambda x: 1 if x in anomaly_states else 0)

    # Basic numeric features
    pi_safe = df["pi"].replace(0, 1)
    df["efficiency"] = df["po"] / pi_safe
    df["scrap"] = df["pi"] - df["po"]
    df["scrap_rate"] = df["scrap"] / pi_safe
    df["tp_sec"] = df["po"] / df["elapsed"].replace(0, 1)

    # Time features
    df["hour"] = df["interval_start"].dt.hour
    df["dayofweek"] = df["interval_start"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Label encode categorical features
    for col in ["equipment_ID", "alarm"]:
        if col in df.columns:
            le = LabelEncoder()
            vals = df[col].astype(str)
            le.fit(vals)
            df[col] = le.transform(vals)

    return df


def build_lstm_sequences(df_feat: pd.DataFrame, seq_len: int = None) -> tuple:

    if seq_len is None:
        seq_len = SEQ_LEN

    # If seq_len is 1, we can use simpler approach
    if seq_len == 1:
        X = df_feat[seq_feature_cols].values.astype("float32")
        X_scaled = lstm_scaler.transform(X)
        X_seqs = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        meta_df = df_feat[["equipment_ID"]].copy()
        if "interval_start" in df_feat.columns:
            meta_df["time"] = df_feat["interval_start"]

        return X_seqs, meta_df

    # For longer sequences, build sliding windows
    df_feat = df_feat.sort_values(["equipment_ID", "interval_start"])
    X_list = []
    meta_rows = []

    for eq_id, grp in df_feat.groupby("equipment_ID"):
        grp = grp.reset_index(drop=True)
        vals = lstm_scaler.transform(grp[seq_feature_cols].values.astype("float32"))

        for i in range(len(grp) - seq_len + 1):
            window = vals[i:i + seq_len]
            X_list.append(window)
            meta_rows.append({
                "equipment_ID": eq_id,
                "time": grp.loc[i + seq_len - 1, "interval_start"]
            })

    if len(X_list) == 0:
        return np.array([]), pd.DataFrame()

    X_seqs = np.array(X_list)
    meta_df = pd.DataFrame(meta_rows)
    return X_seqs, meta_df


# ---------- Public inference functions ----------

def predict_lstm(
        df_raw: pd.DataFrame,
        threshold: float = DEFAULT_THRESHOLD,
        return_full: bool = False
) -> pd.DataFrame:

    # Engineer features
    df_feat = engineer_features_for_lstm(df_raw)

    # Build sequences
    X_seqs, meta_df = build_lstm_sequences(df_feat)

    if len(X_seqs) == 0:
        return pd.DataFrame(columns=["equipment_ID", "time", "anomaly_prob", "anomaly_label"])

    # Predict
    y_prob = lstm_model.predict(X_seqs, verbose=0).ravel()
    labels = (y_prob >= threshold).astype(int)

    # Add predictions to metadata
    meta_df["anomaly_prob"] = y_prob
    meta_df["anomaly_label"] = labels

    if return_full:
        # Merge with original features
        result_df = df_feat.copy()
        result_df["anomaly_prob"] = y_prob[:len(result_df)]
        result_df["anomaly_label"] = labels[:len(result_df)]
        return result_df

    return meta_df


def predict_lstm_batch(
        df_raw: pd.DataFrame,
        equipment_ids: list = None,
        threshold: float = DEFAULT_THRESHOLD
) -> dict:

    if equipment_ids is None:
        equipment_ids = df_raw["equipment_ID"].unique()

    results = {}
    for eq_id in equipment_ids:
        df_eq = df_raw[df_raw["equipment_ID"] == eq_id].copy()
        if not df_eq.empty:
            results[eq_id] = predict_lstm(df_eq, threshold=threshold)

    return results


def get_anomaly_summary(predictions_df: pd.DataFrame) -> dict:

    if predictions_df.empty:
        return {
            "total_intervals": 0,
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "avg_anomaly_score": 0.0,
            "max_anomaly_score": 0.0,
            "min_anomaly_score": 0.0,
        }

    total = len(predictions_df)
    anomaly_count = int(predictions_df["anomaly_label"].sum())
    anomaly_percentage = (anomaly_count / total) * 100

    return {
        "total_intervals": total,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(anomaly_percentage, 2),
        "avg_anomaly_score": float(predictions_df["anomaly_prob"].mean()),
        "max_anomaly_score": float(predictions_df["anomaly_prob"].max()),
        "min_anomaly_score": float(predictions_df["anomaly_prob"].min()),
        "std_anomaly_score": float(predictions_df["anomaly_prob"].std()),
    }


def evaluate_lstm_predictions(predictions_df: pd.DataFrame, true_labels: np.ndarray = None) -> dict:

    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score
    )

    if true_labels is None:
        return {"error": "No ground truth labels provided"}

    y_pred = predictions_df["anomaly_label"].values
    y_prob = predictions_df["anomaly_prob"].values

    # Calculate metrics
    cm = confusion_matrix(true_labels, y_pred)
    f1 = f1_score(true_labels, y_pred)
    precision = precision_score(true_labels, y_pred)
    recall = recall_score(true_labels, y_pred)

    try:
        roc_auc = roc_auc_score(true_labels, y_prob)
    except:
        roc_auc = None

    report = classification_report(true_labels, y_pred, output_dict=True)

    return {
        "confusion_matrix": cm.tolist(),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "classification_report": report,
    }


# ---------- Convenience functions for compatibility ----------

def predict(df_raw: pd.DataFrame, model_type: str = "lstm", **kwargs):

    if model_type.lower() != "lstm":
        raise ValueError(f"Only 'lstm' model is supported. Got: {model_type}")

    return predict_lstm(df_raw, **kwargs)


# ---------- Example usage ----------

if __name__ == "__main__":
    # Example: Load data and run predictions
    print("LSTM-only Inference Module")
    print(f"Loaded LSTM model from: models/lstm_anomaly_model.h5")
    print(f"Loaded scaler from: models/lstm_scaler.pkl")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Features: {seq_feature_cols}")
    print(f"Default threshold: {DEFAULT_THRESHOLD}")

    # Example usage:
    # df = pd.read_csv("data/raw_data.csv")
    # predictions = predict_lstm(df, threshold=0.8)
    # summary = get_anomaly_summary(predictions)
    # print(summary)