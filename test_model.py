#!/usr/bin/env python3
"""
Test script to find optimal threshold for your LSTM model.
This will analyze your data and suggest the best threshold value.
"""

import pandas as pd
import numpy as np
import json
from tools_models import lstm_tool, _build_feature_frame, LSTM_FEATURE_COLS, lstm_scaler, lstm_model
import matplotlib.pyplot as plt


def analyze_scores(csv_path, equipment_id="s_3", max_rows=5000):
    """
    Analyze anomaly score distribution for a given machine.

    Args:
        csv_path: Path to raw_data.csv
        equipment_id: Machine ID to analyze
        max_rows: Maximum rows to process
    """
    print("=" * 70)
    print("🔍 LSTM Anomaly Score Analysis")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total rows: {len(df)}")

    # Filter for specific machine
    df_machine = df[df["equipment_ID"] == equipment_id].head(max_rows)
    print(f"   Rows for {equipment_id}: {len(df_machine)}")

    if df_machine.empty:
        print(f"❌ No data found for equipment_ID: {equipment_id}")
        return

    # Convert to records and build features
    records = df_machine.to_dict(orient="records")
    df_full = _build_feature_frame(records)

    # Get LSTM predictions
    print(f"\n🧠 Running LSTM model...")
    X = df_full[LSTM_FEATURE_COLS].values.astype("float32")
    X_scaled = lstm_scaler.transform(X)
    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y_proba = lstm_model.predict(X_seq, verbose=0).reshape(-1)

    # Score statistics
    print("\n" + "=" * 70)
    print("📊 ANOMALY SCORE DISTRIBUTION")
    print("=" * 70)
    print(f"Minimum:      {y_proba.min():.4f}")
    print(f"5th %ile:     {np.percentile(y_proba, 5):.4f}")
    print(f"25th %ile:    {np.percentile(y_proba, 25):.4f}")
    print(f"Median:       {np.median(y_proba):.4f}")
    print(f"Mean:         {y_proba.mean():.4f} ± {y_proba.std():.4f}")
    print(f"75th %ile:    {np.percentile(y_proba, 75):.4f}")
    print(f"95th %ile:    {np.percentile(y_proba, 95):.4f}")
    print(f"Maximum:      {y_proba.max():.4f}")

    # Test different thresholds
    print("\n" + "=" * 70)
    print("🎯 THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [0.40, 0.45, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.80]
    results = []

    print(f"\n{'Threshold':<12} {'Anomalies':<15} {'%':<10} {'Status'}")
    print("-" * 70)

    for thr in thresholds:
        predictions = (y_proba >= thr).astype(int)
        anomaly_count = predictions.sum()
        anomaly_pct = (anomaly_count / len(predictions)) * 100

        # Determine if this is a good threshold (5-25% anomalies is typical)
        if 5 <= anomaly_pct <= 25:
            status = "✅ GOOD"
        elif 1 <= anomaly_pct < 5:
            status = "⚠️  Few anomalies"
        elif 25 < anomaly_pct <= 40:
            status = "⚠️  Many anomalies"
        elif anomaly_pct > 40:
            status = "❌ Too sensitive"
        else:
            status = "❌ None detected"

        print(f"{thr:<12.2f} {anomaly_count:>5}/{len(predictions):<8} {anomaly_pct:>6.1f}%   {status}")

        results.append({
            'threshold': thr,
            'count': anomaly_count,
            'percentage': anomaly_pct,
            'status': status
        })

    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)

    # Find percentile-based thresholds
    conservative = np.percentile(y_proba, 80)
    balanced = np.percentile(y_proba, 65)
    sensitive = np.percentile(y_proba, 50)

    print(f"\n🎯 Based on your data distribution:")
    print(f"   Conservative (few anomalies):  {conservative:.3f}")
    print(f"   Balanced (recommended):        {balanced:.3f} ⭐")
    print(f"   Sensitive (more anomalies):    {sensitive:.3f}")

    # Check if current scores suggest mostly normal operation
    if y_proba.mean() < 0.6:
        print(f"\n📊 Your data shows mostly NORMAL operation (avg score: {y_proba.mean():.3f})")
        print(f"   This is GOOD! Lower threshold to 0.50-0.55 to detect anomalies.")
    else:
        print(f"\n⚠️  Your data shows higher anomaly scores (avg score: {y_proba.mean():.3f})")
        print(f"   Consider investigating. Threshold 0.65-0.70 may be appropriate.")

    # If there are true labels, calculate optimal threshold
    if "label" in df_full.columns:
        print("\n" + "=" * 70)
        print("🎓 OPTIMAL THRESHOLD (based on ground truth)")
        print("=" * 70)

        from sklearn.metrics import roc_curve, f1_score

        y_true = df_full["label"].values
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)

        # Find threshold that maximizes F1 score
        f1_scores = []
        for thr in thresholds:
            preds = (y_proba >= thr).astype(int)
            f1 = f1_score(y_true, preds)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"\n🏆 Optimal threshold: {best_threshold:.3f}")
        print(f"   F1 Score: {best_f1:.4f}")

        # Youden's J statistic
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        youden_threshold = thresholds_roc[best_j_idx]

        print(f"\n📐 Youden's optimal threshold: {youden_threshold:.3f}")
        print(f"   (Maximizes TPR - FPR)")

    # Plot score distribution
    print("\n" + "=" * 70)
    print("📈 Generating visualization...")
    print("=" * 70)

    plt.figure(figsize=(14, 10))

    # Subplot 1: Score distribution histogram
    plt.subplot(2, 2, 1)
    plt.hist(y_proba, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(y_proba.mean(), color='red', linestyle='--', label=f'Mean: {y_proba.mean():.3f}')
    plt.axvline(0.8, color='orange', linestyle='--', label='Old Threshold: 0.8')
    plt.axvline(balanced, color='green', linestyle='--', label=f'Recommended: {balanced:.3f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Threshold vs Anomaly Count
    plt.subplot(2, 2, 2)
    counts = [r['count'] for r in results]
    thrs = [r['threshold'] for r in results]
    plt.plot(thrs, counts, marker='o', linewidth=2, markersize=8)
    plt.axvline(balanced, color='green', linestyle='--', alpha=0.5, label=f'Recommended: {balanced:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Anomalies Detected')
    plt.title('Anomalies Detected vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Threshold vs Percentage
    plt.subplot(2, 2, 3)
    percentages = [r['percentage'] for r in results]
    plt.plot(thrs, percentages, marker='s', linewidth=2, markersize=8, color='orange')
    plt.axhline(10, color='green', linestyle='--', alpha=0.5, label='Target: 10%')
    plt.axvline(balanced, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('Percentage of Anomalies (%)')
    plt.title('Anomaly Percentage vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 4: Score timeline
    plt.subplot(2, 2, 4)
    plt.plot(y_proba, alpha=0.7, linewidth=1)
    plt.axhline(balanced, color='green', linestyle='--', label=f'Recommended: {balanced:.3f}')
    plt.axhline(0.8, color='orange', linestyle='--', label='Old Threshold: 0.8')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = f"threshold_analysis_{equipment_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Get parameters from command line or use defaults
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_data.csv"
    equipment_id = sys.argv[2] if len(sys.argv) > 2 else "s_3"
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    print(f"\n🚀 Starting threshold analysis...")
    print(f"   CSV: {csv_path}")
    print(f"   Equipment: {equipment_id}")
    print(f"   Max rows: {max_rows}")

    try:
        analyze_scores(csv_path, equipment_id, max_rows)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <csv_path> <equipment_id> <max_rows>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} data/raw_data.csv s_3 5000")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()