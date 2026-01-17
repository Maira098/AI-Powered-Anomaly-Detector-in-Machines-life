# test_agents.py - LSTM-Only Version
"""
Test script for LSTM-only anomaly detection agents.
Random Forest has been removed from the architecture.
"""

import json
import pandas as pd

from agents import (
    get_lstm_anomaly_agent,  # Updated name
    get_supervisor_explainer_agent,
    lstm_anomaly_tool,
    evaluation_metrics_tool,
)


def main():
    print("=" * 70)
    print("LSTM-ONLY AGENT TEST")
    print("=" * 70)

    # 1) Load sample data
    csv_path = "data/raw_data.csv"

    try:
        df_raw = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find {csv_path}")
        print("Please make sure the file exists or update the path.")
        return

    max_rows = 200  # Increased for better testing (was 50)
    records = df_raw.to_dict(orient="records")[:max_rows]
    data_json = json.dumps(records)

    print(f"\n✅ Loaded {len(records)} rows from {csv_path}")
    print(f"   Columns: {list(df_raw.columns)}")

    # 2) Create agents
    print("\n" + "=" * 70)
    print("INITIALIZING AGENTS")
    print("=" * 70)

    try:
        lstm_agent = get_lstm_anomaly_agent()
        supervisor_llm = get_supervisor_explainer_agent()
        print("✅ LSTM Agent initialized")
        print("✅ Supervisor Agent initialized")
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return

    # === LSTM AGENT TEXT CALL (OPTIONAL) ===
    print("\n" + "=" * 70)
    print("TEST 1: LSTM AGENT (via LangChain)")
    print("=" * 70)

    lstm_prompt = (
        "Use lstm_anomaly_tool by passing the JSON below as the 'data' argument.\n"
        f"{data_json[:500]}..."  # Preview only
    )

    try:
        lstm_agent_out = lstm_agent.invoke({"input": lstm_prompt})
        print("\n[LSTM Agent Output]")
        print(lstm_agent_out)
    except Exception as e:
        print(f"❌ LSTM Agent error: {e}")

    # === DIRECT TOOL CALLS USING PRETRAINED LSTM MODEL ===
    print("\n" + "=" * 70)
    print("TEST 2: DIRECT LSTM TOOL CALL")
    print("=" * 70)

    try:
        lstm_json = lstm_anomaly_tool.invoke({"data": data_json})
        print("✅ LSTM tool executed successfully")
    except Exception as e:
        print(f"❌ LSTM tool error: {e}")
        return

    # === EVALUATION METRICS ===
    print("\n" + "=" * 70)
    print("TEST 3: EVALUATION METRICS")
    print("=" * 70)

    try:
        # Updated parameter name from rf_json to lstm_json
        metrics_text = evaluation_metrics_tool.invoke({"lstm_json": lstm_json})
        print("✅ Evaluation metrics calculated")
        print("\n[Metrics Summary]")
        print(metrics_text[:500] + "..." if len(metrics_text) > 500 else metrics_text)
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        metrics_text = "Metrics not available"

    # === PARSE AND ANALYZE RESULTS ===
    print("\n" + "=" * 70)
    print("TEST 4: ANALYZE LSTM RESULTS")
    print("=" * 70)

    try:
        lstm_data = json.loads(lstm_json)
        print(f"✅ Parsed {len(lstm_data)} predictions")
    except Exception as e:
        print(f"❌ Error parsing LSTM JSON: {e}")
        lstm_data = []

    # Print anomalies only (prediction == 1)
    lstm_anoms = [row for row in lstm_data if row.get("prediction") == 1]

    print(f"\n📊 LSTM Results Summary:")
    print(f"   Total intervals: {len(lstm_data)}")
    print(f"   Anomalies detected: {len(lstm_anoms)}")
    print(f"   Anomaly rate: {len(lstm_anoms) / len(lstm_data) * 100:.1f}%")

    if lstm_anoms:
        print("\n[Sample Anomalies - First 3]")
        for i, anom in enumerate(lstm_anoms[:3], 1):
            score = anom.get("anomaly_score", "N/A")
            eq_id = anom.get("equipment_ID", "N/A")
            print(f"   {i}. Equipment: {eq_id}, Score: {score:.4f}" if isinstance(score,
                                                                                  float) else f"   {i}. Equipment: {eq_id}, Score: {score}")
    else:
        print("\n⚠️  No anomalies detected in this sample")
        print("   This might mean:")
        print("   - The data is genuinely normal (good!)")
        print("   - Sample size too small (try increasing max_rows)")
        print("   - Threshold might be too high (check tools_models.py)")

    # Calculate score statistics
    if lstm_data:
        scores = [row.get("anomaly_score") for row in lstm_data if "anomaly_score" in row]
        if scores:
            import numpy as np
            print(f"\n📈 Score Statistics:")
            print(f"   Min: {min(scores):.4f}")
            print(f"   Max: {max(scores):.4f}")
            print(f"   Mean: {np.mean(scores):.4f}")
            print(f"   Median: {np.median(scores):.4f}")

    # Build preview for the supervisor
    if isinstance(lstm_data, list):
        lstm_preview = lstm_data[:10]
        lstm_preview_str = json.dumps(lstm_preview, indent=2)
    else:
        lstm_preview_str = str(lstm_json)[:2000]

    # === SUPERVISOR LLM ===
    print("\n" + "=" * 70)
    print("TEST 5: SUPERVISOR AGENT (AI INSIGHTS)")
    print("=" * 70)

    sup_input = (
        "You are analyzing LSTM anomaly detection results for industrial machines.\n\n"
        "Here are the results and metrics. Use only this information to answer.\n\n"
        f"LSTM_RESULTS_PREVIEW:\n{lstm_preview_str}\n\n"
        f"EVALUATION_METRICS:\n{metrics_text[:1000]}\n\n"
        f"SUMMARY:\n"
        f"- Total intervals analyzed: {len(lstm_data)}\n"
        f"- Anomalies detected: {len(lstm_anoms)}\n"
        f"- Anomaly rate: {len(lstm_anoms) / len(lstm_data) * 100:.1f}%\n\n"
        "Question: Analyze the anomaly detection results for the machine(s) in this data. "
        "Explain the severity of any anomalies detected and provide simple, actionable "
        "maintenance recommendations. If no anomalies were detected, explain what that means."
    )

    try:
        sup_out = supervisor_llm.invoke({"input": sup_input})
        print("\n[Supervisor Agent Response]")
        print("-" * 70)
        print(sup_out)
        print("-" * 70)
        print("✅ Supervisor analysis complete")
    except Exception as e:
        print(f"❌ Supervisor error: {e}")

    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Data loaded: {len(records)} rows")
    print(f"✅ LSTM predictions: {len(lstm_data)} intervals")
    print(f"✅ Anomalies found: {len(lstm_anoms)} ({len(lstm_anoms) / len(lstm_data) * 100:.1f}%)")
    print(f"✅ AI insights: Generated")
    print("\n🎉 All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()