# LSTM-Based Agentic AI System for Industrial Anomaly Detection

## Table of Contents

- Executive Summary
- Introduction  
  - Industrial Maintenance Challenges  
  - Project Objectives  
  - Solution Overview  
  - Report Organization
- System Architecture  
  - Overall Agent-Based Architecture  
  - Agentic Framework  
  - Inference Pipeline Architecture  
  - Deployment Strategy
- Machine Learning Core  
  - Data and Feature Engineering  
  - LSTM Anomaly Model  
  - Anomaly Scoring and Threshold Management
- User Interface & Visualization  
  - Streamlit Dashboard Overview  
  - Machine Health Dashboard  
  - Anomaly Analysis Views  
  - Interactive Features & Export
- Implementation  
  - Technical Stack  
  - Data Pipeline and File Structure  
  - Model & Agent Deployment  
  - Performance and Optimization
- Results & Evaluation  
  - Experimental Setup  
  - LSTM Performance  
  - Threshold Analysis  
  - System-Level Performance  
  - Comparison with Literature  
  - Model Robustness  
  - Model Interpretation  
  - Production Deployment Validation
- Conclusion & Future Work  
  - Project Success Summary  
  - Key Contributions  
  - Future Development Roadmap  
  - Final Recommendations
- References  
- Appendices [file:2]

---

## Executive Summary

This project presents an LSTM-based industrial anomaly detection system that transforms packaging-line maintenance from reactive to proactive using deep learning and agentic AI. [file:2] The system monitors packaging equipment with interval-level production data and detects abnormal behavior through a Long Short-Term Memory (LSTM) neural network. [file:2]

Results are provided via an agentic explanation layer and a Streamlit dashboard, offering both anomaly detection and interpretable maintenance guidance. [file:2] Key achievements include 13 temporal features optimized for sequence modeling, 91.88% accuracy (macro F1 = 0.9104) at threshold 0.8, probability-based severity levels, an agentic framework wrapping the LSTM and evaluation tools, and an operator-focused dashboard for analysis and export. [file:2]

---

## 1. Introduction

### 1.1 Industrial Maintenance Challenges

Modern packaging lines run at high speed and volume, where unplanned downtime, scrap, and quality issues quickly create significant losses. [file:2] Challenges include unplanned downtime costs, data overload from continuous interval data, imbalanced operator attention toward obvious breakdowns, lack of actionable interpretation, and temporal complexity of evolving anomalies. [file:2]

These issues motivate an intelligent LSTM-based system that captures temporal patterns, continuously analyzes operational data, highlights deviations from normal behavior, and supports maintenance decisions. [file:2]

### 1.2 Project Objectives

The project objectives are: [file:2]

1. Temporal anomaly detection with an LSTM model learning sequential packaging-line patterns. [file:2]  
2. Explainable outputs via metrics such as confusion matrix, precision, recall, F1, and visualizations like timelines, distributions, and heatmaps. [file:2]  
3. Agentic reasoning combining LSTM predictions with LLM-based explanations of severity, causes, and recommended actions. [file:2]  
4. Operator-focused Streamlit dashboard exposing model outputs to production and maintenance staff. [file:2]  
5. Scalable design of tools and agents for future models and data sources. [file:2]

### 1.3 Solution Overview

The solution integrates three layers: [file:2]

1. **Machine Learning Core:** LSTM network trained on 13 engineered temporal features to produce anomaly probabilities and binary labels using a threshold of 0.8. [file:2]  
2. **Agentic AI Framework:** LSTM tool and evaluation tool, plus a supervisor agent that consumes outputs and metrics to generate structured maintenance guidance. [file:2]  
3. **Streamlit Dashboard:** CSV upload, machine selection, LSTM-based analysis, metric visualization, supervisor tab, and export of annotated results. [file:2]

### 1.4 Report Organization

Sections 2–8 cover architecture, ML core, visualization, implementation, evaluation, conclusions, references, and appendices, respectively, following a standard project report structure. [file:2]

---

## 2. System Architecture

### 2.1 Overall Agent-Based Architecture

The architecture transforms raw packaging-line CSV data into maintenance intelligence through four conceptual layers. [file:2]

1. **Data Layer:** `raw_data.csv` with interval-level data per `equipment_ID`. [file:2]  
2. **Feature and Model Layer:** Feature engineering to ML-ready sequences; LSTM generates anomaly scores. [file:2]  
3. **Tool and Agent Layer:** Tools expose LSTM and evaluation logic, while a supervisor agent reasons over outputs. [file:2]  
4. **Presentation Layer:** Streamlit app showing dashboards, charts, and explanations. [file:2]

Data flow: `CSV → Feature Engineering → LSTM Tool → Evaluation Tool → Supervisor Agent → Streamlit UI`. [file:2]

### 2.2 Agentic Framework

The agentic framework uses two tools and two LLM-based agents implemented in `agents.py` and backed by Groq-hosted Llama 4. [file:2]

- **LSTM Anomaly Tool (`lstm_anomaly_tool`):** Parses JSON records, engineers 13 temporal features, applies `StandardScaler`, reshapes to \([N, 1, 13]\), runs the pretrained LSTM, applies threshold 0.8, and outputs anomaly_score, prediction, and label. [file:2]  
- **Evaluation Metrics Tool (`evaluation_metrics_tool`):** Consumes LSTM output JSON and computes confusion matrix, accuracy, precision, recall, F1, ROC-AUC, and a classification report. [file:2]

**Agents:**

- **LSTM Anomaly Agent:** Runnable sequence that instructs the LLM to call `lstm_anomaly_tool` with a `data` argument and return anomaly results. [file:2]  
- **Supervisor Explainer Agent:** Runnable sequence where the LLM reads LSTM results, metrics, and machine context and produces severity assessment, root cause hypotheses, and maintenance recommendations. [file:2]

All agents share `meta-llama/llama-4-scout-17b-16e-instruct` via ChatGroq with temperature 0.1 and max tokens 1000. [file:2]

### 2.3 Inference Pipeline Architecture

At runtime the pipeline performs: [file:2]

1. CSV upload and `equipment_ID` selection.  
2. Filtering and sorting by timestamp, with row limit.  
3. JSON serialization of records.  
4. LSTM tool execution (feature engineering, scaling, inference, thresholding).  
5. Evaluation tool computation of metrics.  
6. Health metrics calculation (anomaly rate, health score, severity distribution, risk level).  
7. Supervisor reasoning over outputs and context.  
8. Streamlit visualization of timelines, histograms, heatmaps, supervisor responses, and export options. [file:2]

Typical end-to-end latency is 2–5 seconds for several thousand records. [file:2]

### 2.4 Deployment Strategy

The system targets local or lightweight cloud deployment with a simple project structure. [file:2]

- Local Python environment with virtualenv, no external database required. [file:2]  
- Streamlit UI launched via `streamlit run streamlit_app.py` or deployed to Streamlit Cloud. [file:2]  
- Models and scaler stored as `lstm_anomaly_model.h5`, `lstm_scaler.pkl`, and `meta_lstm.pkl` in `models/`. [file:2]  
- Core files: `tools_models.py`, `agents.py`, `streamlit_app.py`, `inference.py`, `.env` (for `GROQ_API_KEY`). [file:2]

---

## 3. Machine Learning Core

### 3.1 Data and Feature Engineering

The ML core converts raw CSV rows into LSTM-friendly feature sequences. [file:2]

**Labels:**

- Normal class (0): `type == "production"`.  
- Anomaly class (1): `type == "downtime"` or `type == "performance_loss"`. [file:2]

**13 LSTM features:**

- Raw: `elapsed`, `pi`, `po`, `speed`. [file:2]  
- Derived: `efficiency = po / max(pi, 1)`, `scrap = pi - po`, `scrap_rate = scrap / max(pi, 1)`, `tp_sec = po / max(elapsed, 1)`. [file:2]  
- Temporal: `hour`, `dayofweek`, `is_weekend`. [file:2]  
- Categorical encoded: `equipment_ID`, `alarm`. [file:2]

These features are chosen to capture temporal behavior, balance information and noise, and remain interpretable. [file:2]

### 3.2 LSTM Anomaly Model

The LSTM captures sequential dependencies in interval data. [file:2]

**Architecture:**

- Input: shape `(1, 13)`.  
- LSTM layer: 32 units, `return_sequences=False`, dropout 0.4.  
- Dense: 16 units with ReLU and dropout 0.4.  
- Output: 1 unit with sigmoid activation for probability \(s \in [0,1]\). [file:2]

**Training configuration:**

- Loss: binary cross-entropy.  
- Optimizer: Adam (learning rate 0.001).  
- Metric: accuracy, with early stopping on validation loss. [file:2]

**Preprocessing:** `StandardScaler` normalization then reshape to \((n\_samples, 1, 13)\) to stabilize gradients and feature importance. [file:2]

**Outputs:**

- Anomaly score \(s\) where higher values indicate higher anomaly probability.  
- Binary prediction: `s < 0.8 → 0`, `s ≥ 0.8 → 1`. [file:2]

### 3.3 Anomaly Scoring and Threshold Management

Threshold 0.8 was selected via validation. [file:2]

- At 0.8, precision is about 0.94, recall about 0.90, and F1 about 0.91 in the earlier summary, while the later detailed evaluation reports even higher performance (near 99.9%). [file:2]  
- Lower thresholds (0.5, 0.6, 0.7) increase recall but raise false alarms; 0.9 removes false alarms but misses more anomalies. [file:2]

Severity is mapped from score relative to threshold (Normal, Low, Medium, High, Critical). [file:2] Health score is computed as `100 - anomaly_percentage` and mapped into Excellent, Good, Fair, Poor, or Critical categories. [file:2]

---

## 4. User Interface & Visualization

### 4.1 Streamlit Dashboard Overview

The Streamlit dashboard targets engineers and maintenance staff with four main tabs. [file:2]

- Header with project title, usage instructions, and system status. [file:2]  
- Sidebar: CSV uploader, machine selector, row limit slider, “Run LSTM Analysis” button, and configuration. [file:2]  
- Tabs: Machine Health Dashboard, Analytics, AI Supervisor, Export, plus real-time feedback via spinners and messages. [file:2]

### 4.2 Machine Health Dashboard

This tab gives a high-level machine health summary. [file:2]

**Metric cards:**

1. Health score (0–100) with color-coded status.  
2. Anomaly rate (% of flagged intervals).  
3. Total anomalies with total intervals and threshold display.  
4. Machine status (Excellent, Good, Fair, Poor, Critical). [file:2]

**Timeline view:**

- Plotly chart with anomaly scores over time, threshold line at 0.8, and markers for detected anomalies. [file:2]  
- Interactive panning, zooming, tooltips, and export to PNG. [file:2]

**Severity distribution:** Bar chart showing counts of Low, Medium, High, and Critical anomalies to support prioritization. [file:2]

### 4.3 Anomaly Analysis Views

The Analytics tab provides deeper statistical and correlation analysis. [file:2]

- Score histogram with threshold line to inspect distribution and separation between normal and anomalies. [file:2]  
- 13×13 correlation heatmap of features to identify redundant features and relationships. [file:2]  
- Rule-based recommendations panel that maps anomaly rate ranges to immediate, urgent, preventive, or normal-operation maintenance suggestions. [file:2]

### 4.4 Interactive Features & Export

The AI Supervisor and Export tabs provide interaction and integration. [file:2]

- **AI Supervisor:** Context display (machine, summary, anomalies), question input, and structured LLM responses with severity, causes, and recommended actions. [file:2]  
- **Export:** Downloads in CSV (scores, predictions, labels), JSON (structured data), and text reports (executive summary, metrics, recommendations) with timestamped filenames. [file:2]

---

## 5. Implementation

### 5.1 Technical Stack

The system uses a Python-based stack. [file:2]

- Python 3.8+, `pandas`, `numpy`, TensorFlow/Keras, `scikit-learn`, Streamlit, Plotly, LangChain, Groq client, `python-dotenv`, and `joblib`. [file:2]  
- Dependencies specified (e.g., `tensorflow>=2.10.0`, `streamlit>=1.28.0`, `langchain>=0.1.0`, `groq>=0.4.0`). [file:2]

### 5.2 Data Pipeline and File Structure

Data processing flow: load CSV, filter by equipment ID, engineer features, serialize to JSON, invoke LSTM tool, and cache results in Streamlit session state. [file:2]

Caching is implemented with `@st.cache_data` for data and `@st.cache_resource` for models to avoid redundant computation. [file:2] The project structure separates `data/`, `models/`, core Python modules, and configuration files. [file:2]

### 5.3 Model & Agent Deployment

Models are loaded from `models/lstm_anomaly_model.h5` and `models/lstm_scaler.pkl`, with additional metadata in `meta_lstm.pkl`. [file:2]

The LLM is instantiated via ChatGroq using the Llama 4 instruct model, and helper functions create the LSTM anomaly agent and supervisor explainer agent for use inside the app. [file:2]

### 5.4 Performance and Optimization

Optimizations include caching, batch processing, vectorized operations, optional GPU acceleration, and UI-level strategies like progress indicators and row limits. [file:2]

Measured times are about 0.5s for a 10k-row CSV load, 1.0s for feature engineering, 2.0s for LSTM inference on 1k samples, 0.5s for visualization, and roughly 4s end-to-end per analysis. [file:2]

---

## 6. Results & Evaluation

### 6.1 Experimental Setup

The dataset comprises ~238,879 intervals for training and 45,885 for evaluation, drawn from multiple packaging machines over several months. [file:2]

Labels distinguish production vs downtime/performance loss, and splits are stratified, temporally ordered, and free of leakage. [file:2] Metrics include confusion matrix, accuracy, precision, recall, F1, and ROC-AUC. [file:2]

### 6.2 LSTM Performance

With threshold 0.8, the detailed evaluation reports: [file:2]

- Confusion matrix with very few errors, totaling 45,885 samples.  
- Overall accuracy 99.90%, anomaly precision 99.99%, anomaly recall 99.75%, and anomaly F1 99.87%.  
- Macro and weighted F1 scores both around 99.9%.  

These results point to near-perfect anomaly detection with almost no false alarms. [file:2]

### 6.3 Threshold Analysis

A threshold sensitivity study from 0.5 to 0.9 shows: [file:2]

- Lower thresholds increase recall but create more false positives.  
- Threshold 0.8 yields the best F1, minimal false alarms, and high recall.  
- Threshold 0.9 eliminates false alarms but reduces recall enough to be too conservative.  

Consequently, 0.8 is recommended as the operational default, with the option of dynamic, machine-specific adjustment in future work. [file:2]

### 6.4–6.9 System-Level and Robustness Highlights

The report includes: [file:2]

- System-level performance metrics (latency, scalability, dashboard usability).  
- Cross-validation results with very low variance across folds.  
- Temporal stability across recent vs historical data and across shifts.  
- Interpretation insights about feature importance and score distributions.  
- Production deployment validation via pilot A/B testing that showed improved detection rate, drastically reduced false alarms, and higher operator satisfaction.  

---

## 7. Conclusion & Future Work

### 7.1 Project Success Summary

The project delivers an LSTM-based anomaly detection system that achieves high accuracy, low latency, interpretable outputs, and an operator-friendly dashboard powered by an agentic AI framework. [file:2]

### 7.2 Key Contributions

Key contributions include: [file:2]

- LSTM-based temporal anomaly detection tailored to packaging-line data.  
- Optimized 13-feature engineering for real-time sequence modeling.  
- Agentic integration of LSTM with LLM-based supervisor reasoning.  
- Industrially tuned thresholding and severity logic.  
- Comprehensive evaluation with threshold and system-level analysis.  

### 7.3 Future Development Roadmap

Planned enhancements span dynamic thresholding, additional temporal models (GRU/Transformer), real-time streaming integration, feedback loops, multi-machine analysis, and predictive maintenance capabilities. [file:2]

### 7.4 Final Recommendations

For deployment, the report recommends using the LSTM detector with threshold 0.8 as the primary anomaly detector, integrating it into maintenance workflows, monitoring impact on downtime and costs, and iteratively improving models and supervisor logic using operator feedback. [file:2]

---

## References

The report cites foundational and applied works on LSTM, industrial anomaly detection, deep learning for time series, agentic AI frameworks, predictive maintenance, threshold optimization, time series classification, and cyber-physical manufacturing systems. [file:2]
