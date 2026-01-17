# agents.py
import os
import json
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from tools_models import lstm_tool, evaluation_tool

# ---------------- ENV & LLM ----------------
load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ---------------- TOOLS ----------------
@tool
def lstm_anomaly_tool(data: str) -> str:
    """Use pretrained LSTM model to detect anomalies. `data` is a JSON array string."""
    try:
        records = json.loads(data)
    except json.JSONDecodeError:
        return '{"error": "Invalid JSON for lstm_anomaly_tool.data"}'
    return lstm_tool(records)


@tool
def evaluation_metrics_tool(lstm_json: str) -> str:
    """Compute confusion matrix, F1 score, ROC-AUC and plots from LSTM JSON."""
    result = evaluation_tool(lstm_json)
    return json.dumps(result)


# ---------------- LSTM AGENT ----------------
def get_lstm_anomaly_agent():
    """
    LSTM Anomaly Detection Agent.
    Receives JSON machine interval data and uses LSTM model to detect anomalies.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the LSTM Anomaly Detection Agent. "
            "You receive JSON machine interval data as text and should call lstm_anomaly_tool "
            "with one argument named 'data' containing that JSON string."
        ),
        ("human", "{input}"),
    ])

    return RunnableSequence(
        prompt,
        llm.bind_tools([lstm_anomaly_tool]),
        StrOutputParser(),
    )


# Alias for backward compatibility
get_lstm_sequence_agent = get_lstm_anomaly_agent


# ---------------- SUPERVISOR (LLM ONLY, NO TOOLS) ----------------
def get_supervisor_explainer_agent():
    """
    Supervisor LLM that takes LSTM results + metrics as text and explains them.
    Tools are called directly in Python, not via tool-calling.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a supervisor maintenance assistant. "
            "You receive:\n"
            "- LSTM anomaly results (JSON snippet)\n"
            "- Evaluation metrics (text)\n"
            "Analyze the anomaly patterns, identify potential root causes, and provide "
            "clear, actionable maintenance recommendations for the operator."
        ),
        ("human", "{input}"),
    ])

    return RunnableSequence(
        prompt,
        llm,
        StrOutputParser(),
    )