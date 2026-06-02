from prometheus_client import Counter
from prometheus_client import Histogram


# =====================================
# LLM Metrics
# =====================================

llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM requests"
)

tokens_used_total = Counter(
    "tokens_used_total",
    "Total tokens consumed"
)

# =====================================
# Latency Metrics
# =====================================

llm_latency_seconds = Histogram(
    "llm_latency_seconds",
    "LLM request latency"
)

# =====================================
# RAG Metrics
# =====================================

retrieval_requests_total = Counter(
    "retrieval_requests_total",
    "Total retrieval requests"
)

retrieval_failures_total = Counter(
    "retrieval_failures_total",
    "Failed retrieval requests"
)

# =====================================
# Agent Metrics
# =====================================

agent_executions_total = Counter(
    "agent_executions_total",
    "Agent executions",
    ["agent_name"]
)

agent_routing_errors = Counter(
    "agent_routing_errors",
    "Agent routing failures"
)