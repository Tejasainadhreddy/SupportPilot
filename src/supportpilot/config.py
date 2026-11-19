"""Settings from environment variables (same code locally, in tests and on AWS)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-west-2"))

    # DynamoDB: one table for LangGraph checkpoints, one single-table design for business data
    checkpoint_table: str = field(default_factory=lambda: _env("CHECKPOINT_TABLE", "supportpilot-checkpoints"))
    data_table: str = field(default_factory=lambda: _env("DATA_TABLE", "supportpilot-data"))
    dynamodb_endpoint: str = field(default_factory=lambda: _env("DYNAMODB_ENDPOINT"))  # DynamoDB Local
    checkpoint_ttl_days: int = field(default_factory=lambda: int(_env("CHECKPOINT_TTL_DAYS", "14")))

    # Model: "bedrock" (Claude via Converse) or "rules" (offline, deterministic, for tests and local dev)
    llm: str = field(default_factory=lambda: _env("LLM", "rules"))
    llm_model_id: str = field(
        default_factory=lambda: _env("LLM_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"))
    price_input_per_1k: float = field(default_factory=lambda: float(_env("PRICE_LLM_INPUT_PER_1K", "0.001")))
    price_output_per_1k: float = field(default_factory=lambda: float(_env("PRICE_LLM_OUTPUT_PER_1K", "0.005")))

    # Knowledge: "mcp" (KnowledgeMCP over the MCP protocol) or "static" (offline canned policy answers)
    knowledge: str = field(default_factory=lambda: _env("KNOWLEDGE", "static"))
    kmcp_command: str = field(default_factory=lambda: _env("KMCP_COMMAND"))  # default: python -m knowledgemcp...
    kmcp_api_url: str = field(default_factory=lambda: _env("KMCP_API_URL"))
    kmcp_api_key: str = field(default_factory=lambda: _env("KMCP_API_KEY"))
    kmcp_api_key_secret_arn: str = field(default_factory=lambda: _env("KMCP_API_KEY_SECRET_ARN"))

    # Business policy (must match knowledge_docs/support/refund-policy.md)
    auto_refund_limit_usd: float = field(default_factory=lambda: float(_env("AUTO_REFUND_LIMIT_USD", "100")))
    refund_window_days: int = field(default_factory=lambda: int(_env("REFUND_WINDOW_DAYS", "30")))
    max_tool_attempts: int = field(default_factory=lambda: int(_env("MAX_TOOL_ATTEMPTS", "3")))

    # API auth: one key for the chat client, one for human approvers (env locally, Secrets Manager on AWS)
    client_api_key: str = field(default_factory=lambda: _env("CLIENT_API_KEY"))
    approver_api_key: str = field(default_factory=lambda: _env("APPROVER_API_KEY"))
    client_key_secret_arn: str = field(default_factory=lambda: _env("CLIENT_KEY_SECRET_ARN"))
    approver_key_secret_arn: str = field(default_factory=lambda: _env("APPROVER_KEY_SECRET_ARN"))


def get_settings() -> Settings:
    return Settings()
