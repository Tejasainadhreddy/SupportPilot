"""The graph's state. Everything the agent knows lives here, so a checkpoint is a full snapshot."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def add_usage(a: dict | None, b: dict | None) -> dict:
    a, b = a or {}, b or {}
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


class SupportState(TypedDict, total=False):
    # conversation (kept across turns)
    customer_id: str
    messages: Annotated[list[dict], operator.add]     # {"role": "customer"|"assistant", "content": str}
    awaiting: str | None                               # e.g. "refund_details" while we wait for missing fields
    order_id: str | None
    reason: str | None
    amount: float | None

    # this turn
    message: str
    intent: str
    order: dict | None
    policy: dict | None
    approval: dict | None
    refund: dict | None
    ticket: dict | None
    knowledge: dict | None
    reply_kind: str
    facts: dict
    reply: str
    status: str                                        # "complete" | "pending_approval"

    # audit trail (append-only)
    actions: Annotated[list[dict], operator.add]       # every tool call and its result
    trace: Annotated[list[dict], operator.add]         # every node with its latency
    guardrail_events: Annotated[list[str], operator.add]
    usage: Annotated[dict[str, Any], add_usage]        # tokens and model calls


TURN_RESET: dict[str, Any] = {
    "intent": None, "order": None, "policy": None, "approval": None, "refund": None, "ticket": None,
    "knowledge": None, "reply_kind": None, "facts": {}, "reply": None, "status": None,
}
