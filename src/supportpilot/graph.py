"""The SupportPilot LangGraph.

    START -> guard_input -> classify -+-> knowledge ---------------------------------+
                  |                   +-> order_status ------------------------------+
                  | (blocked)         +-> create_ticket -----------------------------+
                  |                   +-> refund_prepare -+-> execute_refund --------+-> respond -> END
                  |                   |                   +-> notify_pending -> human_approval (interrupt)
                  |                   |                   |        approve -> execute_refund
                  |                   |                   |        reject  -> respond
                  |                   |                   +-> respond (missing fields / denied)
                  +-------------------+-> respond (greeting / other)

- Tool nodes have a RetryPolicy for transient errors (throttling, timeouts).
- human_approval calls interrupt(): the run stops, the state is checkpointed, and it continues from that
  exact point when an approver answers (even in a different Lambda invocation days later).
- With a checkpointer, a crash mid-run can be resumed with graph.invoke(None, config).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from .config import Settings
from .guardrails import GuardrailViolation, check_message, check_reply, refund_policy
from .knowledge import KnowledgeClient, KnowledgeUnavailable
from .llm import SupportLLM, Usage, rule_reason
from .observability import MetricUnit, add_metric
from .state import TURN_RESET, SupportState
from .tools import AlreadyRefunded, SupportTools, TransientToolError

ORDER_ID = re.compile(r"\bORD-?\s?(\d{4})\b", re.IGNORECASE)
AMOUNT = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)\s?(?:dollars|usd)\b", re.IGNORECASE)


@dataclass
class Deps:
    llm: SupportLLM
    tools: SupportTools
    knowledge: KnowledgeClient
    settings: Settings = field(default_factory=Settings)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)


def _usage(u: Usage) -> dict:
    return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "llm_calls": u.llm_calls}


def traced(name: str, fn):
    """Record each node's latency in the state (returned by the API) and as a CloudWatch metric."""

    def wrapper(state: SupportState, config):
        start = time.perf_counter()
        out = dict(fn(state, config) or {})
        ms = round((time.perf_counter() - start) * 1000, 2)
        out["trace"] = [{"node": name, "ms": ms}]
        add_metric(f"Node_{name}_Ms", ms, MetricUnit.Milliseconds)
        return out

    return wrapper


def build_graph(deps: Deps, checkpointer=None):
    s = deps.settings
    tool_retry = RetryPolicy(max_attempts=s.max_tool_attempts, initial_interval=0.2, backoff_factor=2.0,
                             retry_on=TransientToolError)

    def reply(kind: str, **facts) -> dict:
        return {"reply_kind": kind, "facts": facts}

    def action(tool: str, ok: bool, **detail) -> dict:
        return {"tool": tool, "ok": ok, **detail}

    # ---- nodes ------------------------------------------------------------------------------------
    def guard_input(state, config):
        raw = state["message"]
        try:
            clean, events = check_message(raw)
        except GuardrailViolation as v:
            add_metric("GuardrailBlocks", 1)
            return {**TURN_RESET, "message": "[blocked message]", "guardrail_events": [v.code],
                    "messages": [{"role": "customer", "content": "[blocked message]"}], **reply("blocked")}
        return {**TURN_RESET, "message": clean, "guardrail_events": events,
                "messages": [{"role": "customer", "content": clean}]}

    def classify(state, config):
        route, usage = deps.llm.route(state["message"], state.get("awaiting"))
        update = {"intent": route.intent, "usage": _usage(usage)}
        if route.intent == "refund":
            # Required fields are extracted by code (regex), never guessed by the model.
            if m := ORDER_ID.search(state["message"]):
                update["order_id"] = f"ORD-{m.group(1)}"
            if m := AMOUNT.search(state["message"]):
                update["amount"] = float(m.group(1) or m.group(2))
            reason = route.reason or rule_reason(state["message"])
            if reason:
                update["reason"] = reason
        elif route.intent == "order_status":
            m = ORDER_ID.search(state["message"])
            update["order_id"] = f"ORD-{m.group(1)}" if m else None
        else:  # customer moved on: forget any half-finished refund
            update.update(awaiting=None, order_id=None, reason=None, amount=None)
        return update

    def knowledge(state, config):
        try:
            result = deps.knowledge.ask(state["message"])
        except KnowledgeUnavailable as exc:  # waking up / too slow: answer now instead of burning the deadline
            add_metric("KnowledgeUnavailable", 1)
            return {"actions": [action("knowledge.ask_question", False, error=str(exc)[:120])],
                    **reply("knowledge_unavailable")}
        except (TimeoutError, ConnectionError, OSError) as exc:
            raise TransientToolError(str(exc)) from exc
        ok = not result.get("refused")
        update = {"knowledge": result,
                  "actions": [action("knowledge.ask_question", ok, citations=len(result.get("citations", [])))]}
        if not ok:
            return {**update, **reply("knowledge_not_found")}
        return {**update, **reply("knowledge_answer", answer=result["answer"], citations=result["citations"])}

    def order_status(state, config):
        oid = state.get("order_id")
        if not oid:
            return reply("order_not_found", order_id=None)
        order = deps.tools.get_order(state["customer_id"], oid)
        act = [action("get_order", order is not None, order_id=oid)]
        if not order:
            return {"actions": act, **reply("order_not_found", order_id=oid)}
        return {"order": order, "actions": act,
                **reply("order_status", order_id=oid, items=order["items"], status=order["status"],
                        delivered_at=order.get("delivered_at"), eta=order.get("eta"))}

    def refund_prepare(state, config):
        missing = [f for f in ("order_id", "reason") if not state.get(f)]
        if missing:
            return {"awaiting": "refund_details", **reply("ask_missing_fields", missing=missing)}
        oid = state["order_id"]
        order = deps.tools.get_order(state["customer_id"], oid)
        acts = [action("get_order", order is not None, order_id=oid)]
        if not order:
            return {"awaiting": "refund_details", "order_id": None, "actions": acts,
                    **reply("order_not_found", order_id=oid)}
        amount = float(state.get("amount") or order["total"])
        already = deps.tools.get_refund(oid) is not None
        decision = refund_policy(order, amount, already, deps.now(), s.auto_refund_limit_usd, s.refund_window_days)
        update = {"order": order, "amount": amount, "policy": asdict(decision), "actions": acts, "awaiting": None}
        if not decision.allowed:
            add_metric("RefundsDenied", 1)
            return {**update, "order_id": None, "reason": None, "amount": None,
                    **reply("refund_denied", order_id=oid, detail=decision.detail)}
        return update

    def route_refund(state) -> str:
        if state.get("reply_kind"):
            return "respond"
        return "notify_pending" if state["policy"]["needs_approval"] else "execute_refund"

    def notify_pending(state, config):
        # Tell the customer now: the run is about to pause until a human decides.
        facts = {"order_id": state["order_id"], "amount": state["amount"]}
        text, usage = deps.llm.write_reply("refund_pending_approval", facts, state["message"])
        return {"status": "pending_approval", "reply_kind": "refund_pending_approval", "facts": facts,
                "reply": text, "messages": [{"role": "assistant", "content": text}], "usage": _usage(usage)}

    def human_approval(state, config):
        thread_id = config["configurable"]["thread_id"]
        request = {"customer_id": state["customer_id"], "order_id": state["order_id"], "amount": state["amount"],
                   "reason": state["reason"], "policy": state["policy"]["detail"]}
        deps.tools.add_pending_approval(thread_id, request)
        add_metric("ApprovalsRequested", 1)
        decision = interrupt(request)  # <- the run pauses here until Command(resume=...) arrives
        approved = decision.get("decision") == "approve"
        deps.tools.resolve_approval(thread_id, "approved" if approved else "rejected",
                                    decision.get("approver", "unknown"), decision.get("note", ""))
        update = {"approval": {"approved": approved, "approver": decision.get("approver"),
                               "note": decision.get("note", "")},
                  "actions": [action("human_approval", approved, approver=decision.get("approver"))]}
        if not approved:
            oid = state["order_id"]
            return {**update, "status": "complete", "order_id": None, "reason": None, "amount": None,
                    **reply("refund_rejected", order_id=oid, note=decision.get("note", ""))}
        return {**update, "reply_kind": None}

    def execute_refund(state, config):
        approver = (state.get("approval") or {}).get("approver") or "auto"
        oid, amount = state["order_id"], state["amount"]
        try:
            refund = deps.tools.issue_refund(state["customer_id"], oid, amount, state["reason"], approver)
        except AlreadyRefunded:
            return {"actions": [action("issue_refund", False, order_id=oid, error="already_refunded")],
                    "order_id": None, "reason": None, "amount": None,
                    **reply("refund_denied", order_id=oid, detail="This order has already been refunded.")}
        add_metric("RefundsIssued", 1)
        add_metric("RefundUSD", amount)
        return {"refund": refund, "status": "complete", "order_id": None, "reason": None, "amount": None,
                "actions": [action("issue_refund", True, order_id=oid, amount=amount, refund_id=refund["refund_id"],
                                   approved_by=approver)],
                **reply("refund_issued", order_id=oid, amount=amount, refund_id=refund["refund_id"])}

    def create_ticket(state, config):
        ticket = deps.tools.create_ticket(state["customer_id"], subject=state["message"][:80],
                                          details=state["message"])
        return {"ticket": ticket, "actions": [action("create_ticket", True, ticket_id=ticket["ticket_id"])],
                **reply("ticket_created", ticket_id=ticket["ticket_id"])}

    def respond(state, config):
        kind = state.get("reply_kind") or ("greeting" if state.get("intent") == "greeting" else "help")
        text, usage = deps.llm.write_reply(kind, state.get("facts") or {}, state["message"])
        text, events = check_reply(text, refund_issued=bool(state.get("refund")))
        return {"reply": text, "reply_kind": kind, "status": state.get("status") or "complete",
                "messages": [{"role": "assistant", "content": text}], "usage": _usage(usage),
                "guardrail_events": events}

    # ---- wiring -----------------------------------------------------------------------------------
    g = StateGraph(SupportState)
    for name, fn, retry in [
        ("guard_input", guard_input, None), ("classify", classify, None),
        ("knowledge", knowledge, tool_retry), ("order_status", order_status, tool_retry),
        ("refund_prepare", refund_prepare, tool_retry), ("notify_pending", notify_pending, None),
        ("human_approval", human_approval, tool_retry), ("execute_refund", execute_refund, tool_retry),
        ("create_ticket", create_ticket, tool_retry), ("respond", respond, None),
    ]:
        g.add_node(name, traced(name, fn), retry_policy=retry)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", lambda st: "respond" if st.get("reply_kind") == "blocked" else "classify",
                            ["respond", "classify"])
    g.add_conditional_edges("classify", lambda st: {
        "policy_question": "knowledge", "order_status": "order_status", "refund": "refund_prepare",
        "ticket": "create_ticket"}.get(st["intent"], "respond"),
        ["knowledge", "order_status", "refund_prepare", "create_ticket", "respond"])
    g.add_conditional_edges("refund_prepare", route_refund, ["respond", "notify_pending", "execute_refund"])
    g.add_edge("notify_pending", "human_approval")
    g.add_conditional_edges("human_approval", lambda st: "respond" if st.get("reply_kind") else "execute_refund",
                            ["respond", "execute_refund"])
    for node in ("knowledge", "order_status", "execute_refund", "create_ticket"):
        g.add_edge(node, "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)
