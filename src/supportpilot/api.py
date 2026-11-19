"""FastAPI service for SupportPilot (Lambda via Mangum, or uvicorn locally / on Fargate).

Customer chat:   POST /chat                      (x-api-key: client key)
                 GET  /sessions/{id}
                 POST /sessions/{id}/resume      (continue a run that failed mid-way, from its checkpoint)
Human approval:  GET  /approvals                 (x-api-key: approver key)
                 POST /approvals/{session_id}    {"decision": "approve" | "reject", "note": "..."}
"""

from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from langgraph.types import Command
from mangum import Mangum
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .observability import MetricUnit, add_metric, logger, metrics, tracer
from .tools import TransientToolError

app = FastAPI(title="SupportPilot API", version=__version__,
              description="LangGraph customer-support agent with human approval for large refunds.")


@lru_cache(maxsize=1)
def settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def runtime():
    """(compiled graph, deps). Built once per Lambda container."""
    from .services import build_app_graph

    return build_app_graph(settings())


@lru_cache(maxsize=1)
def api_keys() -> dict[str, str]:
    s = settings()
    if s.api_keys_secret_arn:
        import boto3

        raw = boto3.client("secretsmanager", region_name=s.region).get_secret_value(
            SecretId=s.api_keys_secret_arn)["SecretString"]
        return json.loads(raw)
    return {"client": s.client_api_key, "approver": s.approver_api_key}


def _require(role: str):
    def check(x_api_key: str | None = Header(default=None), keys: dict = Depends(api_keys)) -> str:
        expected = keys.get(role) or ""
        if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        return role
    return check


class ChatRequest(BaseModel):
    customer_id: str = Field(pattern=r"^C-\d{3,6}$", examples=["C-100"])
    message: str = Field(min_length=1, max_length=2000, examples=["Where is my order ORD-1003?"])
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    approver: str = Field(default="support-lead", max_length=100)
    note: str = Field(default="", max_length=500)


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _counts(values: dict) -> dict:
    return {k: len(values.get(k) or []) for k in ("actions", "trace", "guardrail_events")} | {
        "usage": dict(values.get("usage") or {})}


def _turn_result(session_id: str, before: dict, values: dict, pending: bool) -> dict:
    s = settings()
    usage_now, usage_before = values.get("usage") or {}, before["usage"]
    tin = usage_now.get("input_tokens", 0) - usage_before.get("input_tokens", 0)
    tout = usage_now.get("output_tokens", 0) - usage_before.get("output_tokens", 0)
    cost = round(tin / 1000 * s.price_input_per_1k + tout / 1000 * s.price_output_per_1k, 6)
    trace = (values.get("trace") or [])[before["trace"]:]
    add_metric("Turns", 1)
    add_metric("TurnCostMicroUSD", cost * 1_000_000)
    add_metric("TurnLatencyMs", sum(t["ms"] for t in trace), MetricUnit.Milliseconds)
    return {
        "session_id": session_id,
        "reply": values.get("reply"),
        "status": "pending_approval" if pending else (values.get("status") or "complete"),
        "intent": values.get("intent"),
        "reply_kind": values.get("reply_kind"),
        "policy_code": (values.get("policy") or {}).get("code"),
        "citations": ((values.get("knowledge") or {}).get("citations") or []),
        "actions": (values.get("actions") or [])[before["actions"]:],
        "guardrail_events": (values.get("guardrail_events") or [])[before["guardrail_events"]:],
        "trace": trace,
        "usage": {"input_tokens": tin, "output_tokens": tout,
                  "llm_calls": usage_now.get("llm_calls", 0) - usage_before.get("llm_calls", 0), "cost_usd": cost},
    }


def _run(graph, payload, session_id: str) -> dict:
    snap = graph.get_state(_config(session_id))
    before = _counts(snap.values)
    try:
        values = graph.invoke(payload, _config(session_id))
    except TransientToolError as exc:
        add_metric("RunsFailed", 1)
        logger.warning("run failed after retries; checkpoint kept", extra={"session_id": session_id})
        raise HTTPException(status_code=503, detail={
            "error": "temporarily_unavailable", "session_id": session_id,
            "message": "A backend service is busy. Your conversation is saved; call /sessions/{id}/resume."}) from exc
    pending = bool(graph.get_state(_config(session_id)).interrupts)
    return _turn_result(session_id, before, values, pending)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/chat")
def chat(req: ChatRequest, _: str = Depends(_require("client"))):
    graph, _deps = runtime()
    session_id = req.session_id or uuid.uuid4().hex
    snap = graph.get_state(_config(session_id))
    if snap.values and snap.values.get("customer_id") != req.customer_id:
        raise HTTPException(status_code=403, detail="This session belongs to another customer")
    if snap.interrupts:
        raise HTTPException(status_code=409, detail="This session is waiting for a refund approval")
    if snap.next:
        raise HTTPException(status_code=409, detail="This session has an unfinished run; call /sessions/{id}/resume")
    return _run(graph, {"customer_id": req.customer_id, "message": req.message}, session_id)


@app.post("/sessions/{session_id}/resume")
def resume(session_id: str, _: str = Depends(_require("client"))):
    graph, _deps = runtime()
    snap = graph.get_state(_config(session_id))
    if not snap.next or snap.interrupts:
        raise HTTPException(status_code=409, detail="Nothing to resume")
    return _run(graph, None, session_id)


@app.get("/sessions/{session_id}")
def get_session(session_id: str, _: str = Depends(_require("client"))):
    graph, _deps = runtime()
    snap = graph.get_state(_config(session_id))
    if not snap.values:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "customer_id": snap.values.get("customer_id"),
            "messages": snap.values.get("messages", []), "pending_approval": bool(snap.interrupts),
            "next": list(snap.next)}


@app.get("/approvals")
def list_approvals(_: str = Depends(_require("approver"))):
    _graph, deps = runtime()
    return deps.tools.list_pending_approvals()


@app.post("/approvals/{session_id}")
def decide(session_id: str, req: ApprovalRequest, _: str = Depends(_require("approver"))):
    graph, _deps = runtime()
    if not graph.get_state(_config(session_id)).interrupts:
        raise HTTPException(status_code=404, detail="No pending approval for this session")
    add_metric("ApprovalsDecided", 1)
    return _run(graph, Command(resume={"decision": req.decision, "approver": req.approver, "note": req.note}),
                session_id)


_mangum = Mangum(app, lifespan="off")


@logger.inject_lambda_context(correlation_id_path="requestContext.requestId")
@metrics.log_metrics
@tracer.capture_lambda_handler
def handler(event, context):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    return _mangum(event, context)


def seed_handler(event, context):
    """Run by a CDK Trigger after deploy: loads the demo customers and orders."""
    from .services import build_deps

    count = build_deps(settings()).tools.seed_demo_data()
    logger.info("seeded demo data", extra={"orders": count})
    return {"orders": count}
