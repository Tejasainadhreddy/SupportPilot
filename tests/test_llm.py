import boto3
from botocore.stub import ANY, Stubber

from supportpilot.guardrails import refund_policy
from supportpilot.llm import BedrockSupportLLM, RulesLLM

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def converse_response(text, tin=40, tout=8):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn",
            "usage": {"inputTokens": tin, "outputTokens": tout, "totalTokens": tin + tout}, "metrics": {"latencyMs": 5}}


def test_bedrock_route_parses_json():
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    stub = Stubber(client)
    stub.add_response("converse", converse_response('{"intent": "refund", "reason": "screen cracked"}'),
                      {"modelId": MODEL, "system": ANY, "messages": ANY, "inferenceConfig": ANY})
    with stub:
        route, usage = BedrockSupportLLM(client, MODEL).route("my screen is cracked, money back pls", None)
    assert (route.intent, route.reason) == ("refund", "screen cracked")
    assert (usage.input_tokens, usage.llm_calls) == (40, 1)


def test_bedrock_route_falls_back_after_two_bad_answers():
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    stub = Stubber(client)
    stub.add_response("converse", converse_response("sure! it's a refund"))
    stub.add_response("converse", converse_response('{"intent": "teleport"}'))
    with stub:
        route, usage = BedrockSupportLLM(client, MODEL).route("Where is my order ORD-1003?", None)
    assert route.intent == "order_status" and usage.llm_calls == 2


def test_money_replies_use_exact_templates_not_the_model():
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    with Stubber(client):  # no responses queued: any model call would fail the test
        text, usage = BedrockSupportLLM(client, MODEL).write_reply(
            "refund_issued", {"order_id": "ORD-1", "amount": 12.5, "refund_id": "RF-1"}, "x")
    assert "$12.50" in text and usage.llm_calls == 0


def test_rules_router():
    r = RulesLLM()
    assert r.route("hello there", None)[0].intent == "greeting"
    assert r.route("ORD-1001", "refund_details")[0].intent == "refund"
    assert r.route("can I return a gift card?", None)[0].intent == "policy_question"


def test_policy_boundaries():
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 11, 19, tzinfo=UTC)
    order = {"status": "delivered", "delivered_at": (now - timedelta(days=30)).isoformat(), "total": 100.0}
    assert refund_policy(order, 100.0, False, now, 100, 30).code == "auto_approved"  # exactly at both limits
    assert refund_policy(order, 100.01, False, now, 100, 30).code == "invalid_amount"
    order["total"] = 150.0
    assert refund_policy(order, 100.01, False, now, 100, 30).needs_approval
