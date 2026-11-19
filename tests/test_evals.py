"""Run the conversation evals offline in CI. Policy compliance must be 100% with any model,
because refund rules, approvals and injection blocking are enforced in code, not by the model."""

from evals.run_evals import LocalTarget, load_cases, run
from supportpilot.llm import rule_reason


def test_conversation_evals_policy_is_never_violated():
    result = run(LocalTarget(), load_cases())
    s = result["summary"]
    assert s["conversations"] == 30 and s["errors"] == 0
    assert s["policy_compliance"] == 1.0
    assert s["overall_pass_rate"] >= 0.85  # offline rules router; regression guard only


def test_reason_patterns_handle_plurals():
    assert rule_reason("one cushion was missing parts") == "item was missing parts"
