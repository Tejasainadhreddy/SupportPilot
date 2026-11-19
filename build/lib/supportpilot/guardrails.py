"""Guardrails for a support agent that can move money.

- Input: length limits, prompt-injection detection, and PII redaction (card numbers and SSNs are removed
  before the message is stored in checkpoints or sent to a model).
- Policy: refunds are decided by code, not by the model. The model never chooses whether money moves.
- Output: a reply may not claim a refund happened unless a refund record exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

MAX_MESSAGE_CHARS = 2000

_INJECTION = re.compile("|".join([
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|messages)",
    r"disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|system)",
    r"\byou\s+are\s+now\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak",
    r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions)",
    r"\bsystem\s+prompt\b",
    r"\bact\s+as\s+(an?\s+)?(admin|administrator|supervisor|unrestricted)\b",
    r"\b(auto[- ]?approve|bypass\s+(the\s+)?(approval|policy|limit))\b",
]), re.IGNORECASE)
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_REFUND_CLAIM = re.compile(r"\b(refund(ed)?\s+(has\s+been|was|is)\s+(issued|processed|approved|sent)|"
                           r"(have|has|we've|i've)\s+(issued|processed|refunded))\b", re.IGNORECASE)


class GuardrailViolation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def redact_pii(text: str) -> tuple[str, list[str]]:
    events = []
    if _CARD.search(text):
        text = _CARD.sub("[card number removed]", text)
        events.append("pii_card_redacted")
    if _SSN.search(text):
        text = _SSN.sub("[ssn removed]", text)
        events.append("pii_ssn_redacted")
    return text, events


def check_message(text: str) -> tuple[str, list[str]]:
    """Returns (clean_text, events). Raises GuardrailViolation for blocked input."""
    if not isinstance(text, str) or not text.strip():
        raise GuardrailViolation("empty", "Message is empty.")
    text = re.sub(r"\s+", " ", "".join(c for c in text if c.isprintable() or c == "\n")).strip()
    if len(text) > MAX_MESSAGE_CHARS:
        raise GuardrailViolation("too_long", f"Message must be at most {MAX_MESSAGE_CHARS} characters.")
    if _INJECTION.search(text):
        raise GuardrailViolation("prompt_injection", "Message looks like an attempt to override the assistant's rules.")
    return redact_pii(text)


@dataclass
class PolicyDecision:
    allowed: bool
    needs_approval: bool
    code: str
    detail: str


def refund_policy(order: dict, amount: float, already_refunded: bool, now: datetime,
                  auto_limit: float, window_days: int) -> PolicyDecision:
    """Pure function: the single place where refund rules live (mirrors refund-policy.md)."""
    if already_refunded:
        return PolicyDecision(False, False, "already_refunded", "This order has already been refunded.")
    if order.get("non_refundable"):
        return PolicyDecision(False, False, "non_refundable", "Digital gift cards are non-refundable.")
    if order["status"] != "delivered" or not order.get("delivered_at"):
        return PolicyDecision(False, False, "not_delivered",
                              f"The order is '{order['status']}', and only delivered orders can be refunded.")
    days = (now - datetime.fromisoformat(order["delivered_at"])).days
    if days > window_days:
        return PolicyDecision(False, False, "outside_window",
                              f"The order was delivered {days} days ago; refunds are allowed within "
                              f"{window_days} days.")
    if amount <= 0 or amount > float(order["total"]) + 1e-9:
        return PolicyDecision(False, False, "invalid_amount",
                              "The refund amount must be between 0 and the order total of "
                              f"${float(order['total']):.2f}.")
    if amount > auto_limit:
        return PolicyDecision(True, True, "needs_approval",
                              f"Refunds above ${auto_limit:.0f} need approval from a support lead.")
    return PolicyDecision(True, False, "auto_approved", "Within the automatic refund limit.")


def check_reply(reply: str, refund_issued: bool) -> tuple[str, list[str]]:
    """Block replies that claim a refund happened when it did not."""
    if not refund_issued and _REFUND_CLAIM.search(reply):
        return ("I can't confirm a refund for this request. I can create a ticket so a support agent can follow up.",
                ["false_refund_claim_blocked"])
    return reply, []
