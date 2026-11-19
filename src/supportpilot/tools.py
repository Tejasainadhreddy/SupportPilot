"""Action tools backed by DynamoDB (single-table design).

Keys:
  CUSTOMER#<id> / ORDER#<id>     an order (lookup by customer + order id proves ownership)
  ORDER#<id>    / REFUND         at most one refund per order (conditional write = idempotent)
  CUSTOMER#<id> / TICKET#<id>    support tickets
  APPROVAL      / <thread_id>    refunds waiting for a human
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

TRANSIENT_CODES = {"ProvisionedThroughputExceededException", "ThrottlingException", "InternalServerError",
                   "RequestLimitExceeded", "ServiceUnavailable"}


class TransientToolError(Exception):
    """A temporary failure. LangGraph's RetryPolicy retries nodes that raise this."""


class AlreadyRefunded(Exception):
    pass


def _plain(item: dict[str, Any] | None) -> dict[str, Any] | None:
    """DynamoDB returns Decimals; convert to float/int for JSON and the graph state."""
    if item is None:
        return None
    out = {}
    for k, v in item.items():
        if k in ("PK", "SK"):
            continue
        if isinstance(v, Decimal):
            v = float(v) if k in ("total", "amount") or v != v.to_integral_value() else int(v)
        out[k] = v
    return out


def _call(fn, **kwargs):
    try:
        return fn(**kwargs)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in TRANSIENT_CODES:
            raise TransientToolError(exc.response["Error"]["Code"]) from exc
        raise


class SupportTools:
    def __init__(self, table, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.table = table
        self.now = now

    # ---- orders -------------------------------------------------------------------------------
    def get_order(self, customer_id: str, order_id: str) -> dict | None:
        resp = _call(self.table.get_item, Key={"PK": f"CUSTOMER#{customer_id}", "SK": f"ORDER#{order_id}"})
        return _plain(resp.get("Item"))

    def get_refund(self, order_id: str) -> dict | None:
        resp = _call(self.table.get_item, Key={"PK": f"ORDER#{order_id}", "SK": "REFUND"})
        return _plain(resp.get("Item"))

    # ---- refunds ------------------------------------------------------------------------------
    def issue_refund(self, customer_id: str, order_id: str, amount: float, reason: str,
                     approved_by: str) -> dict:
        record = {
            "PK": f"ORDER#{order_id}", "SK": "REFUND",
            "refund_id": f"RF-{uuid.uuid4().hex[:8].upper()}", "order_id": order_id, "customer_id": customer_id,
            "amount": Decimal(str(round(amount, 2))), "reason": reason, "approved_by": approved_by,
            "created_at": self.now().isoformat(),
        }
        try:
            _call(self.table.put_item, Item=record, ConditionExpression="attribute_not_exists(PK)")
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise AlreadyRefunded(order_id) from exc
            raise
        return _plain(record)

    # ---- tickets ------------------------------------------------------------------------------
    def create_ticket(self, customer_id: str, subject: str, details: str, priority: str = "normal") -> dict:
        ticket = {
            "PK": f"CUSTOMER#{customer_id}", "SK": f"TICKET#{uuid.uuid4().hex[:8].upper()}",
            "subject": subject[:120], "details": details[:2000], "priority": priority, "status": "open",
            "created_at": self.now().isoformat(),
        }
        ticket["ticket_id"] = ticket["SK"].split("#", 1)[1]
        _call(self.table.put_item, Item=ticket)
        return _plain(ticket)

    def list_tickets(self, customer_id: str) -> list[dict]:
        resp = _call(self.table.query, KeyConditionExpression=Key("PK").eq(f"CUSTOMER#{customer_id}")
                     & Key("SK").begins_with("TICKET#"))
        return [_plain(i) for i in resp["Items"]]

    # ---- human approvals ------------------------------------------------------------------------
    def add_pending_approval(self, thread_id: str, payload: dict) -> None:
        item = {"PK": "APPROVAL", "SK": thread_id, "status": "pending", "created_at": self.now().isoformat(),
                **{k: Decimal(str(v)) if isinstance(v, float) else v for k, v in payload.items()}}
        _call(self.table.put_item, Item=item)  # same key on re-run, so it is idempotent

    def list_pending_approvals(self) -> list[dict]:
        resp = _call(self.table.query, KeyConditionExpression=Key("PK").eq("APPROVAL"),
                     FilterExpression=Attr("status").eq("pending"))
        return [{"thread_id": i["SK"], **_plain(i)} for i in resp["Items"]]

    def get_approval(self, thread_id: str) -> dict | None:
        item = _call(self.table.get_item, Key={"PK": "APPROVAL", "SK": thread_id}).get("Item")
        return {"thread_id": thread_id, **_plain(item)} if item else None

    def resolve_approval(self, thread_id: str, decision: str, approver: str, note: str = "") -> None:
        _call(self.table.update_item, Key={"PK": "APPROVAL", "SK": thread_id},
              UpdateExpression="SET #s = :s, approver = :a, note = :n, resolved_at = :t",
              ExpressionAttributeNames={"#s": "status"},
              ExpressionAttributeValues={":s": decision, ":a": approver, ":n": note, ":t": self.now().isoformat()})

    # ---- demo data ------------------------------------------------------------------------------
    def seed_demo_data(self) -> int:
        now = self.now()
        orders = [
            ("C-100", "ORD-1001", "delivered", 5, 45.00, "Wireless mouse", False),
            ("C-100", "ORD-1002", "delivered", 10, 249.99, "Mechanical keyboard and monitor arm", False),
            ("C-100", "ORD-1003", "shipped", None, 89.00, "Noise-cancelling headphones", False),
            ("C-100", "ORD-1004", "delivered", 45, 60.00, "Desk lamp", False),
            ("C-200", "ORD-2001", "delivered", 3, 120.00, "Ergonomic chair cushion set", False),
            ("C-200", "ORD-2002", "processing", None, 30.00, "USB-C cable pack", False),
            ("C-200", "ORD-2003", "delivered", 2, 50.00, "Digital gift card", True),
        ]
        with self.table.batch_writer() as batch:
            for customer, oid, status, days_ago, total, items, non_refundable in orders:
                batch.put_item(Item={
                    "PK": f"CUSTOMER#{customer}", "SK": f"ORDER#{oid}", "order_id": oid, "customer_id": customer,
                    "status": status, "total": Decimal(str(total)), "items": items, "non_refundable": non_refundable,
                    "delivered_at": (now - timedelta(days=days_ago)).isoformat() if days_ago is not None else None,
                    "eta": None if days_ago is not None else (now + timedelta(days=3)).date().isoformat(),
                })
        return len(orders)


def create_table(dynamodb, name: str):
    """Used by tests and local dev (CDK creates the real table)."""
    return dynamodb.create_table(
        TableName=name, BillingMode="PAY_PER_REQUEST",
        KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
        AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"},
                              {"AttributeName": "SK", "AttributeType": "S"}],
    )
