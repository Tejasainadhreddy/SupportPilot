"""Local tools: create tables in DynamoDB Local, chat in the terminal, approve refunds.

    python -m supportpilot.cli init                         # create tables + demo orders
    python -m supportpilot.cli chat --customer C-100        # talk to the agent
    python -m supportpilot.cli approvals                    # list refunds waiting for a human
    python -m supportpilot.cli approve <session_id> --reject --note "used item"
"""

from __future__ import annotations

import argparse
import uuid

import boto3
from langgraph.types import Command

from .checkpoints import create_checkpoint_table
from .config import get_settings
from .services import build_app_graph, build_deps
from .tools import create_table


def main() -> None:
    p = argparse.ArgumentParser(prog="supportpilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    c = sub.add_parser("chat")
    c.add_argument("--customer", default="C-100")
    c.add_argument("--session")
    sub.add_parser("approvals")
    a = sub.add_parser("approve")
    a.add_argument("session")
    a.add_argument("--reject", action="store_true")
    a.add_argument("--note", default="")
    args = p.parse_args()
    s = get_settings()

    if args.cmd == "init":
        ddb = boto3.resource("dynamodb", region_name=s.region, endpoint_url=s.dynamodb_endpoint or None)
        existing = {t.name for t in ddb.tables.all()}
        if s.data_table not in existing:
            create_table(ddb, s.data_table).wait_until_exists()
        if s.checkpoint_table not in existing:
            create_checkpoint_table(ddb, s.checkpoint_table).wait_until_exists()
        print(f"seeded {build_deps(s).tools.reset_demo_data()} demo orders")
        return

    graph, deps = build_app_graph(s)
    if args.cmd == "approvals":
        for item in deps.tools.list_pending_approvals():
            print(f"{item['thread_id']}  {item['order_id']}  ${item['amount']:.2f}  {item['reason']}")
    elif args.cmd == "approve":
        decision = {"decision": "reject" if args.reject else "approve", "approver": "cli", "note": args.note}
        out = graph.invoke(Command(resume=decision), {"configurable": {"thread_id": args.session}})
        print(f"assistant> {out['reply']}")
    else:
        session = args.session or uuid.uuid4().hex
        config = {"configurable": {"thread_id": session}}
        print(f"session {session} (customer {args.customer}). Ctrl-C to quit.")
        while True:
            message = input("you> ").strip()
            if not message:
                continue
            out = graph.invoke({"customer_id": args.customer, "message": message}, config)
            print(f"assistant> {out['reply']}")
            if graph.get_state(config).interrupts:
                print(f"[waiting for approval: python -m supportpilot.cli approve {session}]")
                break


if __name__ == "__main__":
    main()
