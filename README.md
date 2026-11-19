# SupportPilot

[![CI](https://github.com/Tejasainadhreddy/SupportPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Tejasainadhreddy/SupportPilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/agent-LangGraph-5eead4)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK%20(Python)-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A LangGraph customer-support agent that can safely move money.**
It answers policy questions with citations through [KnowledgeMCP](https://github.com/Tejasainadhreddy/KnowledgeMCP)
over MCP, looks up orders, issues refunds and creates tickets. Refunds above a limit pause until a human approves
them, and every step is checkpointed in DynamoDB, so a conversation survives crashes and can resume days later.

![SupportPilot preview](docs/preview.png)

## What it does

| | |
|---|---|
| **Explicit graph** | LangGraph `StateGraph` with typed state, conditional routing, per-node retry policies and an append-only audit trail (actions, node trace, guardrail events, token usage). |
| **Knowledge with citations** | Policy questions go to KnowledgeMCP's `ask_question` tool over the **MCP protocol** (SupportPilot is the MCP client). KnowledgeMCP does hybrid search over the support documents and only answers with citations. |
| **Action tools** | Look up an order, issue a refund (one per order, enforced by a DynamoDB conditional write), create a ticket. |
| **Human-in-the-loop** | Refunds above $100 call LangGraph `interrupt()`. The customer is told it's under review; a support lead approves or rejects through `/approvals`; the run resumes from the exact checkpoint. |
| **Checkpoints and crash recovery** | `DynamoDBSaver` stores state after every node (with TTL). If a dependency keeps failing after retries, the API returns 503 and `/sessions/{id}/resume` continues from the failed node without re-running earlier ones. |
| **Guardrails** | Prompt-injection detection (including "auto-approve"/"bypass the limit"), card-number and SSN redaction before anything is stored, refund rules decided by code (never by the model), required fields extracted by code, and an output check that blocks replies claiming a refund that didn't happen. |
| **Model** | Claude on Bedrock (Converse API) for routing and reply wording; money-related replies use exact templates. Invalid model JSON is retried once, then falls back to a deterministic router. |
| **Evaluation** | 30 scripted conversations scored for resolution, tool-use correctness, policy compliance, cost per conversation and turn latency. CI requires 100% policy compliance. |
| **Observability** | Structured logs, CloudWatch EMF metrics (turns, refunds, approvals, per-node latency, cost), X-Ray, and a CloudWatch dashboard. |
| **Infrastructure** | AWS CDK (Python): Lambda container + HTTP API + DynamoDB on-demand (no idle compute). Optional `-c fargate=true` demo mode runs the same image on ECS Fargate behind an ALB. |

## The graph

Generated from the compiled graph (`graph.get_graph().draw_mermaid()`), so it always matches the code:

![LangGraph](docs/graph.png)

| Node | What it does |
|---|---|
| `guard_input` | Validates the message, blocks injection attempts, redacts card numbers / SSNs |
| `classify` | Claude picks the intent; code extracts order ID and amount |
| `knowledge` | MCP call to KnowledgeMCP `ask_question` (retried on transient errors) |
| `order_status` | Looks up the order for *this* customer only |
| `refund_prepare` | Collects missing fields over turns, loads the order, applies the refund policy |
| `notify_pending` → `human_approval` | Tells the customer, records the request, then `interrupt()` until a human decides |
| `execute_refund` | Idempotent refund write; records who approved it |
| `create_ticket` | Hands the conversation to a human agent |
| `respond` | Words the reply from verified facts only, then checks it |

## Architecture

![Architecture](docs/architecture.png)

## Results

Measured against the deployed stack (Claude on Bedrock, real DynamoDB, live KnowledgeMCP):

| Metric | Value |
|---|---|
| Resolution rate | _to be measured after deployment_ |
| Tool-use accuracy | _to be measured_ |
| Policy compliance | _to be measured_ |
| Avg model cost per conversation | _to be measured_ |
| Turn latency p50 / p95 | _to be measured_ |

CI runs the same 30 conversations with an offline keyword router on every push. Those numbers only check the
plumbing, but CI fails if policy compliance drops below 100%.

## Run it locally

```bash
docker compose up --build                       # DynamoDB Local + API with offline models
open http://localhost:8000/docs                 # x-api-key: local-client (chat) / local-approver (approvals)
```

Or chat in the terminal:

```bash
pip install -e ".[dev,local]"
export DYNAMODB_ENDPOINT=http://localhost:8001 AWS_ACCESS_KEY_ID=local AWS_SECRET_ACCESS_KEY=local
python -m supportpilot.cli init
python -m supportpilot.cli chat --customer C-100       # try: "Refund ORD-1002, the keyboard is defective"
python -m supportpilot.cli approvals
python -m supportpilot.cli approve <session_id>
pytest -q                                              # 33 tests (moto for AWS)
```

Demo customers: `C-100` (orders ORD-1001…1004) and `C-200` (ORD-2001…2003), covering delivered, shipped,
processing, outside-the-window and gift-card cases.

## Deploy to AWS

Prerequisites: [KnowledgeMCP](https://github.com/Tejasainadhreddy/KnowledgeMCP) deployed, AWS CLI configured,
Docker running, Node.js 18+, Bedrock access to Claude Haiku 4.5 in us-west-2.

```bash
# 1. Give SupportPilot its own KnowledgeMCP key, limited to the support documents
cd ../KnowledgeMCP
aws s3 sync ../SupportPilot/knowledge_docs s3://<KnowledgeMCP BucketName>/        # uploads support/*.md
python scripts/create_api_key.py --user supportpilot --teams support --secret-id <KnowledgeMCP ApiKeysSecretArn>

# 2. Deploy
cd ../SupportPilot
pip install -e ".[dev,local]" -r infra/requirements.txt
cd infra
cdk deploy -c kmcp_api_url=<KnowledgeMCP ApiUrl> -c alert_email=you@example.com
aws secretsmanager put-secret-value --secret-id <KnowledgeMcpKeySecret> --secret-string kmcp_...

# 3. Talk to it
CLIENT_KEY=$(aws secretsmanager get-secret-value --secret-id <ClientKeySecret> --query SecretString --output text)
curl -s <ApiUrl>/chat -H "x-api-key: $CLIENT_KEY" -H "content-type: application/json" \
  -d '{"customer_id": "C-100", "message": "Refund ORD-1002, the keyboard is defective"}'
```

Optional demo on Fargate: `cdk deploy ... -c fargate=true` prints `FargateUrl`. The load balancer is billed by
the hour, so redeploy without the flag when the demo is over. `cdk destroy` removes everything.

## Evaluate

```bash
python -m evals.run_evals --target http --url <ApiUrl> \
  --client-key "$CLIENT_KEY" --approver-key "$APPROVER_KEY" --reset-table <DataTableName>
```

`--reset-table` restores the demo orders before each conversation so runs are repeatable.

## Project structure

```
src/supportpilot/
  graph.py         the LangGraph: nodes, routing, retries, interrupt
  state.py         typed state with reducers (messages, actions, trace, usage)
  tools.py         DynamoDB tools (orders, refunds, tickets, approvals)
  guardrails.py    input checks, PII redaction, refund policy, output check
  llm.py           Claude on Bedrock + offline rules model
  knowledge.py     MCP client for KnowledgeMCP + offline stand-in
  checkpoints.py   DynamoDB checkpointer
  api.py           FastAPI: chat, sessions, resume, approvals; Lambda handler
  cli.py           local init / chat / approve
infra/             AWS CDK app + synth tests
evals/             30 scripted conversations + scorer
knowledge_docs/    support policies (uploaded to KnowledgeMCP)
tests/             pytest (moto, botocore Stubber, real MCP integration)
```

## Limitations

- `customer_id` comes from the request body. A real deployment would take it from the customer's login token.
- Prompt-injection detection is pattern-based; the stronger protection is that money decisions are made by code.
- The MCP server is started per knowledge question inside the Lambda, which adds about a second. A long-lived
  MCP connection (or KnowledgeMCP over streamable HTTP) would remove that.
- Store, orders and policies are fictional.

## License

MIT
