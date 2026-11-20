import importlib.util

import pytest
from conftest import NOW, Chat
from fastapi.testclient import TestClient

from supportpilot import api
from supportpilot.checkpoints import build_checkpointer, create_checkpoint_table
from supportpilot.config import Settings
from supportpilot.graph import build_graph
from supportpilot.tools import TransientToolError

CLIENT = {"x-api-key": "client-key"}
APPROVER = {"x-api-key": "approver-key"}


@pytest.fixture
def dynamo_saver(aws):
    create_checkpoint_table(aws, "supportpilot-checkpoints")
    return lambda: build_checkpointer(Settings(checkpoint_table="supportpilot-checkpoints"))


@pytest.fixture
def client(deps, dynamo_saver):
    graph = build_graph(deps, dynamo_saver())
    api.runtime.cache_clear()
    api.app.dependency_overrides = {api.api_keys: lambda: {"client": "client-key", "approver": "approver-key"}}
    original = api.runtime
    api.runtime = lambda: (graph, deps)
    yield TestClient(api.app)
    api.runtime = original
    api.app.dependency_overrides = {}


def test_dynamodb_checkpoint_survives_a_new_process(deps, dynamo_saver, tools):
    """Pause for approval in one 'Lambda', approve from a brand-new graph + saver (another Lambda)."""
    first = Chat(build_graph(deps, dynamo_saver()), thread_id="thread-abc-123")
    assert first.say("Refund ORD-1002, the keyboard is defective")["status"] == "pending_approval"

    second = Chat(build_graph(deps, dynamo_saver()), thread_id="thread-abc-123")
    assert second.graph.get_state(second.config).interrupts  # state came from DynamoDB
    final = second.resume({"decision": "approve", "approver": "lead"})
    assert final["reply_kind"] == "refund_issued" and tools.get_refund("ORD-1002")["approved_by"] == "lead"
    assert [m["role"] for m in final["messages"]] == ["customer", "assistant", "assistant"]


def test_api_auth(client):
    assert client.get("/health").status_code == 200
    assert client.post("/chat", json={"customer_id": "C-100", "message": "hi"}).status_code == 401
    assert client.get("/approvals", headers=CLIENT).status_code == 401  # client key can't approve


def test_api_chat_turns_and_session_history(client):
    r = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "message": "I want a refund"}).json()
    sid = r["session_id"]
    assert r["status"] == "complete" and "order number" in r["reply"] and r["actions"] == []
    r2 = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "session_id": sid,
                                                   "message": "ORD-1001, it arrived broken"}).json()
    assert [a["tool"] for a in r2["actions"]] == ["get_order", "issue_refund"]
    assert [t["node"] for t in r2["trace"]][0] == "guard_input"
    history = client.get(f"/sessions/{sid}", headers=CLIENT).json()
    assert len(history["messages"]) == 4


def test_api_human_in_the_loop(client, tools):
    r = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100",
                                                  "message": "Refund ORD-1002, it's defective"}).json()
    sid = r["session_id"]
    assert r["status"] == "pending_approval"
    blocked = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "session_id": sid, "message": "?"})
    assert blocked.status_code == 409

    pending = client.get("/approvals", headers=APPROVER).json()
    assert pending[0]["thread_id"] == sid and pending[0]["amount"] == 249.99
    done = client.post(f"/approvals/{sid}", headers=APPROVER, json={"decision": "approve", "approver": "maya"}).json()
    assert done["status"] == "complete" and done["actions"][-1]["approved_by"] == "maya"
    assert client.post(f"/approvals/{sid}", headers=APPROVER, json={"decision": "approve"}).status_code == 404


def test_api_session_belongs_to_one_customer(client):
    sid = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "message": "hello"}).json()["session_id"]
    r = client.post("/chat", headers=CLIENT, json={"customer_id": "C-200", "session_id": sid, "message": "hi"})
    assert r.status_code == 403


def test_api_503_then_resume(client, deps):
    real = deps.tools.get_order
    deps.tools.get_order = lambda *a: (_ for _ in ()).throw(TransientToolError("ThrottlingException"))
    r = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "message": "Where is ORD-1003?"})
    assert r.status_code == 503
    sid = r.json()["detail"]["session_id"]
    deps.tools.get_order = real
    resumed = client.post(f"/sessions/{sid}/resume", headers=CLIENT).json()
    assert "shipped" in resumed["reply"]


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("knowledgemcp") is None, reason="knowledgemcp not installed")
def test_knowledge_over_real_mcp_protocol(deps):
    """SupportPilot -> MCP client -> KnowledgeMCP MCP server -> hybrid search + guardrails -> cited answer."""
    from pathlib import Path

    from knowledgemcp.config import Prices
    from knowledgemcp.embeddings import HashEmbedder
    from knowledgemcp.ingest import Ingestor
    from knowledgemcp.llm import ExtractiveLLM
    from knowledgemcp.mcp_server import LocalBackend, create_server
    from knowledgemcp.rag import RagService
    from knowledgemcp.rerank import LexicalReranker
    from knowledgemcp.search import HybridRetriever
    from knowledgemcp.store import InMemoryStore

    from supportpilot.knowledge import MCPKnowledgeClient

    store, emb = InMemoryStore(), HashEmbedder()
    Ingestor(store, emb).ingest_folder(Path(__file__).resolve().parents[1] / "knowledge_docs")
    rag = RagService(HybridRetriever(store, emb, LexicalReranker()), ExtractiveLLM(), Prices())
    server = create_server(LocalBackend(["support"], rag=rag, store=store))

    deps.knowledge = MCPKnowledgeClient(server)
    q = "How many business days until approved refunds reach the original payment method?"
    out = Chat(build_graph(deps)).say(q)
    assert out["intent"] == "policy_question" and "5 to 7 business days" in out["reply"]
    assert out["knowledge"]["citations"][0]["title"] == "Refund Policy"
    assert NOW.year == 2026


def test_new_session_warms_up_knowledge_once(client, deps):
    warmups = []
    deps.knowledge.warmup = lambda: warmups.append(1)
    sid = client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "message": "hello"}).json()["session_id"]
    client.post("/chat", headers=CLIENT, json={"customer_id": "C-100", "session_id": sid, "message": "thanks"})
    assert warmups == [1]
