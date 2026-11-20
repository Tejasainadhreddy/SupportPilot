"""The MCP knowledge client: one long-lived connection, hard deadlines, graceful 'warming up'."""

import asyncio
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import Chat

from supportpilot.graph import build_graph
from supportpilot.knowledge import KnowledgeUnavailable, MCPKnowledgeClient


def result(text=None, error=None):
    if error:
        return SimpleNamespace(is_error=True, content=[SimpleNamespace(text=error)], structured_content=None)
    return SimpleNamespace(is_error=False, content=[],
                           structured_content={"answer": text, "refused": False, "citations": []})


class FakeClient:
    """Stands in for mcp Client: an async context manager with call_tool, driven by a script."""

    def __init__(self, script):
        self.script = script

    def __call__(self, target):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, args):
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, float):
            await asyncio.sleep(step)
            return result("late")
        return step


def test_reconnects_after_a_broken_connection():
    fake = FakeClient([RuntimeError("pipe closed"), result("second answer")])
    client = MCPKnowledgeClient(target=None, client_factory=fake)
    with pytest.raises(ConnectionError):
        client.ask("q1")
    assert client.ask("q2")["answer"] == "second answer"
    assert client.connects == 2


def test_slow_call_hits_the_deadline_and_the_connection_survives():
    fake = FakeClient([5.0, result("fast")])
    client = MCPKnowledgeClient(target=None, client_factory=fake, timeout_s=0.3)
    start = time.perf_counter()
    with pytest.raises(KnowledgeUnavailable):
        client.ask("slow")
    assert time.perf_counter() - start < 2
    assert client.ask("next")["answer"] == "fast" and client.connects == 1


def test_warming_up_becomes_knowledge_unavailable():
    fake = FakeClient([result(error="warming_up: the knowledge base is waking up")])
    with pytest.raises(KnowledgeUnavailable):
        MCPKnowledgeClient(target=None, client_factory=fake).ask("q")


def test_agent_replies_politely_and_does_not_retry_while_knowledge_wakes(deps):
    calls = []

    class Waking:
        def ask(self, q):
            calls.append(q)
            raise KnowledgeUnavailable("warming_up")

    deps.knowledge = Waking()
    out = Chat(build_graph(deps)).say("What is the policy on digital gift cards?")
    assert out["reply_kind"] == "knowledge_unavailable" and "about 10 seconds" in out["reply"]
    assert len(calls) == 1  # no retry storm against the deadline


def test_warmup_is_fire_and_forget(monkeypatch):
    import httpx

    posted = []
    monkeypatch.setattr(httpx, "post", lambda url, **kw: posted.append((url, kw["headers"])))
    MCPKnowledgeClient(target=None, warmup_url="https://kmcp.example/warmup", api_key="k").warmup()
    for _ in range(50):
        if posted:
            break
        time.sleep(0.01)
    assert posted == [("https://kmcp.example/warmup", {"x-api-key": "k"})]


@pytest.mark.integration
@pytest.mark.skipif(importlib.util.find_spec("knowledgemcp") is None, reason="knowledgemcp not installed")
def test_one_connection_serves_many_questions_against_real_knowledgemcp():
    from knowledgemcp.config import Prices
    from knowledgemcp.embeddings import HashEmbedder
    from knowledgemcp.ingest import Ingestor
    from knowledgemcp.llm import ExtractiveLLM
    from knowledgemcp.mcp_server import LocalBackend, create_server
    from knowledgemcp.rag import RagService
    from knowledgemcp.rerank import LexicalReranker
    from knowledgemcp.search import HybridRetriever
    from knowledgemcp.store import InMemoryStore

    store, emb = InMemoryStore(), HashEmbedder()
    Ingestor(store, emb).ingest_folder(Path(__file__).resolve().parents[1] / "knowledge_docs")
    rag = RagService(HybridRetriever(store, emb, LexicalReranker()), ExtractiveLLM(), Prices())
    client = MCPKnowledgeClient(create_server(LocalBackend(["support"], rag=rag, store=store)))
    answers = [client.ask(q)["answer"] for q in (
        "Are digital gift cards non-refundable?",
        "How many business days does express shipping take?",
        "When does the password reset link expire?")]
    # The offline stand-in model's answer quality is not under test here; connection reuse is.
    assert "non-refundable" in answers[0] and "1 to 2 business days" in answers[1] and answers[2]
    assert client.connects == 1
