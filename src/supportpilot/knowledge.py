"""Knowledge answers come from KnowledgeMCP (Project 1) over the MCP protocol.

KnowledgeClient is the small interface the graph needs. StaticKnowledge is an offline stand-in that
reads knowledge_docs/ and returns the best-matching sentence, so tests and local runs need no AWS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

_HERE = Path(__file__).resolve()
# repo checkout: <repo>/knowledge_docs ; container image: /var/task/knowledge_docs
DOCS = next((p / "knowledge_docs" for p in (_HERE.parents[2], _HERE.parents[1]) if (p / "knowledge_docs").is_dir()),
            _HERE.parents[2] / "knowledge_docs")
_STOP = set("a an and are as at be by can do does for from how i if in is it of on or the this to we what when "
            "where which who why will with you your my me".split())


class KnowledgeClient(Protocol):
    def ask(self, question: str) -> dict:
        """Returns {"answer": str, "refused": bool, "citations": [{"ref","title","location","doc_id"}]}."""
        ...


def _terms(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


class StaticKnowledge:
    def __init__(self, root: Path = DOCS):
        self.sentences: list[tuple[str, str, str]] = []  # (title, section, sentence)
        for path in sorted(root.rglob("*.md")):
            title, section = path.stem.replace("-", " ").title(), ""
            for line in path.read_text().splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## "):
                    section = line[3:].strip()
                elif line.strip():
                    for s in re.split(r"(?<=[.!?])\s+", line.strip()):
                        self.sentences.append((title, section, s))

    def ask(self, question: str) -> dict:
        q = _terms(question)
        best, score = None, 0.0
        for title, section, sentence in self.sentences:
            overlap = len(q & _terms(sentence)) / max(1, len(q))
            if overlap > score:
                best, score = (title, section, sentence), overlap
        if not best or score < 0.4:
            return {"answer": "", "refused": True, "citations": []}
        title, section, sentence = best
        return {"answer": f"{sentence} [S1]", "refused": False,
                "citations": [{"ref": "S1", "title": title, "location": f"section '{title} > {section}'",
                               "doc_id": title.lower().replace(" ", "-")}]}


class KnowledgeUnavailable(RuntimeError):
    """KnowledgeMCP is waking up or too slow right now. Not retried: the agent replies and moves on."""


class MCPKnowledgeClient:
    """Calls KnowledgeMCP's `ask_question` tool over the MCP protocol, keeping ONE connection open.

    Production: launches the KnowledgeMCP MCP server once per Lambda container (stdio subprocess), which
    calls the deployed KnowledgeMCP API with SupportPilot's own key, so KnowledgeMCP still decides which
    documents SupportPilot may read. Later questions reuse that connection instead of paying ~1 s to start
    a new process each time.

    Deadlines: each call is capped at `timeout_s` and the subprocess is told to stop waiting for a sleeping
    database after `KMCP_WAIT_BUDGET_S`, so a cold KnowledgeMCP can never push this Lambda past its limit.

    How the connection is kept: MCP clients are async, and their context must be entered and exited by the
    same task. A single worker task on a background event loop owns the connection; `ask()` (sync, called
    from graph nodes) hands it questions through a queue and waits for the answer.
    """

    def __init__(self, target, timeout_s: float = 12.0, warmup_url: str | None = None,
                 api_key: str | None = None, client_factory=None):
        self.target = target  # StdioServerParameters | MCPServer (in-memory, for tests)
        self.timeout_s = timeout_s
        self.warmup_url = warmup_url
        self.api_key = api_key
        self.client_factory = client_factory
        self.connects = 0
        self._loop = None
        self._queue = None
        self._ready = None

    @classmethod
    def from_settings(cls, s) -> MCPKnowledgeClient:
        import os
        import sys

        from mcp.client.client import StdioServerParameters

        key = s.kmcp_api_key
        if s.kmcp_api_key_secret_arn and not key:
            import boto3

            key = boto3.client("secretsmanager", region_name=s.region).get_secret_value(
                SecretId=s.kmcp_api_key_secret_arn)["SecretString"]
        # Run the server with this same interpreter and import path (works inside Lambda images too).
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG")}
        env.update(KMCP_API_URL=s.kmcp_api_url, KMCP_API_KEY=key, KMCP_WAIT_BUDGET_S=str(s.kmcp_wait_budget_s),
                   PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        params = (StdioServerParameters(command=s.kmcp_command, args=[], env=env) if s.kmcp_command else
                  StdioServerParameters(command=sys.executable, args=["-m", "knowledgemcp.mcp_server"], env=env))
        return cls(params, timeout_s=s.kmcp_call_timeout_s, api_key=key,
                   warmup_url=s.kmcp_api_url.rstrip("/") + "/warmup" if s.kmcp_api_url else None)

    # ---- background worker -------------------------------------------------------------------------
    def _start(self) -> None:
        import asyncio
        import threading

        if self._loop is not None:
            return
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-knowledge").start()
        asyncio.run_coroutine_threadsafe(self._worker(), self._loop)
        self._ready.wait(5)

    async def _worker(self) -> None:
        import asyncio

        import anyio

        if self.client_factory is None:
            from mcp.client.client import Client

            self.client_factory = Client
        self._queue = asyncio.Queue()
        self._ready.set()
        pending = None
        while True:  # (re)connect loop
            try:
                async with self.client_factory(self.target) as client:
                    self.connects += 1
                    while True:
                        question, future = pending or await self._queue.get()
                        pending = None
                        try:
                            with anyio.fail_after(self.timeout_s):
                                result = await client.call_tool("ask_question", {"question": question})
                        except TimeoutError:
                            future.set_exception(KnowledgeUnavailable("knowledge call timed out"))
                            continue
                        except Exception as exc:  # broken connection: fail this call, reconnect for the next
                            future.set_exception(ConnectionError(f"MCP connection failed: {exc}"))
                            raise
                        future.set_result(result)
            except Exception:
                await asyncio.sleep(0.2)

    # ---- public API ----------------------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        import concurrent.futures

        self._start()
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (question, future))
        try:
            result = future.result(timeout=self.timeout_s + 3)
        except concurrent.futures.TimeoutError as exc:
            raise KnowledgeUnavailable("knowledge call timed out") from exc
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)[:300]
            if "warming_up" in text:
                raise KnowledgeUnavailable(text)
            raise ConnectionError(text)
        return result.structured_content

    def warmup(self) -> None:
        """Fire-and-forget: ask KnowledgeMCP to start waking its database before the customer needs it."""
        if not self.warmup_url:
            return
        import threading

        def go():
            try:
                import httpx

                httpx.post(self.warmup_url, headers={"x-api-key": self.api_key or ""}, timeout=3.0)
            except Exception:
                pass

        threading.Thread(target=go, daemon=True).start()
