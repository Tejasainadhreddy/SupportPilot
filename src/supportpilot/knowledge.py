"""Knowledge answers come from KnowledgeMCP (Project 1) over the MCP protocol.

KnowledgeClient is the small interface the graph needs. StaticKnowledge is an offline stand-in that
reads knowledge_docs/ and returns the best-matching sentence, so tests and local runs need no AWS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

DOCS = Path(__file__).resolve().parents[2] / "knowledge_docs"
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


class MCPKnowledgeClient:
    """Calls KnowledgeMCP's `ask_question` tool over the MCP protocol.

    Production: launches the `knowledgemcp-mcp` stdio server (installed from the KnowledgeMCP repo), which
    calls the deployed KnowledgeMCP API with this service's API key, so KnowledgeMCP enforces which
    documents SupportPilot may read.
    Tests: pass an in-process MCPServer and the client talks to it over an in-memory transport.
    """

    def __init__(self, target, timeout_s: float = 25.0):
        self.target = target  # StdioServerParameters | MCPServer
        self.timeout_s = timeout_s

    @classmethod
    def from_settings(cls, s) -> MCPKnowledgeClient:
        import os

        from mcp.client.client import StdioServerParameters

        key = s.kmcp_api_key
        if s.kmcp_api_key_secret_arn and not key:
            import boto3

            key = boto3.client("secretsmanager", region_name=s.region).get_secret_value(
                SecretId=s.kmcp_api_key_secret_arn)["SecretString"]
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "PYTHONPATH", "LAMBDA_TASK_ROOT")}
        env.update(KMCP_API_URL=s.kmcp_api_url, KMCP_API_KEY=key)
        return cls(StdioServerParameters(command=s.kmcp_command, args=[], env=env))

    def ask(self, question: str) -> dict:
        import anyio
        from mcp.client.client import Client

        async def call() -> dict:
            with anyio.fail_after(self.timeout_s):
                async with Client(self.target) as client:
                    result = await client.call_tool("ask_question", {"question": question})
            if result.is_error:
                raise ConnectionError(" ".join(getattr(c, "text", "") for c in result.content)[:300])
            return result.structured_content

        return anyio.run(call)
