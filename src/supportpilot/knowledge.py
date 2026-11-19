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
