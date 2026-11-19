"""Wire the real (AWS) or offline components from Settings."""

from __future__ import annotations

import boto3

from .checkpoints import build_checkpointer
from .config import Settings
from .graph import Deps, build_graph
from .knowledge import MCPKnowledgeClient, StaticKnowledge
from .llm import BedrockSupportLLM, RulesLLM
from .tools import SupportTools


def build_deps(s: Settings) -> Deps:
    table = boto3.resource("dynamodb", region_name=s.region,
                           endpoint_url=s.dynamodb_endpoint or None).Table(s.data_table)
    llm = (BedrockSupportLLM(boto3.client("bedrock-runtime", region_name=s.region), s.llm_model_id)
           if s.llm == "bedrock" else RulesLLM())
    knowledge = MCPKnowledgeClient.from_settings(s) if s.knowledge == "mcp" else StaticKnowledge()
    return Deps(llm=llm, tools=SupportTools(table), knowledge=knowledge, settings=s)


def build_app_graph(s: Settings, deps: Deps | None = None, checkpointer=None):
    deps = deps or build_deps(s)
    return build_graph(deps, checkpointer or build_checkpointer(s)), deps
