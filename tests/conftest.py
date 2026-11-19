import os
import uuid
from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ.setdefault("POWERTOOLS_TRACE_DISABLED", "1")
os.environ.setdefault("POWERTOOLS_LOG_LEVEL", "WARNING")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from supportpilot.config import Settings  # noqa: E402
from supportpilot.graph import Deps, build_graph  # noqa: E402
from supportpilot.knowledge import StaticKnowledge  # noqa: E402
from supportpilot.llm import RulesLLM  # noqa: E402
from supportpilot.tools import SupportTools, create_table  # noqa: E402

NOW = datetime(2026, 11, 19, 18, 0, tzinfo=UTC)


@pytest.fixture
def aws():
    with mock_aws():
        yield boto3.resource("dynamodb", region_name="us-west-2")


@pytest.fixture
def tools(aws):
    t = SupportTools(create_table(aws, "supportpilot-data"), now=lambda: NOW)
    t.seed_demo_data()
    return t


@pytest.fixture
def deps(tools):
    return Deps(llm=RulesLLM(), tools=tools, knowledge=StaticKnowledge(), settings=Settings(), now=lambda: NOW)


class Chat:
    """Small helper that drives the compiled graph like the API does."""

    def __init__(self, graph, customer_id="C-100", thread_id=None):
        self.graph = graph
        self.customer_id = customer_id
        self.config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}

    def say(self, message):
        return self.graph.invoke({"customer_id": self.customer_id, "message": message}, self.config)

    def resume(self, value):
        from langgraph.types import Command

        return self.graph.invoke(Command(resume=value), self.config)


@pytest.fixture
def graph(deps):
    return build_graph(deps, InMemorySaver())


@pytest.fixture
def chat(graph):
    return Chat(graph)
