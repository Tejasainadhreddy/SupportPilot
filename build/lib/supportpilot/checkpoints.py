"""Checkpoint storage. On AWS: DynamoDB (survives restarts and lets any Lambda resume any conversation)."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from .config import Settings


def build_checkpointer(s: Settings, local: bool = False):
    if local:
        return InMemorySaver()
    from langgraph_checkpoint_aws import DynamoDBSaver

    return DynamoDBSaver(table_name=s.checkpoint_table, region_name=s.region,
                         endpoint_url=s.dynamodb_endpoint or None, ttl_seconds=s.checkpoint_ttl_days * 86400)


def create_checkpoint_table(dynamodb, name: str):
    """Schema expected by langgraph-checkpoint-aws (CDK creates the real table)."""
    return dynamodb.create_table(
        TableName=name, BillingMode="PAY_PER_REQUEST",
        KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}, {"AttributeName": "SK", "KeyType": "RANGE"}],
        AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"},
                              {"AttributeName": "SK", "AttributeType": "S"}],
    )
