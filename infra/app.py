#!/usr/bin/env python3
"""CDK entry point.

    cd infra
    cdk deploy -c kmcp_api_url=https://<id>.execute-api.us-west-2.amazonaws.com -c alert_email=you@example.com
    cdk deploy ... -c fargate=true     # optional demo: ECS Fargate + ALB (costs money while it exists)
"""

import os

import aws_cdk as cdk
from supportpilot_stack import SupportPilotStack

app = cdk.App()
ctx = app.node.try_get_context
SupportPilotStack(
    app, "SupportPilotStack",
    alert_email=ctx("alert_email"), kmcp_api_url=ctx("kmcp_api_url"),
    fargate=str(ctx("fargate")).lower() == "true",
    llm_model_id=ctx("llm_model_id") or "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
                        region=os.environ.get("CDK_DEFAULT_REGION", "us-west-2")),
)
app.synth()
