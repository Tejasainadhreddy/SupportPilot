"""Synthesize the stack (no AWS calls, no Docker build) and check the cost and security choices."""

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template
from supportpilot_stack import SupportPilotStack


def synth(**kw) -> Template:
    app = cdk.App()
    return Template.from_stack(SupportPilotStack(
        app, "Test", env=cdk.Environment(account="123456789012", region="us-west-2"), **kw))


def test_default_is_serverless_with_no_idle_compute():
    t = synth()
    t.resource_count_is("AWS::ElasticLoadBalancingV2::LoadBalancer", 0)
    t.resource_count_is("AWS::ECS::Service", 0)
    t.resource_count_is("AWS::EC2::NatGateway", 0)
    t.resource_count_is("AWS::DynamoDB::Table", 2)
    t.has_resource_properties("AWS::DynamoDB::Table", {
        "BillingMode": "PAY_PER_REQUEST", "TimeToLiveSpecification": {"AttributeName": "ttl", "Enabled": True}})


def test_container_lambdas_and_throttled_api():
    t = synth()
    fns = t.find_resources("AWS::Lambda::Function", {"Properties": {"PackageType": "Image"}})
    assert len(fns) == 2
    t.has_resource_properties("AWS::ApiGatewayV2::Stage", {
        "DefaultRouteSettings": {"ThrottlingBurstLimit": 10, "ThrottlingRateLimit": 5}})


def test_knowledge_mode_follows_context():
    def env_of(t):
        fns = t.find_resources("AWS::Lambda::Function", {"Properties": {"PackageType": "Image"}})
        return next(iter(fns.values()))["Properties"]["Environment"]["Variables"]

    assert env_of(synth())["KNOWLEDGE"] == "static"
    assert env_of(synth(kmcp_api_url="https://kmcp.example"))["KNOWLEDGE"] == "mcp"


def test_secrets_are_generated_not_hardcoded():
    synth().resource_count_is("AWS::SecretsManager::Secret", 3)


def test_fargate_demo_mode_adds_alb_without_nat():
    t = synth(fargate=True)
    t.resource_count_is("AWS::ECS::Service", 1)
    t.resource_count_is("AWS::ElasticLoadBalancingV2::LoadBalancer", 1)
    t.resource_count_is("AWS::EC2::NatGateway", 0)
    t.has_resource_properties("AWS::ElasticLoadBalancingV2::TargetGroup", {"HealthCheckPath": "/health"})


def test_budget_and_dashboard():
    synth().resource_count_is("AWS::Budgets::Budget", 0)
    synth(alert_email="me@example.com").has_resource_properties("AWS::Budgets::Budget", {
        "Budget": Match.object_like({"BudgetLimit": {"Amount": 10, "Unit": "USD"}})})
    synth().resource_count_is("AWS::CloudWatch::Dashboard", 1)
