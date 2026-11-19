"""AWS infrastructure for SupportPilot.

Default (always-on, low cost): Lambda (container) behind an HTTP API, DynamoDB on-demand for checkpoints
and business data, Secrets Manager for keys. Nothing here bills by the hour while idle except secrets.

Optional demo mode (-c fargate=true): the same image on ECS Fargate behind an Application Load Balancer.
A load balancer bills by the hour even when idle (roughly $16+/month), so deploy it for a demo and remove
it with `cdk deploy` without the flag (or `cdk destroy`).
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, Tags, triggers
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE, SERVICE = "SupportPilot", "supportpilot"
EXCLUDE = ["infra", ".git", "evals/results", "**/__pycache__", ".venv", "docs", "tests"]


class SupportPilotStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, alert_email: str | None = None,
                 kmcp_api_url: str | None = None, fargate: bool = False,
                 llm_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        Tags.of(self).add("Project", "SupportPilot")

        # --- State: LangGraph checkpoints + business data (on-demand, no idle cost) -------------------
        checkpoints = dynamodb.Table(
            self, "Checkpoints", partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST, time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY)
        data = dynamodb.Table(
            self, "Data", partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST, removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=False))

        # --- Secrets: generated client/approver keys; KnowledgeMCP key is filled in after deploy ------
        def generated(sid: str, desc: str) -> secretsmanager.Secret:
            return secretsmanager.Secret(self, sid, description=desc, generate_secret_string=(
                secretsmanager.SecretStringGenerator(exclude_punctuation=True, password_length=40)))

        client_key = generated("ClientKey", "SupportPilot chat client API key")
        approver_key = generated("ApproverKey", "SupportPilot human approver API key")
        kmcp_key = secretsmanager.Secret(self, "KnowledgeMcpKey",
                                         description="KnowledgeMCP API key for SupportPilot (team: support)")

        env = {
            "DATA_TABLE": data.table_name, "CHECKPOINT_TABLE": checkpoints.table_name,
            "LLM": "bedrock", "LLM_MODEL_ID": llm_model_id,
            "KNOWLEDGE": "mcp" if kmcp_api_url else "static", "KMCP_API_URL": kmcp_api_url or "",
            "KMCP_API_KEY_SECRET_ARN": kmcp_key.secret_arn,
            "CLIENT_KEY_SECRET_ARN": client_key.secret_arn, "APPROVER_KEY_SECRET_ARN": approver_key.secret_arn,
            "POWERTOOLS_SERVICE_NAME": SERVICE, "POWERTOOLS_METRICS_NAMESPACE": NAMESPACE,
        }
        bedrock = iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=[
            "arn:aws:bedrock:*::foundation-model/*", f"arn:aws:bedrock:*:{self.account}:inference-profile/*"])

        def grant_runtime(grantee: iam.IGrantable, role: iam.IRole | None = None) -> None:
            for t in (checkpoints, data):
                t.grant_read_write_data(grantee)
            for sec in (client_key, approver_key, kmcp_key):
                sec.grant_read(grantee)
            (role or grantee.grant_principal).add_to_principal_policy(bedrock)

        def image(cmd: str) -> lambda_.DockerImageCode:
            return lambda_.DockerImageCode.from_image_asset(str(REPO_ROOT), cmd=[cmd], exclude=EXCLUDE,
                                                            platform=ecr_assets.Platform.LINUX_AMD64)

        def function(fid: str, cmd: str, timeout: Duration) -> lambda_.Function:
            fn = lambda_.DockerImageFunction(
                self, fid, code=image(cmd), architecture=lambda_.Architecture.X86_64, memory_size=1024,
                timeout=timeout, environment=env, tracing=lambda_.Tracing.ACTIVE,
                log_group=logs.LogGroup(self, f"{fid}Logs", retention=logs.RetentionDays.TWO_WEEKS,
                                        removal_policy=RemovalPolicy.DESTROY))
            grant_runtime(fn)
            return fn

        api_fn = function("ApiFunction", "supportpilot.api.handler", Duration.seconds(29))
        seed_fn = function("SeedFunction", "supportpilot.api.seed_handler", Duration.minutes(2))
        triggers.Trigger(self, "SeedDemoData", handler=seed_fn, execute_after=[data],
                         execute_on_handler_change=True)

        http_api = apigw.HttpApi(self, "HttpApi", api_name="supportpilot", create_default_stage=False,
                                 default_integration=integrations.HttpLambdaIntegration("ApiIntegration", api_fn))
        stage = apigw.HttpStage(self, "DefaultStage", http_api=http_api, stage_name="$default", auto_deploy=True,
                                throttle=apigw.ThrottleSettings(rate_limit=5, burst_limit=10))

        # --- Optional: Fargate behind an ALB (demo only) ------------------------------------------------
        if fargate:
            vpc = ec2.Vpc(self, "Vpc", max_azs=2, nat_gateways=0, subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)])
            svc = ecs_patterns.ApplicationLoadBalancedFargateService(
                self, "Fargate", vpc=vpc, cpu=512, memory_limit_mib=1024, desired_count=1,
                public_load_balancer=True, assign_public_ip=True,
                task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                    image=ecs.ContainerImage.from_asset(str(REPO_ROOT), exclude=EXCLUDE,
                                                        platform=ecr_assets.Platform.LINUX_AMD64),
                    entry_point=["python3", "-m", "uvicorn"],
                    command=["supportpilot.api:app", "--host", "0.0.0.0", "--port", "8000"],
                    container_port=8000, environment=env,
                    log_driver=ecs.LogDrivers.aws_logs(stream_prefix="supportpilot",
                                                       log_retention=logs.RetentionDays.ONE_WEEK)))
            svc.target_group.configure_health_check(path="/health")
            grant_runtime(svc.task_definition.task_role, svc.task_definition.task_role)
            CfnOutput(self, "FargateUrl", value=f"http://{svc.load_balancer.load_balancer_dns_name}")

        # --- Observability ----------------------------------------------------------------------------
        def m(name: str, stat: str = "Sum") -> cw.Metric:
            return cw.Metric(namespace=NAMESPACE, metric_name=name, dimensions_map={"service": SERVICE},
                             statistic=stat, period=Duration.minutes(5), label=f"{name} ({stat})")

        dash = cw.Dashboard(self, "Dashboard", dashboard_name="SupportPilot")
        dash.add_widgets(
            cw.GraphWidget(title="Turns and failures", left=[m("Turns"), m("RunsFailed"), m("GuardrailBlocks")]),
            cw.GraphWidget(title="Turn latency (ms)", left=[m("TurnLatencyMs", "p50"), m("TurnLatencyMs", "p95")]),
            cw.GraphWidget(title="Refunds", left=[m("RefundsIssued"), m("RefundsDenied"), m("ApprovalsRequested"),
                                                   m("ApprovalsDecided")]),
            cw.GraphWidget(title="Refunded USD", left=[m("RefundUSD")]),
            cw.GraphWidget(title="Model cost (micro-USD)", left=[m("TurnCostMicroUSD")]),
            cw.GraphWidget(title="Node latency p95 (ms)", left=[m(f"Node_{n}_Ms", "p95") for n in
                                                                 ("classify", "knowledge", "refund_prepare",
                                                                  "execute_refund", "respond")]),
            cw.GraphWidget(title="Lambda errors", left=[api_fn.metric_errors()]),
        )

        if alert_email:
            budgets.CfnBudget(
                self, "MonthlyBudget",
                budget=budgets.CfnBudget.BudgetDataProperty(
                    budget_name="supportpilot-monthly", budget_type="COST", time_unit="MONTHLY",
                    budget_limit=budgets.CfnBudget.SpendProperty(amount=10, unit="USD")),
                notifications_with_subscribers=[budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type=kind, comparison_operator="GREATER_THAN", threshold=pct,
                        threshold_type="PERCENTAGE"),
                    subscribers=[budgets.CfnBudget.SubscriberProperty(subscription_type="EMAIL",
                                                                      address=alert_email)])
                    for kind, pct in (("ACTUAL", 80), ("FORECASTED", 100))])

        CfnOutput(self, "ApiUrl", value=stage.url)
        CfnOutput(self, "DataTableName", value=data.table_name)
        CfnOutput(self, "ClientKeySecret", value=client_key.secret_name)
        CfnOutput(self, "ApproverKeySecret", value=approver_key.secret_name)
        CfnOutput(self, "KnowledgeMcpKeySecret", value=kmcp_key.secret_name)
        CfnOutput(self, "DashboardUrl", value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
                                              f"?region={self.region}#dashboards:name=SupportPilot")
