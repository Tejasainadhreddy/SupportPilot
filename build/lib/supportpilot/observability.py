"""Structured logs, CloudWatch EMF metrics and X-Ray traces (AWS Lambda Powertools)."""

from __future__ import annotations

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit

SERVICE = "supportpilot"
logger = Logger(service=SERVICE)
metrics = Metrics(namespace="SupportPilot", service=SERVICE)
tracer = Tracer(service=SERVICE)


def add_metric(name: str, value: float, unit=MetricUnit.Count) -> None:
    try:
        metrics.add_metric(name=name, unit=unit, value=value)
    except Exception:
        logger.debug("metric dropped", extra={"metric": name})


__all__ = ["logger", "metrics", "tracer", "add_metric", "MetricUnit"]
