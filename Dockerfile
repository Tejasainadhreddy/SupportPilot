# One image: Lambda (API + seed handlers) and, with an entrypoint override, Fargate / docker compose (uvicorn).
FROM public.ecr.aws/lambda/python:3.12

ARG KNOWLEDGEMCP_REF=main
COPY pyproject.toml README.md /tmp/build/
COPY src /tmp/build/src
COPY knowledge_docs /tmp/build/knowledge_docs
# KnowledgeMCP provides the MCP server that SupportPilot launches as an MCP client.
RUN pip install --no-cache-dir "/tmp/build[local]" \
        "knowledgemcp[mcp] @ https://github.com/Tejasainadhreddy/KnowledgeMCP/archive/${KNOWLEDGEMCP_REF}.zip" \
        --target "${LAMBDA_TASK_ROOT}" \
    && cp -r /tmp/build/knowledge_docs "${LAMBDA_TASK_ROOT}/knowledge_docs" \
    && rm -rf /tmp/build

CMD ["supportpilot.api.handler"]
