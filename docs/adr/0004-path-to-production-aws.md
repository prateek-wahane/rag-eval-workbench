# ADR 0004: Path to production on AWS

**Status:** Proposed

| Concern | Workbench (today) | Production on AWS |
|---|---|---|
| Ingestion | Markdown files in `data/corpus` | S3 bucket + event-driven Lambda; chunk/embed on change; versioned corpus |
| Embeddings | fastembed on CPU | Same model in a container (ECS/Lambda), or a managed embedding endpoint |
| Vector store | FAISS flat file | OpenSearch Serverless (vector + BM25 hybrid) or Aurora PostgreSQL + pgvector |
| LLM | Claude API / local Ollama | Claude via Amazon Bedrock or the Claude API; refusal fallback; per-route effort |
| API | CLI | FastAPI on ECS Fargate/EKS behind API Gateway; auth via Cognito/IAM |
| Security | Local only | Secrets Manager, KMS, private VPC endpoints, least-privilege IAM, PII redaction before indexing, prompt-injection tests in CI |
| Observability | Printed latency/tokens/cost | OpenTelemetry traces (retrieve, generate, judge spans), CloudWatch dashboards, cost per answer |
| Quality gate | `ragwb eval-retrieval` / `eval-generation` | Same eval in CI on every prompt/model/chunking change; block merge on regression |

## Key risks
- **Stale content:** policy changes must reindex within minutes. Track the corpus version in every answer's trace.
- **Prompt injection via documents:** treat retrieved text as data, and test with poisoned documents.
- **Cost drift:** budget alarms per environment, and track $/1k answers as an SLO next to latency.
