# Configuration

| Concern | Local and tests | Deployed |
| --- | --- | --- |
| Language and packaging | Python 3.12, uv, `src/` layout | Same, one container image |
| Database | Postgres 16 with pgvector in compose | Supabase through the transaction pooler |
| Queue | ElasticMQ in compose; an in-memory fake in unit tests | SQS FIFO with a dead-letter queue |
| Infrastructure | none | One AWS SAM stack: API, worker, sweeper and preparation Lambdas, SQS, SNS alarm, EventBridge Scheduler, SSM parameters, ECR |
| Chat models | Fake models | Claude, GPT OSS 120B and Nova through Bedrock; Gemini through Google GenAI for the eval only |
| Embeddings | Fake embedder | Titan Text Embeddings V2 through Bedrock |
| Tracing | none | Langfuse Cloud, masked before sending |

Model IDs are configuration, never code: each model's ID is a `CHAT_MODELS__<KEY>__MODEL_ID` setting in `.env.example`, and changing a model is a configuration change. GPT OSS 120B is OpenAI's open-weight model, not frontier GPT.

Settings are read in this order, first match wins: values passed in code, then AWS Systems Manager Parameter Store when `SSM_PARAMETER_PREFIX` is set, then environment variables, then `.env`. Every setting and its local default is listed in [.env.example](../.env.example).
