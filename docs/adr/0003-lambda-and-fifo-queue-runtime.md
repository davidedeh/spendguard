# Graph runs execute in Lambda workers fed by a FIFO queue

Every graph advance (`start` or `resume`) is a message on an SQS FIFO queue grouped by submission. The message is consumed by a Lambda worker built from the same container image as the API, which itself runs as a Lambda behind a Function URL. Several mechanisms work together:

- **The checkpointer** lets a redelivered message continue from the last completed step.
- **The FIFO grouping** stops two workers advancing the same submission at once.
- **A status check** turns stale or duplicate messages into no-ops.
- **A dead-letter queue** catches messages that keep failing.
- **A scheduled sweeper** re-enqueues submissions that are stuck. It closes the gap when the API stops after recording a decision but before enqueueing it.

This replaces FastAPI background tasks on ECS Fargate. That setup had nothing to resume a crashed run, gave the webhook no stable URL without a load balancer, and that load balancer is billed every hour whether used or not. App Runner was also ruled out because it closed to new customers in April 2026. What we gave up: Lambda's 15-minute limit on a single advance, and per-invocation trace flushing.

## Considered Options

- **Fargate with a load balancer.** Looks more like a long-running service, but it costs money while idle and has to be remembered and scaled to zero.
- **A Postgres-backed queue (`SKIP LOCKED`).** No new infrastructure, but retries and the dead-letter queue would be hand-written.
