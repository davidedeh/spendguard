# Developing and running locally

You need Docker, uv and make. Nothing in this section calls a model or AWS, apart from the demo with real models at its end.

```sh
make install        # uv sync --frozen
make up             # shared compose stack: Postgres with pgvector, ElasticMQ
cp .env.example .env
make migrate        # creates this branch's database and applies migrations
make check          # ruff, mypy --strict, all tests
```

The tests drive the whole flow (signed webhooks, the graph, the worker, approver decisions, the mock ERP and precedents) against the compose Postgres, the in-memory queue or ElasticMQ, and fake models.

To serve the API and call it by hand:

```sh
make seed                           # load the Halden seed data
make issue-token APPROVER=EMP-001   # prints a bearer token once
make run-api                        # serves on http://localhost:8000; leave it running
```

Then, from another terminal:

```sh
make smoke URL=http://localhost:8000    # health is 200, an unsigned webhook is 401
read -rs APPROVER_TOKEN && export APPROVER_TOKEN   # paste the token; it stays out of history and process lists
curl -H @- "http://localhost:8000/submissions?status=pending_review" <<<"Authorization: Bearer $APPROVER_TOKEN"
```

A submission posted to the local API is queued but not advanced until a worker consumes the local queue; deployed, the worker runs as a Lambda fed by SQS.

## Demo with real models

`make demo` posts three signed sample submissions taken from the frozen eval dataset (a clean small claim, a claim whose purpose is unclear, and a claim with injected instructions in its business purpose), waits for each to be processed, lists the pending reviews and prints one finding with its clause and quotes. The demo itself only talks to the local API. The worker and the policy index call the models configured in `.env`, so set their credentials there first, and leave `SSM_PARAMETER_PREFIX` empty: the demo refuses to start if it is set.

```sh
make seed                           # the samples refer to the Halden seed data
make index-policy                   # embeds changed policy clauses with the configured embedding model
make issue-token APPROVER=EMP-001   # prints a bearer token once
make run-api                        # terminal 1; needs WEBHOOK_SIGNING_KEY and DECISION_EVENT_SIGNING_KEY in .env
make run-worker                     # terminal 2; advances submissions with the configured models
read -rs APPROVER_TOKEN && export APPROVER_TOKEN   # terminal 3; paste the token
make demo                           # terminal 3; URL=<base URL> if the API is not on http://localhost:8000
```

`make demo DECIDE=1` goes on to decide the two claims waiting for review as that approver: it dismisses the unclear-purpose finding with a reason and approves that claim, and rejects the claim with injected instructions with a reason. It then shows each decision executed, the one record the mock ERP received for each, the precedent written from the dismissal, and the 409 a second decision on the same claim gets. It reads the mock ERP records and the precedent from this branch's local database.

`make record-demo` regenerates the video. It runs its own copy of the stack on a separate branch name and port (8020 by default), runs `python -m spendguard.demo_presentation` (the same flow as `make demo DECIDE=1`, printed for the recording) and keeps each line with the time it appeared, marks only that run's traces public in Langfuse, takes stills of the injection claim's trace with Playwright, reads each review's real duration from Langfuse, and renders the edit with [Remotion](https://www.remotion.dev) from `video/`. `make render-demo` re-renders from the last take without calling any model. When `recordings/narration/` holds the narration clips (`beat-1.m4a` to `beat-7.m4a`, with beat 4 as `beat-4a` and `beat-4b`), it also cleans them (silence trimmed, light noise reduction, loudness normalised to -16 LUFS), transcribes them locally with whisper.cpp to find each cue word, and renders a narrated cut, `recordings/spendguard-demo-narrated.mp4`, whose captions and highlights follow the voice. That needs `brew install whisper-cpp` and its `ggml-base.en.bin` model in `~/.cache/whisper-cpp/`. It needs Node, `jq` and Playwright's Chromium (`uv run --with playwright==1.58.0 python -m playwright install chromium`), calls the configured models and sends traces to Langfuse. Logs and both cuts go to the ignored `recordings/` directory and the take to `video/public/take/`; the README's copy is uploaded to GitHub by hand.

The samples keep their eval case IDs as claim IDs, so running the demo again returns the same submissions rather than creating new ones. A message the worker fails on, or cannot read, stays queued and is retried, and after three receives it moves to the dead-letter queue. `make purge-queue` empties this branch's local queue and its dead-letter queue; it refuses to run unless `SQS_ENDPOINT_URL` points at ElasticMQ. `make reset-local` drops and rebuilds this branch's local database, seeds it and purges its queue, for a clean rerun; it refuses when `DATABASE_URL` is set, and afterwards the policy needs `make index-policy` and approvers need new tokens.

The policy index records which embedding model produced each clause's vectors, and `make index-policy` re-embeds any clause whose text or embedding model has changed.

One compose stack serves every worktree. The database and queue names derive from the sanitised git branch name, so branches share services without sharing state.

To tear down, stop `make run-api` with Ctrl-C, then:

```sh
make down                     # stop the compose stack; the database volume is kept
docker compose down --volumes # stop it and delete every branch's data
```

`make help` lists every target.

## Tests

- `make test` runs every test, including database and queue tests. Database and queue fixtures raise when compose is down; they never skip.
- A run with any skipped test fails.
- Tests build settings with `tests/settings_factory.py`. No test reads `.env` or needs credentials, and no test calls a real model or AWS.

CI runs the same checks on every pull request and on main against Postgres and ElasticMQ service containers, validates the SAM template with `sam validate` and cfn-lint, and builds the wheel and the Docker image.
