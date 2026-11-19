# Evaluation

## Method

`make eval` runs every configured row against the frozen `eval-v2` dataset in [evals/cases/](../evals/cases/) and refuses to report if the dataset no longer hashes to [evals/dataset.lock.json](../evals/dataset.lock.json). Only the test split is reported; the dev split exists so prompt work never touches the test set. The eval calls real models, so it runs from a laptop. `.github/workflows/eval.yml` can also run it when dispatched by hand, but it needs a role that can invoke Bedrock models in the repository variable `AWS_EVAL_ROLE_ARN`, which this repository does not create, and it passes no Langfuse keys, so a CI run makes no Langfuse dataset runs. `make eval` refuses to start when `DATABASE_URL` is set, because it deletes every precedent in the database it runs against, or when any row's model has no price in `MODEL_PRICES` or no rate in `EVAL_REQUESTS_PER_MINUTE`. Each model's calls are paced to that many requests a minute, shared by every row that uses the model, so a run stays under the account's Bedrock requests-per-minute quota.

The rows are set by `EVAL_ROWS` in `.env.example`: Claude Haiku (the primary), Claude Sonnet, GPT OSS 120B (OpenAI's open-weight model), Nova Pro (AWS-native), a Gemini Pro-class model through Google's own API, and Claude Haiku given the whole policy instead of retrieved clauses, which measures what retrieval contributes. Each row runs every case once through the real graph, at temperature 0, with memory cleared at the start of the row. The first case of each memory pair runs before its follow-up, and the pair's scripted dispositions are applied as an approver's decision.

Per row:

- **Judgement precision and recall**, pooled over the four judgement codes, with 95% Wilson intervals and per-code counts. Injection cases are left out of the pool and scored only as injection survival.
- **Route and recommendation accuracy**: what the approver experiences.
- **Verify drop rate**: the share of model findings verification removed, so a model that paraphrases instead of quoting is not mistaken for one with poor judgement.
- **Injection survival**: the share of expected findings in injection cases still found despite the injected text.
- **Retrieval recall (top k per query)**: the share of gold clauses for expected judgement findings that retrieval returned, where each of the planner's queries returns its top k clauses. It does not apply to the whole-policy row.
- **p50 latency and cost per submission**: cost is token counts times the prices in `MODEL_PRICES`, each recorded with the date it was priced.

Rule-check precision and recall are reported once, not per row, and only as a consistency check: the rule labels were computed by the same rule functions the graph runs, so agreement shows the checks are wired into the graph correctly, not that they are accurate.

Each memory follow-up gets its pair's scripted reason as a precedent before it runs, in every row, built against the current clause text; eval rows record no precedents of their own, so a row that missed the paired finding still sees the precedent. Some follow-ups are controls, whose precedent does not apply: their right answer is to raise the labelled code. A memory on versus off comparison on the primary model reports, for follow-ups whose precedent applies, how many raised the labelled code and how many raised any judgement code, side by side, and how many were cleared with the pair's precedent, forced to review, failed or silently cleared; for controls, how many raised the labelled code, were cleared by the precedent or failed; with memory on, how often the recommendation matches the label, which assumes the precedent, so memory off is not scored on it; and every finding a follow-up raised on another code. With memory off no precedent is available, so raising the labelled code is the right answer on every follow-up. A follow-up whose check got no verdict, or whose flag or clearance verify dropped, is counted as forced review rather than as a silent clearance. It is in the dated results file, not the table below.

Each run writes a results file in `evals/results/`, named with the date and the first 8 characters of the run's identifier (`eval-v2-<date>-<run>.md`), with the exact parameters, prices and per-code counts, and replaces the table below, which names the exact model IDs measured. `make eval` never overwrites a results file, so a second run on the same day keeps the first. As each row finishes, its per-case results, token totals and run identifier are saved as JSON in a folder named like the results file, so a run that stops part-way keeps what it has paid for. A case that raises is recorded as failed and the row carries on: its expected judgement findings count as misses, its route and recommendation as wrong, and it adds no findings, latency or cost. All model clients are built before the first row, so a configuration error stops the run before any spend. `make eval CASES_PER_ROW=2` is a pre-flight: each row runs whole rounds of one case from every slice (a memory follow-up brings its pair's first case) until it has at least that many cases, so every metric gets data, and it writes a results file marked as a pre-flight, leaving this README alone.

`make eval-rerun RUN=<results file name without .md>` reruns the cases of a finished full run that failed on the provider's side, such as Bedrock being briefly unavailable, and never cases the model itself failed. Before any model call it rebuilds the results file from the row files and the current settings, and refuses unless the rebuild matches the file exactly, so a merge cannot pick up changed prices or parameters. Each row's cases run again with their seeded precedents, under a new identifier and as their own Langfuse dataset run. Their results replace the failed ones in the row files, the results file and the table below, saved as each row finishes, so an interrupted rerun keeps the rows it finished. The results file lists each rerun with the error it replaced and the request rate it ran at, and `reruns.json` in the run's folder keeps the history.

With both Langfuse keys set, `make eval` also uploads the reported cases to a Langfuse dataset named `eval-v2-<first 8 characters of the dataset hash>` and records each row as one dataset run, named with the date, the row's label and the run's identifier, with every case traced. A pre-flight's dataset runs also carry "pre-flight" in their names. The memory comparison is not uploaded.

## Case provenance

The cases were drafted by models, under constraints that keep labels honest:

- **Code-generated briefs.** For each case, code generated a brief from the slice plan ([evals/slice_plan.yaml](../evals/slice_plan.yaml)), the rendered policy and the seed data: the slice, submission type, intended findings, subject, dates and which history rows the drafter could see. Drafters never saw the implementation.
- **Rotated across three model families on Bedrock.** Claude Sonnet, GPT OSS 120B and Nova Pro were each assigned about a third of every slice. When an assigned drafter could not produce a valid case, another drafted it, and each case's `drafted_by` records the provider and model that actually wrote it.
- **Labels computed by code where possible.** Drafters wrote the submission, its judgement findings and any scripted review. Rule findings, route and recommendation were computed by the rule, route and recommendation functions, and a draft whose rule findings differed from its brief, or that raised a judgement finding outside it, was rejected and redrafted.
- **Reviewed before the freeze.** The project owner reviewed every case's judgement labels and realism, then froze the dataset in the commit tagged `eval-v1`.
- **Revised as eval-v2 after blind audits.** Audit passes that saw no labels or results led to eval-v2's policy, label and case changes, and two further blind passes checked every v2 case before it was frozen. Four cases were written during that audit, and their `drafted_by` names the model that wrote them. The changes and the reason for each are in [evals/CHANGELOG.md](../evals/CHANGELOG.md).

Possible own-family bias: the Claude, GPT OSS and Nova Pro rows are evaluated partly on cases their own family drafted, which may favour them. Gemini drafted no cases. `drafted_by` lets anyone split the results by drafter.

## Reading the results

GPT OSS 120B is OpenAI's open-weight model, not frontier GPT.

Overlapping 95% intervals are never claimed as a win.

Production default: Haiku unless the eval shows it worse. With 20 positives the eval can only detect large gaps, so this is a cost-led default, not a demonstrated equivalence.
