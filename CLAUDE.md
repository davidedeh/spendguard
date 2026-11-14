# SpendGuard

A portfolio project: an agent that reviews expense claims and supplier invoices for the fictional Halden Group against a written policy, with a human approver in the loop. Everything is fictional: no real companies, people, data or credentials.

## Where the design lives

- The specs are GitHub issues labelled `ready-for-agent`: Spec 1 Foundations (#1), Spec 2 System (#2), Spec 3 agentkit (#3), each with sub-issue tickets. Read the ticket and its parent spec before starting.
- `GLOSSARY.md`: the domain vocabulary. Use its terms in code, tests and docs (submission, requester, finding, disposition, precedent).
- `docs/adr/`: decisions that constrain the design. Respect them; propose a new ADR rather than working around one.
- `docs/references.md`: public policies used for structure and tone only. Never copy their text.
- `docs/draft-spec.md` is superseded by the specs and ADRs. Do not build from it.
- `AGENTS.md`: the stack, the code layout, the test fixtures and the commands.

## AWS

Implementation agents must not call live AWS APIs or create AWS resources; infrastructure changes go through template.yaml, validated with sam validate and cfn-lint. Live AWS actions happen only at G3, by me.

Tests never call a real model or AWS service: use the fake models, the in-memory queue, and the compose stack.

## Model IDs

Model and inference-profile IDs come from configuration only, never code, so changing a model is a config change.
