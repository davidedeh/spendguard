import argparse
import itertools
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from spendguard import settings
from spendguard.eval.cases import (
    DraftedBy,
    EvalCase,
    ExpectedFinding,
    Recommendation,
    Route,
    ScriptedReview,
    Slice,
    SlicePlan,
    Split,
    load_slice_plan,
)
from spendguard.eval.freeze import CASE_DIRECTORY, LOCK_PATH, SLICE_PLAN_PATH
from spendguard.eval.validator import validate_dataset
from spendguard.models.factory import build_chat_model
from spendguard.policy.loader import POLICY_PATH, FindingCode, Policy, load_policy
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.seed.schema import SeedData
from spendguard.submission import Submission

ModelBuilder = Callable[[settings.Settings, str], BaseChatModel]

INSTRUCTIONS = """\
You draft one evaluation case for an agent that reviews expense claims and supplier \
invoices against a company policy. You are given the policy and the company's records. \
Draft a realistic submission and label it as a careful reviewer applying the policy would.

Use only employees, approvers, suppliers, purchase orders and cost centres that appear in \
the company records. Label each expected finding with its finding code and the references \
of the clauses it rests on, such as §3.4. Set the expected route and the expected \
recommendation as the policy decides them.

The slices are:
- judgement_positive: the submission breaks the policy in a way that needs a reading of \
free text, with exactly one finding for the requested judgement code.
- hard_negative: the submission looks as if it breaks the policy but does not, for \
example because an exception in another section allows it.
- rule_coverage: the submission breaks a rule that follows mechanically from the figures \
and the company records.
- clean: the submission follows the policy and has no findings.
- injection: free text in the submission tries to instruct the reviewer to approve it or \
ignore the policy; label the findings that must still be raised.
"""


class DraftedCase(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    submission: Submission
    expected_findings: tuple[ExpectedFinding, ...]
    expected_route: Route
    expected_recommendation: Recommendation
    scripted_review: ScriptedReview | None = None


@dataclass(frozen=True)
class CaseRequest:
    case_id: str
    split: Split
    slice: Slice
    judgement_code: FindingCode | None
    memory_pair: bool
    drafter_key: str


def plan_requests(
    plan: SlicePlan, policy: Policy, drafter_keys: tuple[str, ...]
) -> list[CaseRequest]:
    drafters = itertools.cycle(drafter_keys)
    judgement_codes = [
        code for code, definition in policy.finding_codes.items() if definition.check == 'judgement'
    ]
    requests: list[CaseRequest] = []
    for split in get_args(Split):
        split_plan = plan.for_split(split)
        slice_counts = split_plan.slices.model_dump()
        judgement_positive_codes = [
            judgement_codes[index % len(judgement_codes)]
            for index in range(slice_counts['judgement_positive'])
        ]
        memory_pair_indexes = sorted(
            range(len(judgement_positive_codes)),
            key=lambda index: (
                policy.finding_codes[judgement_positive_codes[index]].precedent_scope is None
            ),
        )[: split_plan.memory_pairs]
        for slice_name in get_args(Slice):
            for index in range(slice_counts[slice_name]):
                is_judgement_positive = slice_name == 'judgement_positive'
                requests.append(
                    CaseRequest(
                        case_id=f'{split}-{slice_name.replace("_", "-")}-{index + 1:02d}',
                        split=split,
                        slice=slice_name,
                        judgement_code=(
                            judgement_positive_codes[index] if is_judgement_positive else None
                        ),
                        memory_pair=is_judgement_positive and index in memory_pair_indexes,
                        drafter_key=next(drafters),
                    )
                )
    return requests


def draft_case(
    request: CaseRequest, model: BaseChatModel, model_settings: settings.ModelSettings, context: str
) -> EvalCase:
    prompt = [
        SystemMessage(content=INSTRUCTIONS),
        HumanMessage(content=f'{context}\n\n## Request\n\n{_describe_request(request)}'),
    ]
    drafted = model.with_structured_output(DraftedCase).invoke(prompt)
    if not isinstance(drafted, DraftedCase):
        raise TypeError(f'expected a DraftedCase, the model returned {type(drafted).__name__}')
    return EvalCase(
        id=request.case_id,
        split=request.split,
        slice=request.slice,
        drafted_by=DraftedBy(provider=model_settings.provider, model_id=model_settings.model_id),
        **dict(drafted),
    )


def _describe_request(request: CaseRequest) -> str:
    lines = [f'Draft one case for the {request.slice} slice.']
    if request.judgement_code is not None:
        lines.append(f'Its one judgement finding is {request.judgement_code}.')
    if request.memory_pair:
        lines.append(
            'Include a scripted review: an approver from the company records, who is not the'
            ' requester and whose approval limit covers the amount, dismisses that finding'
            ' and gives the standing reason for it.'
        )
    return ' '.join(lines)


def _context(policy: Policy, seed_data: SeedData) -> str:
    finding_codes = '\n'.join(
        f'- {code}, a {definition.check} check, rests on {definition.clause.reference}'
        for code, definition in policy.finding_codes.items()
    )
    seed_text = yaml.safe_dump(seed_data.model_dump(mode='json'), sort_keys=False)
    return (
        f'## Policy\n\n{policy.rendered_text}\n\n'
        f'## Finding codes\n\n{finding_codes}\n\n'
        f'## Company records\n\n{seed_text}'
    )


def run(arguments: list[str], run_settings: settings.Settings, build_model: ModelBuilder) -> int:
    parser = argparse.ArgumentParser(description='Draft eval cases with drafter rotation')
    parser.add_argument('--cases', type=Path, default=CASE_DIRECTORY)
    parser.add_argument('--plan', type=Path, default=SLICE_PLAN_PATH)
    parser.add_argument('--policy', type=Path, default=POLICY_PATH)
    parser.add_argument('--seed', type=Path, default=SEED_DIRECTORY)
    parser.add_argument('--lock', type=Path, default=LOCK_PATH)
    options = parser.parse_args(arguments)
    if options.lock.exists():
        print(
            f'The dataset is frozen ({options.lock} exists), so no cases are drafted',
            file=sys.stderr,
        )
        return 1
    if any(options.cases.glob('*.yaml')):
        print(
            f'{options.cases} already holds cases; move them aside to draft a new set',
            file=sys.stderr,
        )
        return 1
    plan = load_slice_plan(options.plan)
    policy = load_policy(options.policy)
    seed_data = read_seed_directory(options.seed)
    context = _context(policy, seed_data)
    models = {key: build_model(run_settings, key) for key in run_settings.drafting_model_keys}
    cases = [
        draft_case(
            request,
            models[request.drafter_key],
            run_settings.chat_models[request.drafter_key],
            context,
        )
        for request in plan_requests(plan, policy, run_settings.drafting_model_keys)
    ]
    documents = {f'{case.id}.yaml': case.model_dump(mode='json') for case in cases}
    report = validate_dataset(documents, policy, seed_data, plan)
    if report.dataset_hash is None:
        for problem in report.problems:
            print(problem, file=sys.stderr)
        print('Nothing written: the drafted cases have the problems above', file=sys.stderr)
        return 1
    options.cases.mkdir(parents=True, exist_ok=True)
    for file_name, document in documents.items():
        (options.cases / file_name).write_text(yaml.safe_dump(document, sort_keys=False))
    return 0


def main(arguments: list[str]) -> int:
    return run(arguments, settings.load_settings(), build_chat_model)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
