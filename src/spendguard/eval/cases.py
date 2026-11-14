from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, StringConstraints, model_validator

from spendguard.policy.loader import CheckKind, FindingCode
from spendguard.rules.decisions import Recommendation, Route
from spendguard.seed.schema import EmployeeReference
from spendguard.settings import Provider
from spendguard.strict_model import StrictModel
from spendguard.submission import Submission

Split = Literal['test', 'dev']
Slice = Literal['judgement_positive', 'hard_negative', 'rule_coverage', 'clean', 'injection']
Disposition = Literal['upheld', 'dismissed']
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
Count = Annotated[int, Field(ge=0)]
SINGLE_FINDING_SLICES: dict[Slice, CheckKind] = {
    'judgement_positive': 'judgement',
    'rule_coverage': 'rule',
}


class ExpectedFinding(StrictModel):
    id: NonEmptyText
    code: FindingCode
    gold_clauses: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]


class ScriptedDisposition(StrictModel):
    finding_id: NonEmptyText
    disposition: Disposition
    reason: NonEmptyText | None = None


class ScriptedReview(StrictModel):
    approver: EmployeeReference
    dispositions: Annotated[tuple[ScriptedDisposition, ...], Field(min_length=1)]

    @model_validator(mode='after')
    def _dismissals_have_a_reason(self) -> Self:
        unexplained_finding_ids = [
            disposition.finding_id
            for disposition in self.dispositions
            if disposition.disposition == 'dismissed' and disposition.reason is None
        ]
        if unexplained_finding_ids:
            raise ValueError(
                f'dismissing finding {", ".join(unexplained_finding_ids)} needs a reason'
            )
        return self


class DraftedBy(StrictModel):
    provider: Provider
    model_id: NonEmptyText


class EvalCase(StrictModel):
    id: NonEmptyText
    split: Split
    slice: Slice
    submission: Submission
    expected_findings: tuple[ExpectedFinding, ...]
    expected_route: Route
    expected_recommendation: Recommendation
    scripted_review: ScriptedReview | None = None
    drafted_by: DraftedBy


class SliceCounts(StrictModel):
    judgement_positive: Count
    hard_negative: Count
    rule_coverage: Count
    clean: Count
    injection: Count


class SplitPlan(StrictModel):
    slices: SliceCounts
    judgement_positives_per_code: Count
    memory_pairs: Count


class SlicePlan(StrictModel):
    test: SplitPlan
    dev: SplitPlan

    def for_split(self, split: Split) -> SplitPlan:
        return self.test if split == 'test' else self.dev


def load_slice_plan(path: Path) -> SlicePlan:
    return SlicePlan.model_validate(yaml.safe_load(path.read_text()))


def read_case_directory(directory: Path) -> dict[str, object]:
    if not directory.is_dir():
        raise FileNotFoundError(f'no case directory at {directory}')
    return {
        file.name: yaml.safe_load(file.read_text()) for file in sorted(directory.glob('*.yaml'))
    }
