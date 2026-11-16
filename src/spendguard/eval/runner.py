from pathlib import Path

from spendguard.eval.cases import EvalCase, SlicePlan, read_case_directory
from spendguard.eval.freeze import read_locked_hash
from spendguard.eval.validator import validate_dataset
from spendguard.policy.loader import Policy
from spendguard.seed.schema import SeedData


class DatasetRefusedError(Exception):
    pass


def frozen_cases(
    case_directory: Path, plan: SlicePlan, policy: Policy, seed_data: SeedData, lock_path: Path
) -> tuple[EvalCase, ...]:
    if not lock_path.is_file():
        raise DatasetRefusedError(f'There is no lock file at {lock_path}: run make freeze first')
    cases = read_case_directory(case_directory)
    report = validate_dataset(cases, policy, seed_data, plan)
    locked_hash = read_locked_hash(lock_path)
    if report.dataset_hash != locked_hash:
        raise DatasetRefusedError(
            f'{case_directory} hashes to {report.dataset_hash}, which does not match the lock'
            f' {locked_hash} in {lock_path}'
        )
    return tuple(EvalCase.model_validate(content) for content in cases.values())
