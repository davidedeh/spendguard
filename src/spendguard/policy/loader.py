import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, get_args

import jinja2
import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

FindingCode = Literal[
    'OVER_CATEGORY_LIMIT',
    'MISSING_RECEIPT',
    'DUPLICATE_SUBMISSION',
    'SPLIT_PURCHASE',
    'NON_APPROVED_SUPPLIER',
    'MISSING_PO',
    'BUDGET_EXCEEDED',
    'CURRENCY_ANOMALY',
    'WEEKEND_SPEND_UNJUSTIFIED',
    'PURPOSE_UNCLEAR',
    'RECEIPT_MISMATCH',
    'ALCOHOL_NOT_PERMITTED',
]
FINDING_CODES: tuple[FindingCode, ...] = get_args(FindingCode)
POLICY_PATH = Path(__file__).parent / 'policy.yaml'
Severity = Literal['low', 'medium', 'high']
CheckKind = Literal['rule', 'judgement']
PrecedentScope = Literal['employee', 'supplier']
SubmissionType = Literal['expense_claim', 'supplier_invoice']

type ParameterValue = int | Decimal | str | list[ParameterValue] | dict[str, ParameterValue]


class PolicyError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class _ClauseSource(_StrictModel):
    id: str
    heading: str | None = None
    text: str


class _SectionSource(_StrictModel):
    heading: str
    clauses: tuple[_ClauseSource, ...]


class _FindingCodeSource(_StrictModel):
    severity: Severity
    clause: str
    check: CheckKind
    precedent_eligible: bool
    precedent_scope: PrecedentScope | None = None


class _ExpenseCategorySource(_StrictModel):
    label: str
    unit: str
    limits: dict[str, Decimal]


_CATEGORY_LIMITS_ADAPTER = TypeAdapter(dict[str, _ExpenseCategorySource])


class _ApplicabilitySource(_StrictModel):
    expense_claim: tuple[FindingCode, ...]
    supplier_invoice: tuple[FindingCode, ...]


class _PolicySource(_StrictModel):
    title: str
    version: str
    parameters: dict[str, ParameterValue]
    sections: tuple[_SectionSource, ...]
    finding_codes: dict[FindingCode, _FindingCodeSource]
    applicability: _ApplicabilitySource


@dataclass(frozen=True)
class Clause:
    id: str
    reference: str
    heading_path: tuple[str, ...]
    text: str
    hash: str


@dataclass(frozen=True)
class FindingCodeDefinition:
    code: FindingCode
    severity: Severity
    check: CheckKind
    precedent_scope: PrecedentScope | None
    clause: Clause


@dataclass(frozen=True)
class ExpenseCategory:
    key: str
    label: str
    unit: str
    limits: Mapping[str, Decimal]


@dataclass(frozen=True)
class Policy:
    title: str
    version: str
    parameters: Mapping[str, ParameterValue]
    clauses: tuple[Clause, ...]
    finding_codes: Mapping[FindingCode, FindingCodeDefinition]
    applicability: Mapping[SubmissionType, tuple[FindingCode, ...]]
    expense_categories: Mapping[str, ExpenseCategory]
    rendered_text: str


_WHITESPACE_PATTERN = re.compile(r'\s+')
_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)
_DOCUMENT_TEMPLATE = _TEMPLATE_ENVIRONMENT.from_string(
    '# {{ title }}\n'
    '\n'
    'Version {{ version }}\n'
    '{% for section_heading, clauses in sections %}'
    '\n## {{ loop.index }}. {{ section_heading }}\n'
    '{% for clause in clauses %}'
    '\n**{{ clause.reference }}'
    '{% if clause.heading_path | length > 1 %} {{ clause.heading_path[-1] }}.{% endif %}** '
    '{{ clause.text }}\n'
    '{% endfor %}'
    '{% endfor %}'
)


def load_policy(path: Path) -> Policy:
    try:
        source = _PolicySource.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as error:
        raise PolicyError(f'{path} is not a valid policy:\n{error}') from error
    sections = [
        (
            section_source.heading,
            tuple(
                _render_clause(clause_source, section_source, section_number, clause_number, source)
                for clause_number, clause_source in enumerate(section_source.clauses, start=1)
            ),
        )
        for section_number, section_source in enumerate(source.sections, start=1)
    ]
    clauses = tuple(clause for _, section_clauses in sections for clause in section_clauses)
    missing_codes = [code for code in FINDING_CODES if code not in source.finding_codes]
    if missing_codes:
        raise PolicyError(f'The policy does not define finding codes {missing_codes}')
    clause_id_counts = Counter(clause.id for clause in clauses)
    duplicate_ids = sorted(clause_id for clause_id, count in clause_id_counts.items() if count > 1)
    if duplicate_ids:
        raise PolicyError(f'Clause ids must be unique: {duplicate_ids}')
    clauses_by_id = {clause.id: clause for clause in clauses}
    for code, code_source in source.finding_codes.items():
        if code_source.clause not in clauses_by_id:
            raise PolicyError(
                f'{code} rests on clause {code_source.clause!r}, which does not exist'
            )
        if code_source.precedent_eligible and code_source.precedent_scope is None:
            raise PolicyError(f'{code} is precedent eligible but has no precedent_scope')
        if not code_source.precedent_eligible and code_source.precedent_scope is not None:
            raise PolicyError(f'{code} is instance only but has a precedent_scope')
    finding_codes = {
        code: FindingCodeDefinition(
            code=code,
            severity=code_source.severity,
            check=code_source.check,
            precedent_scope=code_source.precedent_scope,
            clause=clauses_by_id[code_source.clause],
        )
        for code, code_source in source.finding_codes.items()
    }
    return Policy(
        title=source.title,
        version=source.version,
        parameters=source.parameters,
        clauses=clauses,
        finding_codes=finding_codes,
        applicability={
            'expense_claim': source.applicability.expense_claim,
            'supplier_invoice': source.applicability.supplier_invoice,
        },
        expense_categories=_expense_categories(path, source),
        rendered_text=_DOCUMENT_TEMPLATE.render(
            title=source.title, version=source.version, sections=sections
        ),
    )


def _expense_categories(path: Path, source: _PolicySource) -> dict[str, ExpenseCategory]:
    if 'category_limits' not in source.parameters:
        raise PolicyError(f'{path} has no category_limits parameter')
    try:
        category_sources = _CATEGORY_LIMITS_ADAPTER.validate_python(
            source.parameters['category_limits']
        )
    except ValidationError as error:
        raise PolicyError(f'{path} has invalid category_limits:\n{error}') from error
    return {
        key: ExpenseCategory(
            key=key,
            label=category_source.label,
            unit=category_source.unit,
            limits=category_source.limits,
        )
        for key, category_source in category_sources.items()
    }


def _render_clause(
    clause_source: _ClauseSource,
    section_source: _SectionSource,
    section_number: int,
    clause_number: int,
    source: _PolicySource,
) -> Clause:
    try:
        text = _TEMPLATE_ENVIRONMENT.from_string(clause_source.text).render(source.parameters)
    except jinja2.UndefinedError as error:
        raise PolicyError(f'Clause {clause_source.id!r} uses {error.message}') from error
    text = text.strip()
    heading_path: tuple[str, ...] = (section_source.heading,)
    if clause_source.heading is not None:
        heading_path = (*heading_path, clause_source.heading)
    return Clause(
        id=clause_source.id,
        reference=f'§{section_number}.{clause_number}',
        heading_path=heading_path,
        text=text,
        hash=_clause_hash(text),
    )


def _clause_hash(text: str) -> str:
    normalised_text = _WHITESPACE_PATTERN.sub(' ', text).strip()
    return hashlib.sha256(normalised_text.encode()).hexdigest()
