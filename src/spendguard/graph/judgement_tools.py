import json
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool

from spendguard.graph.state import SuppliedPrecedent
from spendguard.policy.loader import FindingCode
from spendguard.privacy.redaction import redact
from spendguard.seed.schema import SeedData
from spendguard.submission import ExpenseClaim, Submission

NO_SUPPLIER = 'This submission has no supplier.'
NO_COST_CENTRE = 'This submission has no cost centre.'
NO_PUBLIC_HOLIDAY = "This submission's date is not a public holiday."


@dataclass(frozen=True)
class SubmissionTools:
    tools: list[BaseTool]
    supplied_precedents: list[SuppliedPrecedent]


def _cost_centre_reference(
    submission: Submission, requester: str, seed_data: SeedData
) -> str | None:
    if isinstance(submission, ExpenseClaim):
        return next(
            (
                employee.cost_centre
                for employee in seed_data.employees
                if employee.reference == requester
            ),
            None,
        )
    if submission.purchase_order is not None:
        return next(
            (
                order.cost_centre
                for order in seed_data.purchase_orders
                if order.reference == submission.purchase_order
            ),
            None,
        )
    return submission.cost_centre


def submission_tools(
    submission: Submission,
    requester: str,
    seed_data: SeedData,
    available_precedents: Sequence[SuppliedPrecedent],
) -> SubmissionTools:
    supplier = None if isinstance(submission, ExpenseClaim) else submission.supplier
    cost_centre = _cost_centre_reference(submission, requester, seed_data)
    supplied_precedents: list[SuppliedPrecedent] = []

    def get_employee() -> str:
        employee = next(
            employee for employee in seed_data.employees if employee.reference == requester
        )
        approval_limit = next(
            (
                approver.approval_limit_gbp
                for approver in seed_data.approvers
                if approver.employee == requester
            ),
            None,
        )
        return json.dumps(
            {
                'reference': employee.reference,
                'grade': employee.grade,
                'cost_centre': employee.cost_centre,
                'manager': employee.manager,
                'approval_limit_gbp': None if approval_limit is None else f'{approval_limit:.2f}',
            }
        )

    def get_supplier() -> str:
        found = next(
            (candidate for candidate in seed_data.suppliers if candidate.reference == supplier),
            None,
        )
        if found is None:
            return json.dumps({'error': NO_SUPPLIER})
        return json.dumps({'reference': found.reference, 'approved': found.approved})

    def get_cost_centre() -> str:
        found = next(
            (centre for centre in seed_data.cost_centres if centre.reference == cost_centre),
            None,
        )
        if found is None:
            return json.dumps({'error': NO_COST_CENTRE})
        return json.dumps(
            {
                'reference': found.reference,
                'owner': found.owner,
                'budget_gbp': f'{found.budget_gbp:.2f}',
                'spend_to_date_gbp': f'{found.spend_to_date_gbp:.2f}',
            }
        )

    def get_public_holiday() -> str:
        found = next(
            (
                holiday
                for holiday in seed_data.public_holidays
                if holiday.holiday_on == submission.dated_on
            ),
            None,
        )
        if found is None:
            return json.dumps({'error': NO_PUBLIC_HOLIDAY})
        return json.dumps(
            {'holiday_on': found.holiday_on.isoformat(), 'description': found.description}
        )

    def get_precedents(code: FindingCode) -> str:
        found: list[SuppliedPrecedent] = [
            precedent | {'reason': redact(precedent['reason'])}
            for precedent in available_precedents
            if precedent['code'] == code
        ]
        supplied_precedents.extend(found)
        return json.dumps(found)

    tools: list[BaseTool] = [
        StructuredTool.from_function(
            get_employee,
            description=(
                "The requester's grade, cost centre, manager and approval limit (null when they"
                ' are not an approver).'
            ),
        ),
        StructuredTool.from_function(
            get_supplier, description="The supplier's status: whether it is on the approved list."
        ),
        StructuredTool.from_function(
            get_cost_centre,
            description='The cost centre charged for this spend: owner, budget and spend to date.',
        ),
        StructuredTool.from_function(
            get_public_holiday,
            description="The Halden Group public holiday on this submission's date, if any.",
        ),
        StructuredTool.from_function(
            get_precedents,
            description=(
                'Precedents for a finding code: earlier approver dismissals of that code for this'
                ' submission. Cite a precedent by its precedent_id when you rely on it.'
            ),
        ),
    ]
    return SubmissionTools(tools=tools, supplied_precedents=supplied_precedents)
