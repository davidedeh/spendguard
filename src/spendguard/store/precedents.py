from spendguard.graph.state import SuppliedPrecedent
from spendguard.policy.loader import FindingCode

TABLE_NAME = 'precedents'


def recent_precedents(subject: str, code: FindingCode) -> list[SuppliedPrecedent]:
    return []
