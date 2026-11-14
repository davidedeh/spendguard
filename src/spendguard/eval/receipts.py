import re
from dataclasses import dataclass
from decimal import Decimal

MONTH_NAME = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?'
DATE_OR_TIME_PATTERN = re.compile(
    r'\d{4}[-\u2011/.]\d{1,2}[-\u2011/.]\d{1,2}'
    r'|\d{1,2}[-\u2011/.]\d{1,2}[-\u2011/.]\d{2,4}'
    rf'|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_NAME},?\s+\d{{4}}'
    rf'|{MONTH_NAME}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}'
    r'|\d{1,2}:\d{2}(?::\d{2})?',
    re.IGNORECASE,
)
TOTAL_LINE_PATTERN = re.compile(
    r'^\s*total\b[^\d\n]*(?P<number>\d[\d,]*(?:\.\d{1,2})?)\s*$', re.IGNORECASE
)
TRAILING_AMOUNT_PATTERN = re.compile(
    r'(?:^|\s)(?P<minus>[-\u2212]?)(?:[A-Z]{3}\s?)?[£$€]?(?P<minus_after_symbol>[-\u2212]?)'
    r'(?P<number>\d[\d,]*(?:\.\d{1,2})?)(?:\s?[A-Z]{3})?\s*$'
)
LETTER_PATTERN = re.compile(r'[^\W\d_]')
LABEL_SEPARATORS = ' \t:;,@*\u2013\u2014-'
NON_ADDITIVE_LABEL_PATTERNS = (
    re.compile(r'\bsub[\s-]?total\b', re.IGNORECASE),
    re.compile(r'^net\b', re.IGNORECASE),
    re.compile(r'^(?:amount\s+)?(?:paid|payment|tendered|due)\b', re.IGNORECASE),
    re.compile(
        r'^(?:card|visa|mastercard|amex|maestro|contactless|cash|change|balance)\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(?:vat|tax)\b.*\binc|\binc\w*\.?\s+(?:vat|tax)\b|\bof which\b', re.IGNORECASE),
)
TIP_LABELS = ('tip', 'gratuity', 'service')
TIP_LABEL_TEXT = f'{", ".join(TIP_LABELS[:-1])} or {TIP_LABELS[-1]}'
TIP_LABEL_PATTERN = re.compile(rf'\b(?:{"|".join(TIP_LABELS)})', re.IGNORECASE)


@dataclass(frozen=True)
class PricedLine:
    label: str
    amount: Decimal


@dataclass(frozen=True)
class Receipt:
    total: Decimal
    priced_lines: tuple[PricedLine, ...]
    payment_slip_lines: tuple[PricedLine, ...]

    @property
    def line_sum(self) -> Decimal:
        return sum((line.amount for line in self.priced_lines), Decimal(0))

    @property
    def removable_amounts(self) -> frozenset[Decimal]:
        sums: set[Decimal] = set()
        for line in self.priced_lines:
            if line.amount > 0:
                sums |= {line.amount} | {amount + line.amount for amount in sums}
        return frozenset(sums)

    @property
    def tips(self) -> tuple[Decimal, ...]:
        return tuple(
            line.amount for line in self.payment_slip_lines if TIP_LABEL_PATTERN.search(line.label)
        )


def read_receipt(receipt_text: str) -> Receipt | None:
    lines = receipt_text.splitlines()
    total_matches = ((index, TOTAL_LINE_PATTERN.match(line)) for index, line in enumerate(lines))
    index, total_match = next(
        ((index, match) for index, match in total_matches if match is not None), (0, None)
    )
    if total_match is None:
        return None
    priced_lines = tuple(
        line
        for line in _priced_lines(lines[:index])
        if not any(pattern.search(line.label) for pattern in NON_ADDITIVE_LABEL_PATTERNS)
    )
    if not priced_lines:
        return None
    return Receipt(
        total=_amount(total_match.group('number')),
        priced_lines=priced_lines,
        payment_slip_lines=tuple(_priced_lines(lines[index + 1 :])),
    )


def _priced_lines(lines: list[str]) -> list[PricedLine]:
    priced: dict[int, PricedLine] = {}
    previous_index: int | None = None
    for index, line in enumerate(lines):
        text = DATE_OR_TIME_PATTERN.sub(' ', line).rstrip()
        match = TRAILING_AMOUNT_PATTERN.search(text)
        if match is not None:
            amount = _amount(match.group('number'))
            if match.group('minus') or match.group('minus_after_symbol'):
                amount = -amount
            label = text[: match.start()].strip(LABEL_SEPARATORS)
            if not LETTER_PATTERN.search(label) and previous_index is not None:
                label = lines[previous_index].strip(LABEL_SEPARATORS)
                priced.pop(previous_index, None)
            priced[index] = PricedLine(label=label, amount=amount)
        if line.strip():
            previous_index = index
    return list(priced.values())


def _amount(number: str) -> Decimal:
    return Decimal(number.replace(',', ''))
