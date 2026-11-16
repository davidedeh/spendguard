import re

CARD_NUMBER_PATTERN = re.compile(r'(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])')
EMAIL_PATTERN = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
PHONE_NUMBER_PATTERN = re.compile(r'(?<![\w+(-])(?:\+|\(|0)[\d ()-]{8,}\d(?![\w-])')
PHONE_NUMBER_DIGIT_COUNTS = range(10, 16)


def _redact_phone_number(match: re.Match[str]) -> str:
    digit_count = sum(character.isdigit() for character in match.group())
    return '[PHONE]' if digit_count in PHONE_NUMBER_DIGIT_COUNTS else match.group()


def _passes_luhn_check(digits: str) -> bool:
    total = 0
    for position, digit in enumerate(int(character) for character in reversed(digits)):
        doubled = digit * 2 if position % 2 == 1 else digit
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


def _redact_card_number(match: re.Match[str]) -> str:
    digits = re.sub(r'\D', '', match.group())
    return '[CARD]' if _passes_luhn_check(digits) else match.group()


def redact(text: str) -> str:
    text = CARD_NUMBER_PATTERN.sub(_redact_card_number, text)
    text = EMAIL_PATTERN.sub('[EMAIL]', text)
    return PHONE_NUMBER_PATTERN.sub(_redact_phone_number, text)
