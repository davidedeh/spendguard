import unicodedata

KEPT_CONTROL_CHARACTERS = frozenset('\n\r\t')
REMOVED_CATEGORIES = frozenset({'Cc', 'Cf'})


def _is_hidden(character: str) -> bool:
    return (
        unicodedata.category(character) in REMOVED_CATEGORIES
        and character not in KEPT_CONTROL_CHARACTERS
    )


def clean_field(text: str, maximum_length: int) -> str:
    visible = ''.join(character for character in text if not _is_hidden(character))
    return visible[:maximum_length]
