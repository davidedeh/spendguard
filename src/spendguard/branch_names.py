import hashlib
import os
import re
import subprocess

POSTGRES_IDENTIFIER_LIMIT = 63
SQS_QUEUE_NAME_LIMIT = 80
FIFO_SUFFIX = '.fifo'
DEAD_LETTER_SUFFIX = '-dead-letter'
DIGEST_LENGTH = 8

disallowed_character_pattern = re.compile(r'[^a-z0-9]+')


def sanitise_branch_name(branch_name: str) -> str:
    return disallowed_character_pattern.sub('_', branch_name.lower()).strip('_')


def _branch_digest(branch_name: str) -> str:
    return hashlib.sha256(branch_name.encode()).hexdigest()[:DIGEST_LENGTH]


def _bounded_name(prefix: str, branch_name: str, separator: str, limit: int) -> str:
    digest = _branch_digest(branch_name)
    room_for_branch = limit - len(prefix) - len(digest) - 2 * len(separator)
    readable_part = sanitise_branch_name(branch_name)[:room_for_branch].strip('_')
    if separator != '_':
        readable_part = readable_part.replace('_', separator)
    return separator.join(part for part in (prefix, readable_part, digest) if part)


def current_branch_name() -> str:
    for variable in ('GITHUB_HEAD_REF', 'GITHUB_REF_NAME'):
        if branch_name := os.environ.get(variable):
            return branch_name
    completed = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def database_name_for_branch(branch_name: str) -> str:
    return _bounded_name('spendguard', branch_name, '_', POSTGRES_IDENTIFIER_LIMIT)


def queue_name_for_branch(branch_name: str) -> str:
    limit = SQS_QUEUE_NAME_LIMIT - len(FIFO_SUFFIX) - len(DEAD_LETTER_SUFFIX)
    return _bounded_name('spendguard', branch_name, '-', limit) + FIFO_SUFFIX


def dead_letter_queue_name_for(queue_name: str) -> str:
    return queue_name.removesuffix(FIFO_SUFFIX) + DEAD_LETTER_SUFFIX + FIFO_SUFFIX
