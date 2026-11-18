import hashlib
import hmac
import re

SIGNATURE_HEADER = 'X-Signature'
signature_header_pattern = re.compile(r't=(?P<timestamp>\d{1,12}),v1=(?P<digest>[0-9a-f]{64})')


def _digest(key: bytes, timestamp: int, raw_body: bytes) -> str:
    signed_payload = f'{timestamp}.'.encode() + raw_body
    return hmac.new(key, signed_payload, hashlib.sha256).hexdigest()


def sign(key: bytes, timestamp: int, raw_body: bytes) -> str:
    return f't={timestamp},v1={_digest(key, timestamp, raw_body)}'


def verify_signature(
    key: bytes, header: str | None, raw_body: bytes, now: int, tolerance_seconds: int
) -> bool:
    if header is None:
        return False
    match = signature_header_pattern.fullmatch(header)
    if match is None:
        return False
    timestamp = int(match['timestamp'])
    if abs(now - timestamp) > tolerance_seconds:
        return False
    return hmac.compare_digest(match['digest'], _digest(key, timestamp, raw_body))
