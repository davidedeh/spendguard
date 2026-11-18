import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit('usage: capture_demo.py <api base URL> <approver token> <capture JSON>')
    base_url, token, capture_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    started = time.monotonic()
    lines: list[dict[str, object]] = []
    with subprocess.Popen(
        [sys.executable, '-m', 'spendguard.demo_presentation', base_url, token],
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(
                {'seconds': round(time.monotonic() - started, 2), 'text': line.rstrip('\n')}
            )
            print(line, end='', flush=True)
    capture_path.write_text(json.dumps({'lines': lines}, indent=2))
    if process.returncode != 0:
        sys.exit(process.returncode)


if __name__ == '__main__':
    main()
