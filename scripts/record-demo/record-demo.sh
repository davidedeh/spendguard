#!/usr/bin/env bash
#
# Run the presentation demo on a stack of its own, capture its output with timings and stills of
# the injection claim's Langfuse trace, then render the edited video. Calls the configured models
# and sends traces to Langfuse.

set -euo pipefail

repository="$(cd "$(dirname "$0")/../.." && pwd)"
scripts="$repository/scripts/record-demo"
take="$repository/video/public/take"
logs="$repository/recordings"
port="${RECORDING_PORT:-8020}"
playwright_version='1.58.0'
injection_sample='from spendguard.demo import INJECTION_CASE, demo_sample
from spendguard.eval.freeze import CASE_DIRECTORY
sample = demo_sample(CASE_DIRECTORY, INJECTION_CASE)'

for tool in jq node; do
  command -v "$tool" >/dev/null || { echo "Install $tool first" >&2; exit 1; }
done

cd "$repository"
injection_key="$(uv run python -c "$injection_sample; print(sample.idempotency_key)")"
injected_purpose="$(uv run python -c "$injection_sample; print(sample.delivery['claim']['business_purpose'])")"
rm -rf "$take"
rm -f "$logs"/*.log "$logs"/*.json
mkdir -p "$take" "$logs"

export BRANCH_NAME="${BRANCH_NAME:-$(git branch --show-current)-recording}"
export DECISION_EVENT_URL="http://localhost:$port/mock-erp/decision-events"

echo "Preparing a fresh stack for $BRANCH_NAME (logs in recordings/)"
make reset-local index-policy >>"$logs/setup.log" 2>&1 || {
  tail -3 "$logs/setup.log" >&2
  echo 'Preparing the stack failed; see recordings/setup.log' >&2
  exit 1
}

uv run uvicorn spendguard.api.app:create_production_app --factory --port "$port" \
  >"$logs/api.log" 2>&1 &
api_pid=$!
uv run python -m spendguard.worker >"$logs/worker.log" 2>&1 &
worker_pid=$!
trap 'kill "$api_pid" "$worker_pid" 2>/dev/null || true' EXIT

for _ in $(seq 60); do
  curl -sf "http://localhost:$port/health" >/dev/null && break
  sleep 1
done
curl -sf "http://localhost:$port/health" >/dev/null || { echo 'The API did not start' >&2; exit 1; }
token="$(uv run python -m spendguard.security.approver_tokens EMP-001 2>>"$logs/setup.log")"

echo 'Running the presentation demo'
uv run python "$scripts/capture_demo.py" "http://localhost:$port" "$token" "$take/capture.json"

echo 'Sharing the demo traces'
uv run python "$scripts/share_traces.py" "$logs/worker.log" >"$logs/traces.json"
trace_url="$(jq -r --arg key "$injection_key" '.[$key][0].url' "$logs/traces.json")"

echo "Capturing stills of $trace_url"
uv run --no-project --with "playwright==$playwright_version" python -I \
  "$scripts/trace_shots.py" "$trace_url" "$take" "$injected_purpose" >"$take/trace.json"

repository_url="$(git remote get-url origin | sed -e 's/\.git$//' -e 's|^git@github.com:|https://github.com/|')"
uv run python "$scripts/assemble_props.py" "$take/capture.json" "$logs/traces.json" \
  "$take/trace.json" "$repository_url" "$take/props.json"

"$scripts/render-demo.sh"
