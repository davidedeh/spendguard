#!/usr/bin/env bash
#
# Render the edited demo video from the last take that record-demo.sh captured, and a narrated
# version when recordings/narration holds the narration clips.

set -euo pipefail

repository="$(cd "$(dirname "$0")/../.." && pwd)"
output="${OUTPUT:-$repository/recordings/spendguard-demo.mp4}"
narrated_output="${NARRATED_OUTPUT:-$repository/recordings/spendguard-demo-narrated.mp4}"
narration_source="$repository/recordings/narration"
github_video_upload_limit_bytes=10000000

warn_if_too_large() {
  echo "Wrote $1 ($(du -h "$1" | cut -f1))"
  if [ "$(stat -f %z "$1")" -gt "$github_video_upload_limit_bytes" ]; then
    echo 'It is over 10 MB: compress it before uploading it to GitHub' >&2
  fi
}

cd "$repository/video"
[ -f public/take/props.json ] || { echo 'No take yet: run make record-demo' >&2; exit 1; }
npm ci --no-audit --no-fund >/dev/null
mkdir -p "$(dirname "$output")"
npx remotion render src/index.ts SpendGuardDemo "$output" --props=public/take/props.json \
  --codec=h264 --crf=20 --log=error
warn_if_too_large "$output"

if [ ! -d "$narration_source" ]; then
  echo "No narration in recordings/narration: skipping the narrated version"
  exit 0
fi
command -v whisper-cli >/dev/null || { echo 'Install whisper.cpp first: brew install whisper-cpp' >&2; exit 1; }
(cd "$repository" && uv run python scripts/record-demo/prepare_narration.py \
  "$narration_source" video/public/take/narration >/dev/null)
npx remotion render src/index.ts SpendGuardDemoNarrated "$narrated_output" \
  --props=public/take/props.json --codec=h264 --crf=20 --audio-codec=aac --audio-bitrate=128k --log=error
warn_if_too_large "$narrated_output"
