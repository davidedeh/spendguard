#!/usr/bin/env bash
#
# Build, push and deploy the SAM stack from a laptop, then prepare the database and smoke-test the API.
# Needs AWS credentials for the target account, Docker, and ALARM_EMAIL.

set -euo pipefail

STACK_NAME="${STACK_NAME:-spendguard}"
AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION
: "${ALARM_EMAIL:?Set ALARM_EMAIL to the address that receives dead-letter queue alarms}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Commit or stash your changes first: the image is tagged with the commit it was built from." >&2
  exit 1
fi
image_tag="$(git rev-parse HEAD)"

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

deploy_stack() {
  aws cloudformation deploy \
    --stack-name "$STACK_NAME" \
    --template-file template.yaml \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
    --no-fail-on-empty-changeset \
    --parameter-overrides "ImageTag=$1" "AlarmEmail=$ALARM_EMAIL"
}

if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
  echo "Creating $STACK_NAME without functions so the image repository exists before the first push."
  deploy_stack ''
  echo "Confirm the SNS subscription email sent to $ALARM_EMAIL to receive dead-letter alarms."
fi

repository_uri="$(stack_output ImageRepositoryUri)"
image_uri="$repository_uri:$image_tag"
aws ecr get-login-password | docker login --username AWS --password-stdin "${repository_uri%%/*}"
BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker build --platform linux/arm64 -t "$image_uri" .
docker push "$image_uri"

deploy_stack "$image_tag"

preparation_function="$(stack_output PreparationFunctionName)"
preparation_response="$(mktemp)"
trap 'rm -f "$preparation_response"' EXIT
function_error="$(
  aws lambda invoke --function-name "$preparation_function" --cli-read-timeout 310 \
    --cli-binary-format raw-in-base64-out --payload '{"action": "prepare"}' \
    --query FunctionError --output text "$preparation_response"
)"
if [[ "$function_error" != "None" ]]; then
  echo "Database preparation failed: $(cat "$preparation_response")" >&2
  exit 1
fi
echo "Database migrated and policy indexed."

api_url="$(stack_output ApiUrl)"
uv run python -m spendguard.deployment.smoke "$api_url"
echo "Deployed $image_tag to $api_url"
