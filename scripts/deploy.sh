#!/usr/bin/env bash
#
# Build, push and deploy the SAM stack, then prepare the database and smoke-test the API.
# Needs AWS credentials for the target account, Docker, SAM CLI and ALARM_EMAIL.
# CLOUDFORMATION_EXECUTION_ROLE_ARN: role CloudFormation deploys as (the pipeline role cannot create IAM resources).
# PIPELINE_EXECUTION_ROLE_ARN: role the stack lets invoke the preparation function; kept from the last deploy when unset.

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
account_id="$(aws sts get-caller-identity --query Account --output text)"
repository_uri="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com/$STACK_NAME"

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

# SAM CLI requires --image-repository for any template with image functions, but it never pushes:
# the functions' ImageUri is an intrinsic, which sam deploy leaves as it is.
deploy_stack() {
  # SAM CLI rejects an empty override, and the template's empty ImageTag default is the stack without
  # functions.
  local parameter_overrides=("AlarmEmail=$ALARM_EMAIL")
  if [[ -n "$1" ]]; then
    parameter_overrides+=("ImageTag=$1")
  fi
  if [[ -n "${PIPELINE_EXECUTION_ROLE_ARN:-}" ]]; then
    parameter_overrides+=("PipelineExecutionRoleArn=$PIPELINE_EXECUTION_ROLE_ARN")
  fi
  local role_arguments=()
  if [[ -n "${CLOUDFORMATION_EXECUTION_ROLE_ARN:-}" ]]; then
    role_arguments=(--role-arn "$CLOUDFORMATION_EXECUTION_ROLE_ARN")
  fi
  sam deploy \
    --template-file template.yaml \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --image-repository "$repository_uri" \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND \
    --no-confirm-changeset \
    --no-fail-on-empty-changeset \
    ${role_arguments[@]+"${role_arguments[@]}"} \
    --parameter-overrides "${parameter_overrides[@]}"
}

if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
  echo "Creating $STACK_NAME without functions so the image repository exists before the first push."
  deploy_stack ''
  echo "Confirm the SNS subscription email sent to $ALARM_EMAIL to receive dead-letter alarms."
fi

image_uri="$repository_uri:$image_tag"
aws ecr get-login-password | docker login --username AWS --password-stdin "${repository_uri%%/*}"
BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker build --platform linux/arm64 -t "$image_uri" .
docker push "$image_uri"

deploy_stack "$image_tag"

preparation_function="$(stack_output PreparationFunctionName)"
preparation_response="$(mktemp)"
trap 'rm -f "$preparation_response"' EXIT
invoke_preparation() {
  aws lambda invoke --function-name "$preparation_function" --cli-read-timeout 310 \
    --cli-binary-format raw-in-base64-out --payload '{"action": "prepare"}' \
    --query FunctionError --output text "$preparation_response"
}
# A permission granted by the deploy that just finished can take seconds to reach IAM.
for attempt in 1 2 3 4 5 6; do
  if function_error="$(invoke_preparation)"; then
    break
  fi
  if [[ "$attempt" == 6 ]]; then
    echo "Could not invoke $preparation_function." >&2
    exit 1
  fi
  sleep 10
done
if [[ "$function_error" != "None" ]]; then
  echo "Database preparation failed: $(cat "$preparation_response")" >&2
  exit 1
fi
echo "Database migrated and policy indexed."

api_url="$(stack_output ApiUrl)"
uv run python -m spendguard.deployment.smoke "$api_url"
echo "Deployed $image_tag to $api_url"
