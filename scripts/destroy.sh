#!/usr/bin/env bash
#
# Delete the SAM stack. The image repository is emptied and removed with it, and the
# SecureString parameters written by make secrets are kept.

set -euo pipefail

STACK_NAME="${STACK_NAME:-spendguard}"
AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION

if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
  echo "There is no $STACK_NAME stack in $AWS_REGION."
  exit 0
fi

repository_uri="$(
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
    --query "Stacks[0].Outputs[?OutputKey=='ImageRepositoryUri'].OutputValue" --output text
)"
read -r -p "Delete stack $STACK_NAME in $AWS_REGION and every image in $repository_uri? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || exit 1

aws cloudformation delete-stack --stack-name "$STACK_NAME"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"

repository_name="${repository_uri#*/}"
if aws ecr describe-repositories --repository-names "$repository_name" >/dev/null 2>&1; then
  echo "The image repository $repository_name still exists." >&2
  exit 1
fi
echo "Deleted $STACK_NAME and its image repository."
