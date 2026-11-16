#!/usr/bin/env bash
#
# Deletes the stack holding the deploy pipeline's bootstrap resources. The GitHub OIDC
# provider is account-wide: if this stack created it, deleting the stack removes it for
# every other role that trusts it.

set -euo pipefail

STACK_NAME="${STACK_NAME:-spendguard}"
AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION
bootstrap_stack="aws-sam-cli-managed-$STACK_NAME-pipeline-resources"

if ! aws cloudformation describe-stacks --stack-name "$bootstrap_stack" >/dev/null 2>&1; then
  echo "There is no $bootstrap_stack stack in $AWS_REGION."
  exit 0
fi

if oidc_provider="$(
  aws cloudformation describe-stack-resource --stack-name "$bootstrap_stack" \
    --logical-resource-id OidcProvider \
    --query StackResourceDetail.PhysicalResourceId --output text 2>/dev/null
)" && [[ -n "$oidc_provider" ]]; then
  echo "WARNING: $bootstrap_stack created the GitHub OIDC provider $oidc_provider."
  echo "Deleting the stack deletes the provider, and every role in this account that trusts GitHub Actions"
  echo "through it stops working, including roles other projects created."
else
  echo "The GitHub OIDC provider was not created by $bootstrap_stack and is kept."
fi

read -r -p "Delete $bootstrap_stack in $AWS_REGION? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || exit 1

aws cloudformation delete-stack --stack-name "$bootstrap_stack"
aws cloudformation wait stack-delete-complete --stack-name "$bootstrap_stack"
echo "Deleted $bootstrap_stack. Its artifacts bucket and that bucket's logging bucket are retained by design;"
echo "empty and delete them in the S3 console if you no longer want them."
