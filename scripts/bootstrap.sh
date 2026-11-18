#!/usr/bin/env bash
#
# Create the pipeline resources GitHub Actions deploys with: an IAM role that only this repository's
# main branch can assume through GitHub's OIDC provider, a CloudFormation execution role and an
# artifacts bucket. An existing GitHub OIDC provider in the account is reused. Images go to the
# stack's own repository, so the bootstrap is given its ARN instead of creating a second one.

set -euo pipefail

STACK_NAME="${STACK_NAME:-spendguard}"
AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION
DEPLOYMENT_BRANCH=main

origin_url="$(git remote get-url origin)"
repository_path="${origin_url%.git}"
repository_path="${repository_path#*github.com[:/]}"
github_owner="${repository_path%%/*}"
github_repository="${repository_path#*/}"

# Repositories with immutable OIDC subjects sign tokens as repo:OWNER@OWNER-ID/REPO@REPO-ID, and
# SAM builds the trusted subject from these two values.
subject_prefix="$(gh api "repos/$github_owner/$github_repository/actions/oidc/customization/sub" \
  --jq '.sub_claim_prefix // empty')"
if [[ -n "$subject_prefix" ]]; then
  subject_path="${subject_prefix#repo:}"
  github_owner="${subject_path%%/*}"
  github_repository="${subject_path#*/}"
fi

account_id="$(aws sts get-caller-identity --query Account --output text)"
image_repository_arn="arn:aws:ecr:$AWS_REGION:$account_id:repository/$STACK_NAME"

# SAM CLI lets a saved .aws-sam/pipeline/pipelineconfig.toml override --github-org and
# --github-repo, so it runs in an empty directory.
sam_directory="$(mktemp -d)"
trap 'rm -rf "$sam_directory"' EXIT
cd "$sam_directory"

# SAM CLI replaces --oidc-provider with --cicd-provider's value, so both name GitHub Actions.
sam pipeline bootstrap \
  --no-interactive \
  --no-confirm-changeset \
  --stage "$STACK_NAME" \
  --region "$AWS_REGION" \
  --permissions-provider oidc \
  --cicd-provider github-actions \
  --oidc-provider github-actions \
  --oidc-provider-url https://token.actions.githubusercontent.com \
  --oidc-client-id sts.amazonaws.com \
  --github-org "$github_owner" \
  --github-repo "$github_repository" \
  --deployment-branch "$DEPLOYMENT_BRANCH" \
  --image-repository "$image_repository_arn"

bootstrap_output() {
  aws cloudformation describe-stacks --stack-name "aws-sam-cli-managed-$STACK_NAME-pipeline-resources" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

echo "Set these repository variables so deploy.yml can assume the pipeline role:"
echo "  gh variable set AWS_REGION --body '$AWS_REGION'"
echo "  gh variable set ALARM_EMAIL --body '<address for dead-letter alarms>'"
echo "  gh variable set PIPELINE_EXECUTION_ROLE_ARN --body '$(bootstrap_output PipelineExecutionRole)'"
echo "  gh variable set CLOUDFORMATION_EXECUTION_ROLE_ARN --body '$(bootstrap_output CloudFormationExecutionRole)'"
