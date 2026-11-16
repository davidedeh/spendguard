#!/usr/bin/env bash
#
# Write the deployed stack's secrets as SSM SecureString parameters under /$PARAMETER_PATH.
# CloudFormation cannot create SecureString parameters, so they live outside the stack.
# Input is hidden; Enter on an existing parameter keeps it.

set -euo pipefail

PARAMETER_PATH="${PARAMETER_PATH:-spendguard}"
AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_REGION

parameter_exists() {
  aws ssm get-parameter --name "/$PARAMETER_PATH/$1" >/dev/null 2>&1
}

put_secure_string() {
  PARAMETER_NAME="/$PARAMETER_PATH/$1" PARAMETER_VALUE="$2" python3 -c '
import json, os
print(json.dumps({
    "Name": os.environ["PARAMETER_NAME"],
    "Value": os.environ["PARAMETER_VALUE"],
    "Type": "SecureString",
    "Overwrite": True,
}))' | aws ssm put-parameter --cli-input-json file:///dev/stdin >/dev/null
  echo "  wrote /$PARAMETER_PATH/$1"
}

# ask_secret NAME PROMPT WHEN_EMPTY, where WHEN_EMPTY is "required", "generate" or "optional".
ask_secret() {
  local name="$1" prompt="$2" when_empty="$3" value=""
  if parameter_exists "$name"; then
    printf '%s [Enter keeps the current value] ' "$prompt"
  else
    printf '%s ' "$prompt"
  fi
  read -rs value
  printf '\n'
  if [[ -z "$value" ]] && parameter_exists "$name"; then
    echo "  kept /$PARAMETER_PATH/$name"
    return
  fi
  if [[ -z "$value" ]]; then
    case "$when_empty" in
      required) echo "  $name is required." >&2; exit 1 ;;
      generate) value="$(openssl rand -hex 32)" ;;
      optional) echo "  skipped $name"; return ;;
    esac
  fi
  put_secure_string "$name" "$value"
}

echo "Writing SecureString parameters under /$PARAMETER_PATH in $AWS_REGION."
ask_secret DATABASE_URL "Supabase transaction pooler URI (port 6543):" required
ask_secret WEBHOOK_SIGNING_KEY "Webhook signing key (Enter generates one):" generate
ask_secret DECISION_EVENT_SIGNING_KEY "Decision event signing key (Enter generates one):" generate
ask_secret LANGFUSE_PUBLIC_KEY "Langfuse public key (Enter skips tracing):" optional
ask_secret LANGFUSE_SECRET_KEY "Langfuse secret key (Enter skips tracing):" optional
echo "Read a generated key back with: aws ssm get-parameter --with-decryption --name /$PARAMETER_PATH/WEBHOOK_SIGNING_KEY"
echo "Functions read these at cold start; redeploy or wait for new instances to pick up changes."
