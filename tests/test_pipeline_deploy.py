import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from cfnlint.decode import cfn_yaml

REPOSITORY_ROOT = Path(__file__).parent.parent
TEMPLATE: dict[str, Any] = cfn_yaml.load(str(REPOSITORY_ROOT / 'template.yaml'))
OIDC_PROVIDER_ARN = 'arn:aws:iam::111122223333:oidc-provider/token.actions.githubusercontent.com'
BOOTSTRAP_OWNS_THE_PROVIDER = f"""
case "$*" in
  *describe-stack-resource*OidcProvider*) echo '{OIDC_PROVIDER_ARN}' ;;
esac
true
"""
STACK_WITH_FUNCTIONS = """
case "$*" in
  sts\\ get-caller-identity*) echo 111122223333 ;;
  *PreparationFunctionName*) echo a-preparation-function ;;
  *ApiUrl*) echo https://an-api.example ;;
  lambda\\ invoke*) echo None ;;
esac
true
"""
# AWS CLI v2 reads a --cli-input-json file:// path twice before parsing it, so input that can
# only be read once, such as a pipe, arrives empty the second time.
SSM_WITH_NO_PARAMETERS = """
case "$1 $2" in
  ssm\\ get-parameter) exit 1 ;;
  ssm\\ put-parameter)
    for argument in "$@"; do
      [[ "$argument" == file://* ]] && input_path="${argument#file://}"
    done
    cat "$input_path" > /dev/null
    python3 -c 'import json, sys; print(json.dumps(json.load(sys.stdin)))' \\
      < "$input_path" >> "$PUT_PARAMETERS_LOG" || { echo 'Invalid JSON received.' >&2; exit 252; }
    ;;
esac
"""
COMMIT_SHA = 'c0ffee0000000000000000000000000000000001'
BOOTSTRAP_STACK = 'aws-sam-cli-managed-a-stack-pipeline-resources'
CLOUDFORMATION_ROLE_ARN = 'arn:aws:iam::111122223333:role/a-cloudformation-role'
PIPELINE_ROLE_ARN = 'arn:aws:iam::111122223333:role/a-pipeline-role'


def deploy_workflow() -> dict[Any, Any]:
    workflow: dict[Any, Any] = yaml.safe_load(
        (REPOSITORY_ROOT / '.github/workflows/deploy.yml').read_text()
    )
    return workflow


def fake_command(directory: Path, name: str, body: str) -> None:
    command = directory / name
    command.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "$CALLS_LOG"\n{body}\n')
    command.chmod(0o755)


def run_script(
    script: str,
    tmp_path: Path,
    fakes: dict[str, str],
    reply: str = '',
    environment: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    calls_log = tmp_path / 'calls.log'
    calls_log.touch()
    for name in ('aws', 'sam', 'git', 'gh', 'docker', 'uv'):
        fake_command(fake_bin, name, fakes.get(name, ''))
    completed = subprocess.run(
        [str(REPOSITORY_ROOT / 'scripts' / script)],
        input=reply,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            'PATH': f'{fake_bin}:{os.environ["PATH"]}',
            'CALLS_LOG': str(calls_log),
            'STACK_NAME': 'a-stack',
            'AWS_REGION': 'eu-west-2',
            **(environment or {}),
        },
        check=False,
    )
    return completed, calls_log.read_text().splitlines()


def option_values(arguments: list[str]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for position, argument in enumerate(arguments):
        if argument.startswith('--'):
            following = arguments[position + 1] if position + 1 < len(arguments) else '--'
            values[argument] = None if following.startswith('--') else following
    return values


def test_the_deploy_workflow_runs_only_on_a_push_to_main() -> None:
    triggers = deploy_workflow()[True]

    assert triggers == {'push': {'branches': ['main']}}


def test_the_deploy_workflow_reads_the_alarm_email_from_a_secret_so_run_logs_mask_it() -> None:
    environment = deploy_workflow()['jobs']['deploy']['env']

    assert environment['ALARM_EMAIL'] == '${{ secrets.ALARM_EMAIL }}'


@pytest.mark.parametrize(
    'origin_url', ['https://github.com/an-owner/a-repo.git', 'git@github.com:an-owner/a-repo.git']
)
def test_bootstrap_trusts_only_this_repositorys_main_branch_without_prompting(
    tmp_path: Path, origin_url: str
) -> None:
    completed, calls = run_script(
        'bootstrap.sh',
        tmp_path,
        fakes={
            'aws': '[[ "$*" == sts\\ get-caller-identity* ]] && echo 111122223333; true',
            'git': f'echo "{origin_url}"',
        },
    )

    assert completed.returncode == 0, completed.stderr
    [sam_call] = [call for call in calls if call.startswith('sam ')]
    assert sam_call.split()[1:3] == ['pipeline', 'bootstrap']
    assert option_values(sam_call.split()) == {
        '--no-interactive': None,
        '--no-confirm-changeset': None,
        '--stage': 'a-stack',
        '--region': 'eu-west-2',
        '--permissions-provider': 'oidc',
        '--cicd-provider': 'github-actions',
        '--oidc-provider': 'github-actions',
        '--oidc-provider-url': 'https://token.actions.githubusercontent.com',
        '--oidc-client-id': 'sts.amazonaws.com',
        '--github-org': 'an-owner',
        '--github-repo': 'a-repo',
        '--deployment-branch': 'main',
        '--image-repository': 'arn:aws:ecr:eu-west-2:111122223333:repository/a-stack',
    }
    printed_commands = [line.split()[:4] for line in completed.stdout.splitlines()]
    assert ['gh', 'secret', 'set', 'ALARM_EMAIL'] in printed_commands
    assert ['gh', 'variable', 'set', 'ALARM_EMAIL'] not in printed_commands


def test_bootstrap_trusts_the_immutable_subject_github_issues_for_the_repository(
    tmp_path: Path,
) -> None:
    completed, calls = run_script(
        'bootstrap.sh',
        tmp_path,
        fakes={
            'aws': '[[ "$*" == sts\\ get-caller-identity* ]] && echo 111122223333; true',
            'git': 'echo https://github.com/an-owner/a-repo.git',
            'gh': (
                '[[ "$*" == "api repos/an-owner/a-repo/actions/oidc/customization/sub"* ]]'
                ' && echo "repo:an-owner@9100/a-repo@145"; true'
            ),
        },
    )

    assert completed.returncode == 0, completed.stderr
    [sam_call] = [call for call in calls if call.startswith('sam ')]
    options = option_values(sam_call.split())
    assert (options['--github-org'], options['--github-repo']) == ('an-owner@9100', 'a-repo@145')


def test_bootstrap_ignores_the_github_repository_sam_saved_from_an_earlier_run(
    tmp_path: Path,
) -> None:
    saved_config = tmp_path / '.aws-sam' / 'pipeline' / 'pipelineconfig.toml'
    saved_config.parent.mkdir(parents=True)
    saved_config.write_text('[default.pipeline_bootstrap.parameters]\ngithub_org = "old"\n')

    completed, _ = run_script(
        'bootstrap.sh',
        tmp_path,
        fakes={
            'aws': '[[ "$*" == sts\\ get-caller-identity* ]] && echo 111122223333; true',
            'git': 'echo https://github.com/an-owner/a-repo.git',
            'sam': (
                '[[ -e .aws-sam/pipeline/pipelineconfig.toml ]]'
                ' && echo "sam saw a saved config"; true'
            ),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert 'sam saw a saved config' not in completed.stdout


def test_destroy_bootstrap_warns_about_the_shared_oidc_provider_and_stops_on_no(
    tmp_path: Path,
) -> None:
    completed, calls = run_script(
        'destroy-bootstrap.sh', tmp_path, fakes={'aws': BOOTSTRAP_OWNS_THE_PROVIDER}, reply='n\n'
    )

    assert completed.returncode != 0
    assert OIDC_PROVIDER_ARN in completed.stdout
    assert 'every role in this account that trusts GitHub Actions' in completed.stdout
    assert not [call for call in calls if 'delete-stack' in call]


def test_destroy_bootstrap_keeps_a_provider_it_did_not_create_and_deletes_on_yes(
    tmp_path: Path,
) -> None:
    completed, calls = run_script(
        'destroy-bootstrap.sh',
        tmp_path,
        fakes={'aws': '[[ "$*" == *describe-stack-resource* ]] && exit 254; true'},
        reply='y\n',
    )

    assert completed.returncode == 0, completed.stderr
    assert f'was not created by {BOOTSTRAP_STACK} and is kept' in completed.stdout
    assert 'WARNING' not in completed.stdout
    assert [call for call in calls if 'delete-stack' in call] == [
        f'aws cloudformation delete-stack --stack-name {BOOTSTRAP_STACK}'
    ]


def test_deploy_runs_sam_deploy_against_the_stacks_repository_as_the_execution_role(
    tmp_path: Path,
) -> None:
    completed, calls = run_script(
        'deploy.sh',
        tmp_path,
        fakes={
            'aws': STACK_WITH_FUNCTIONS,
            'git': f'[[ "$1" == rev-parse ]] && echo {COMMIT_SHA}; true',
        },
        environment={
            'ALARM_EMAIL': 'alarms@example.com',
            'CLOUDFORMATION_EXECUTION_ROLE_ARN': CLOUDFORMATION_ROLE_ARN,
            'PIPELINE_EXECUTION_ROLE_ARN': PIPELINE_ROLE_ARN,
        },
    )

    assert completed.returncode == 0, completed.stderr
    [sam_call] = [call for call in calls if call.startswith('sam ')]
    sam_arguments = sam_call.split()
    assert sam_arguments[1] == 'deploy'
    options = option_values(sam_arguments)
    assert options['--image-repository'] == '111122223333.dkr.ecr.eu-west-2.amazonaws.com/a-stack'
    assert options['--role-arn'] == CLOUDFORMATION_ROLE_ARN
    assert options['--stack-name'] == 'a-stack'
    assert f'ImageTag={COMMIT_SHA}' in sam_arguments
    assert f'PipelineExecutionRoleArn={PIPELINE_ROLE_ARN}' in sam_arguments
    assert [call for call in calls if call.startswith('uv ')] == [
        'uv run python -m spendguard.deployment.smoke https://an-api.example'
    ]


def test_a_first_deploy_creates_the_stack_without_an_image_tag_then_deploys_the_commit(
    tmp_path: Path,
) -> None:
    no_stack_yet = STACK_WITH_FUNCTIONS.replace(
        '  lambda\\ invoke*) echo None ;;\n',
        '  lambda\\ invoke*) echo None ;;\n  cloudformation\\ describe-stacks*) exit 254 ;;\n',
    )
    completed, calls = run_script(
        'deploy.sh',
        tmp_path,
        fakes={
            'aws': no_stack_yet,
            'git': f'[[ "$1" == rev-parse ]] && echo {COMMIT_SHA}; true',
        },
        environment={'ALARM_EMAIL': 'alarms@example.com'},
    )

    assert completed.returncode == 0, completed.stderr
    first_deploy, second_deploy = [call.split() for call in calls if call.startswith('sam ')]
    assert [argument for argument in first_deploy if argument.startswith('ImageTag')] == []
    assert f'ImageTag={COMMIT_SHA}' in second_deploy


def test_a_laptop_deploy_without_pipeline_roles_deploys_as_the_caller_and_keeps_the_previous_grant(
    tmp_path: Path,
) -> None:
    completed, calls = run_script(
        'deploy.sh',
        tmp_path,
        fakes={
            'aws': STACK_WITH_FUNCTIONS,
            'git': f'[[ "$1" == rev-parse ]] && echo {COMMIT_SHA}; true',
        },
        environment={'ALARM_EMAIL': 'alarms@example.com'},
    )

    assert completed.returncode == 0, completed.stderr
    [sam_call] = [call for call in calls if call.startswith('sam ')]
    assert '--role-arn' not in sam_call
    assert 'PipelineExecutionRoleArn' not in sam_call


def test_the_image_repository_is_named_after_the_stack_so_bootstrap_can_grant_pushes_to_it() -> (
    None
):
    repository = TEMPLATE['Resources']['ImageRepository']['Properties']

    assert repository['RepositoryName'] == {'Ref': 'AWS::StackName'}


def test_the_pipeline_role_given_to_the_stack_may_invoke_only_the_preparation_function() -> None:
    [grant] = [
        resource
        for resource in TEMPLATE['Resources'].values()
        if resource['Type'] == 'AWS::IAM::Policy'
        and resource['Properties']['Roles']
        == [{'Fn::Select': [1, {'Fn::Split': ['/', {'Ref': 'PipelineExecutionRoleArn'}]}]}]
    ]

    assert TEMPLATE['Conditions'][grant['Condition']] == {
        'Fn::And': [
            {'Condition': 'HasImage'},
            {'Fn::Not': [{'Fn::Equals': [{'Ref': 'PipelineExecutionRoleArn'}, '']}]},
        ]
    }
    assert grant['Properties']['PolicyDocument']['Statement'] == [
        {
            'Effect': 'Allow',
            'Action': 'lambda:InvokeFunction',
            'Resource': {'Fn::GetAtt': ['PreparationFunction', 'Arn']},
        }
    ]
    assert TEMPLATE['Parameters']['PipelineExecutionRoleArn']['Default'] == ''


def test_secrets_reach_ssm_byte_for_byte_whatever_characters_they_hold(tmp_path: Path) -> None:
    put_parameters_log = tmp_path / 'put-parameters.log'
    database_url = 'postgresql://postgres.ref:p"a\\ss$wo@rd%21!x@pooler.example:6543/postgres'
    webhook_signing_key = '  a key with \\ backslash and spaces  '
    langfuse_public_key = 'pk-lf-$HOME-!$(date)'

    completed, _ = run_script(
        'secrets.sh',
        tmp_path,
        fakes={'aws': SSM_WITH_NO_PARAMETERS},
        reply=f'{database_url}\n{webhook_signing_key}\n\n{langfuse_public_key}\n\n',
        environment={'PARAMETER_PATH': 'a-path', 'PUT_PARAMETERS_LOG': str(put_parameters_log)},
    )

    assert completed.returncode == 0, completed.stderr
    written = [json.loads(line) for line in put_parameters_log.read_text().splitlines()]
    assert {parameter['Name'] for parameter in written} == {
        '/a-path/DATABASE_URL',
        '/a-path/WEBHOOK_SIGNING_KEY',
        '/a-path/DECISION_EVENT_SIGNING_KEY',
        '/a-path/LANGFUSE_PUBLIC_KEY',
    }
    values = {parameter['Name']: parameter['Value'] for parameter in written}
    assert values['/a-path/DATABASE_URL'] == database_url
    assert values['/a-path/WEBHOOK_SIGNING_KEY'] == webhook_signing_key
    assert values['/a-path/LANGFUSE_PUBLIC_KEY'] == langfuse_public_key
    assert len(values['/a-path/DECISION_EVENT_SIGNING_KEY']) == 64
    assert all(
        (parameter['Type'], parameter['Overwrite']) == ('SecureString', True)
        for parameter in written
    )
