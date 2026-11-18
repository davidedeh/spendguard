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
    for name in ('aws', 'sam', 'git', 'docker', 'uv'):
        fake_command(fake_bin, name, fakes.get(name, ''))
    completed = subprocess.run(
        [str(REPOSITORY_ROOT / 'scripts' / script)],
        input=reply,
        capture_output=True,
        text=True,
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
