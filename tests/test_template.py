import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import boto3
import pytest
from botocore.stub import Stubber
from cfnlint.decode import cfn_yaml

from spendguard.queue.messages import MAX_RECEIVE_COUNT
from spendguard.settings import VISIBILITY_TIMEOUTS_PER_WORKER_TIMEOUT, Settings
from spendguard.settings import load_settings as load_settings_from_files
from test_parameter_store import parameter

if TYPE_CHECKING:
    from mypy_boto3_ssm import SSMClient

TEMPLATE: dict[str, Any] = cfn_yaml.load(str(Path(__file__).parent.parent / 'template.yaml'))
RESOURCES: dict[str, dict[str, Any]] = TEMPLATE['Resources']
SETTINGS_LAMBDA_PROVIDES = {'aws_region'}
unpatched_boto3_client = boto3.client


def resources_of_type(resource_type: str) -> dict[str, dict[str, Any]]:
    return {name: body for name, body in RESOURCES.items() if body['Type'] == resource_type}


def test_the_template_has_no_vpc_nat_gateway_or_load_balancer() -> None:
    networking_types = {
        body['Type']
        for body in RESOURCES.values()
        if body['Type'].startswith(('AWS::EC2::', 'AWS::ElasticLoadBalancing'))
    }
    functions_in_a_vpc = [
        name
        for name, body in resources_of_type('AWS::Serverless::Function').items()
        if 'VpcConfig' in body['Properties']
    ]

    assert networking_types == set()
    assert functions_in_a_vpc == []
    assert 'VpcConfig' not in TEMPLATE['Globals']['Function']


def test_the_image_repository_keeps_only_the_five_newest_images() -> None:
    (repository,) = resources_of_type('AWS::ECR::Repository').values()
    lifecycle = json.loads(repository['Properties']['LifecyclePolicy']['LifecyclePolicyText'])

    (rule,) = lifecycle['rules']
    assert rule['selection'] == {
        'tagStatus': 'any',
        'countType': 'imageCountMoreThan',
        'countNumber': 5,
    }
    assert rule['action'] == {'type': 'expire'}


def test_the_queue_visibility_timeout_is_at_least_six_worker_timeouts() -> None:
    worker = RESOURCES['WorkerFunction']['Properties']
    queue = RESOURCES['SubmissionQueue']['Properties']
    environment = TEMPLATE['Globals']['Function']['Environment']['Variables']

    assert queue['VisibilityTimeout'] >= VISIBILITY_TIMEOUTS_PER_WORKER_TIMEOUT * worker['Timeout']
    assert int(environment['WORKER_TIMEOUT_SECONDS']) == worker['Timeout']
    assert int(environment['QUEUE_VISIBILITY_TIMEOUT_SECONDS']) == queue['VisibilityTimeout']


def test_a_message_moves_to_the_dead_letter_queue_after_three_receives() -> None:
    redrive = RESOURCES['SubmissionQueue']['Properties']['RedrivePolicy']

    assert redrive['maxReceiveCount'] == MAX_RECEIVE_COUNT
    assert redrive['deadLetterTargetArn'] == {'Fn::GetAtt': ['DeadLetterQueue', 'Arn']}


def test_every_setting_is_configured_for_the_functions() -> None:
    environment = TEMPLATE['Globals']['Function']['Environment']['Variables']
    configured = {key.split('__')[0].lower() for key in environment}

    assert configured | SETTINGS_LAMBDA_PROVIDES == set(Settings.model_fields)


substitution_pattern = re.compile(r'\$\{([^}]+)\}')


def resolved(value: Any) -> str:
    if isinstance(value, str):
        return value
    if 'Ref' in value:
        name = value['Ref']
        parameter_definition = TEMPLATE['Parameters'].get(name, {})
        return str(parameter_definition.get('Default', f'https://{name.lower()}.invalid/'))
    return substitution_pattern.sub(
        lambda match: resolved({'Ref': match.group(1)}), value['Fn::Sub']
    )


def function_environment(function_name: str) -> dict[str, str]:
    function_variables = RESOURCES[function_name]['Properties']['Environment']['Variables']
    variables = TEMPLATE['Globals']['Function']['Environment']['Variables'] | function_variables
    return {key: resolved(value) for key, value in variables.items()}


def stubbed_parameter_store(service_name: Literal['ssm'], region_name: str) -> 'SSMClient':
    client = unpatched_boto3_client('ssm', region_name=region_name)
    stubber = Stubber(client)
    stubber.add_response(
        'get_parameters_by_path',
        {
            'Parameters': [
                parameter('/spendguard/DATABASE_URL', 'postgresql://pooler.invalid:6543/postgres'),
                parameter('/spendguard/WEBHOOK_SIGNING_KEY', 'webhook key'),
                parameter('/spendguard/DECISION_EVENT_SIGNING_KEY', 'decision key'),
            ]
        },
        {'Path': '/spendguard', 'WithDecryption': True, 'Recursive': False},
    )
    stubber.activate()
    return client


@pytest.mark.parametrize('function_name', ['ApiFunction', 'WorkerFunction', 'PreparationFunction'])
def test_each_function_environment_with_its_secrets_loads_into_deployed_settings(
    monkeypatch: pytest.MonkeyPatch, function_name: str
) -> None:
    for key in function_environment(function_name):
        monkeypatch.delenv(key, raising=False)
    for key, value in function_environment(function_name).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    monkeypatch.setattr(boto3, 'client', stubbed_parameter_store)

    settings = load_settings_from_files(None)

    assert settings.app_environment == 'deployed'
    assert settings.database_url is not None
    assert settings.database_url.get_secret_value() == 'postgresql://pooler.invalid:6543/postgres'
    assert settings.webhook_signing_key is not None
    assert settings.webhook_signing_key.get_secret_value() == 'webhook key'
    assert settings.langfuse_public_key is None
    assert (
        settings.chat_models['primary'].model_id
        == TEMPLATE['Parameters']['PrimaryModelId']['Default']
    )
