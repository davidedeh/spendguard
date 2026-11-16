import boto3
from botocore.stub import Stubber

from spendguard.settings import parameter_store_values


def parameter(name: str, value: str) -> dict[str, object]:
    return {'Name': name, 'Type': 'SecureString', 'Value': value}


def test_every_parameter_under_the_prefix_becomes_the_setting_it_names() -> None:
    client = boto3.client('ssm', region_name='eu-west-2')
    with Stubber(client) as stubber:
        stubber.add_response(
            'get_parameters_by_path',
            {
                'Parameters': [parameter('/spendguard/WEBHOOK_SIGNING_KEY', 'webhook key')],
                'NextToken': 'page-2',
            },
            {'Path': '/spendguard', 'WithDecryption': True, 'Recursive': False},
        )
        stubber.add_response(
            'get_parameters_by_path',
            {
                'Parameters': [
                    parameter('/spendguard/DATABASE_URL', 'postgresql://pooler/postgres'),
                    parameter('/spendguard/LANGFUSE_SECRET_KEY', 'langfuse secret'),
                ]
            },
            {
                'Path': '/spendguard',
                'WithDecryption': True,
                'Recursive': False,
                'NextToken': 'page-2',
            },
        )

        values = parameter_store_values(client, '/spendguard')

    assert values == {
        'webhook_signing_key': 'webhook key',
        'database_url': 'postgresql://pooler/postgres',
        'langfuse_secret_key': 'langfuse secret',
    }
