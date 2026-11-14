import sys

from spendguard.policy.loader import POLICY_PATH, load_policy

sys.stdout.write(load_policy(POLICY_PATH).rendered_text)
