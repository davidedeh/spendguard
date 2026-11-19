# Deploying

Every push to main deploys through `.github/workflows/deploy.yml`, which assumes an AWS role through GitHub's OIDC provider, so the repository holds no AWS keys. One-off setup, with admin credentials for the account, Docker, SAM CLI and a signed-in GitHub CLI:

```sh
make bootstrap      # pipeline role trusted only by this repository's main branch
make secrets        # SecureString parameters the functions read at start-up
```

`make secrets` asks for the Supabase transaction pooler URL and the signing keys, generating the keys if you press Enter. `make bootstrap` prints the repository variables and the alarm email secret `deploy.yml` reads. `make deploy ALARM_EMAIL=you@example.com` runs the same script from a laptop: it builds and pushes the arm64 image tagged with the commit, deploys the stack, runs migrations and indexes the policy through the preparation function, then smoke-tests the API URL. Confirm the SNS subscription email to receive dead-letter alarms.

The deployed database starts with no approvers. Seed it and issue an approver token from a laptop, pointing the local commands at Supabase's session pooler URI (port 5432; `DATABASE_URL` in the environment overrides `.env`):

```sh
read -rs DATABASE_URL && export DATABASE_URL   # paste the session pooler URI
make seed
make issue-token APPROVER=EMP-001
unset DATABASE_URL
```

To tear down, `make destroy` deletes the stack and its images, and `make destroy-bootstrap` deletes the pipeline resources, warning first if that would also delete the account's GitHub OIDC provider.

Images live in the stack's own repository, which keeps the five newest. The bootstrap is pointed at that repository rather than creating a second one.

## What the bootstrap grants

Read from the bootstrap template that SAM CLI 1.172.1 ships (`samcli/lib/pipeline/bootstrap/stage_resources.yaml`):

- **Pipeline execution role**, assumed by GitHub Actions. Trust: `sts:AssumeRoleWithWebIdentity` from `token.actions.githubusercontent.com` with audience `sts.amazonaws.com` and subject `repo:<owner>/<repo>:ref:refs/heads/main`, or `repo:<owner>@<owner-id>/<repo>@<repo-id>:ref:refs/heads/main` when the repository uses GitHub's immutable subjects, so pull requests, other branches and jobs that set a GitHub `environment` cannot assume it. Permissions: create, describe and execute CloudFormation change sets and delete stacks on any stack; pass the CloudFormation execution role; read and write the artifacts bucket; push and pull images in the stack's repository. The stack adds `lambda:InvokeFunction` on the preparation function only.
- **CloudFormation execution role**, assumed only by the CloudFormation service. Permissions: `Action: *` on `Resource: *`, full administrator. This is broad: anyone who can push to main can deploy a template that does anything in the account. It is accepted for a single-owner demo account; narrowing it means passing your own role to `sam pipeline bootstrap --cloudformation-execution-role` with the IAM, Lambda, SQS, SNS, CloudWatch, ECR, EventBridge Scheduler and SSM actions this template uses.
- **Artifacts bucket** and its logging bucket, both retained when the bootstrap stack is deleted.
