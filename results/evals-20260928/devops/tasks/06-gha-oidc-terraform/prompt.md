Write a GitHub Actions workflow `.github/workflows/terraform.yml` for a Terraform root module in the `infra/` directory that deploys to AWS.

Requirements:
- Authenticate to AWS with GitHub OIDC by assuming the role `arn:aws:iam::123456789012:role/gha-terraform` in `eu-west-1`. No long-lived AWS keys anywhere.
- On pull requests to `main`: `terraform fmt -check`, `terraform init`, `terraform validate` and `terraform plan` (non-interactive).
- On pushes to `main`: the same checks, then `terraform apply` of the reviewed configuration, gated by a GitHub Environment named `production`.
- Least-privilege `permissions` for the `GITHUB_TOKEN`.
- Never run two applies at once.

Return the workflow in one ```yaml fenced block. It must pass `actionlint` with no findings.
