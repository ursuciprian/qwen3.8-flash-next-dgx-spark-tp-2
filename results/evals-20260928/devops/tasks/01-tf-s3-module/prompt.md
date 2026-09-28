Write a reusable Terraform module (Terraform >= 1.5, AWS provider ~> 5.0) that creates a private S3 bucket for application logs.

Requirements:
- Variables: `bucket_name` (string), `kms_key_arn` (string), `noncurrent_days` (number, default 90), `tags` (map(string), default {}).
- Versioning enabled.
- Default encryption with the given KMS key (`aws:kms`), with S3 Bucket Keys enabled.
- All four public access block settings set to true.
- Object ownership `BucketOwnerEnforced` (ACLs disabled).
- One lifecycle rule that expires noncurrent object versions after `noncurrent_days` days and aborts incomplete multipart uploads after 7 days.
- A bucket policy that denies any request not made over TLS.
- Outputs `bucket_id` and `bucket_arn`.
- Use the standalone resources (`aws_s3_bucket_versioning`, `aws_s3_bucket_server_side_encryption_configuration`, ...), not deprecated inline blocks on `aws_s3_bucket`.

Return exactly four files: `versions.tf`, `variables.tf`, `main.tf`, `outputs.tf`. Put each file in its own fenced code block, and put the file name on the line directly before its block, formatted as `### <filename>`. The code must pass `terraform fmt -check` and `terraform validate`.
