Review this IAM policy, which is attached to the execution role of the Lambda function `thumbnailer` in account 123456789012, region eu-west-1.

What the function actually does:
- reads objects under `s3://acme-uploads/incoming/` (objects are encrypted with the KMS key `arn:aws:kms:eu-west-1:123456789012:key/0a1b2c3d-1111-2222-3333-444455556666`),
- writes thumbnails to `s3://acme-thumbs/` (same KMS key, SSE-KMS),
- writes one item per image to the DynamoDB table `thumbnails`,
- writes its own CloudWatch Logs.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "s3:*", "Resource": "*" },
    { "Effect": "Allow", "Action": "dynamodb:*", "Resource": "*" },
    { "Effect": "Allow", "Action": ["iam:PassRole", "iam:GetRole"], "Resource": "*" },
    { "Effect": "Allow", "Action": "logs:*", "Resource": "*" },
    { "Effect": "Allow", "Action": ["kms:Decrypt", "kms:GenerateDataKey"], "Resource": "*" }
  ]
}
```

1. List the findings ranked by risk, one or two lines each.
2. Return a least-privilege replacement policy in one ```json fenced block. Scope every statement to specific resource ARNs; use no service-level wildcards such as `s3:*`.
