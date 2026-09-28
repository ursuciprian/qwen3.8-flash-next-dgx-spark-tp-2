`terraform validate` fails on this configuration (`main.tf`). Find every error, explain each one in a single line, then return the corrected, complete `main.tf` in one ```hcl fenced block. Keep the intent: one private subnet per availability zone and a database security group that allows PostgreSQL only from those subnets. The result must pass `terraform fmt -check` and `terraform validate`.

```hcl
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "azs" { type = list(string) default = ["eu-west-1a", "eu-west-1b"] }

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "private" {
  count             = length(var.azs)
  vpc_id            = aws_vpc.main.vpc_id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count)
  availability_zone = var.azs[count.index]
  tags = { Name = "private-${var.azs[count.index]}" }
}

resource "aws_security_group" "db" {
  name   = "db"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 5432 to_port = 5432 protocol = "tcp" cidr_blocks = aws_subnet.private.cidr_block }
}

output "subnet_ids" {
  value = aws_subnet.private.id
}
```
