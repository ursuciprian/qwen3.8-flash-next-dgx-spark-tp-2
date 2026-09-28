A colleague wants to apply this Terraform plan to the production account right now. Review it: give the resource counts in the form `X to add, Y to change, Z to destroy`, the blast radius, whether it is safe to apply, and what should be changed so the intended rename can be done safely.

```
Terraform will perform the following actions:

  # aws_cloudwatch_metric_alarm.db_cpu will be updated in-place
  ~ resource "aws_cloudwatch_metric_alarm" "db_cpu" {
      ~ dimensions = {
          ~ "DBInstanceIdentifier" = "orders-prod" -> "orders-prod-v2"
        }
        id         = "orders-prod-cpu"
    }

  # aws_db_instance.main must be replaced
-/+ resource "aws_db_instance" "main" {
      ~ address                  = "orders-prod.cxyz.eu-west-1.rds.amazonaws.com" -> (known after apply)
        allocated_storage        = 500
        deletion_protection      = false
        engine                   = "postgres"
      ~ id                       = "db-ABCDEFGHIJKLMNOP" -> (known after apply)
      ~ identifier               = "orders-prod" -> "orders-prod-v2" # forces replacement
        instance_class           = "db.r6g.2xlarge"
        skip_final_snapshot      = true
        # (41 unchanged attributes hidden)
    }

  # aws_db_parameter_group.main will be updated in-place
  ~ resource "aws_db_parameter_group" "main" {
        id   = "orders-pg15"
      + parameter {
          + apply_method = "pending-reboot"
          + name         = "max_connections"
          + value        = "800"
        }
    }

  # aws_security_group.legacy_db will be destroyed
  - resource "aws_security_group" "legacy_db" {
      - id   = "sg-0123456789abcdef0"
      - name = "legacy-db"
    }

  # aws_security_group_rule.app_to_db will be created
  + resource "aws_security_group_rule" "app_to_db" {
      + from_port = 5432
      + to_port   = 5432
      + type      = "ingress"
    }

Plan: 2 to add, 2 to change, 2 to destroy.
```
