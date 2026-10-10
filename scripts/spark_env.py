"""Addresses of the Spark pair for the Python scripts in this folder; see spark_env.sh for the variables.

A SPARK_* value comes from the environment, else from ${SPARK_PAIR_ENV:-~/.config/spark-pair.env}."""
import os
import sys

ENV_FILE = os.environ.get("SPARK_PAIR_ENV") or os.path.expanduser("~/.config/spark-pair.env")


def spark(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        for line in open(ENV_FILE):
            k, _, v = line.strip().partition("=")
            if k == name and v:
                return v
    except FileNotFoundError:
        pass
    sys.exit(f"error: {name} is not set; export it or add {name}=<value> to {ENV_FILE} "
             "(template: scripts/spark-pair.env.example)")
