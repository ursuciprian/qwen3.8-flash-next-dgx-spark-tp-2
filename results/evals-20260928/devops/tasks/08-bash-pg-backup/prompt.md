Write a production-quality bash script `pg-backup.sh`.

Behaviour:
- `--db NAME` (required): database to dump with `pg_dump NAME` (connection settings come from the usual PG* environment variables).
- `--out-dir DIR` (default `/var/backups/pg`): creates it if missing and writes the dump there, gzip-compressed, as `NAME-YYYYmmddTHHMMSS.sql.gz`.
- `--keep N` (default 7): after a successful backup, keep only the newest N backups of that database in DIR and delete the older ones.
- `--dry-run`: print what would be done, but create, write and delete nothing.
- `-h`/`--help` prints usage and exits 0; an unknown option or a missing `--db` prints usage to stderr and exits 2.
- A failed dump must not leave a partial file behind, and must make the script exit non-zero (the pipeline must not hide a `pg_dump` failure).
- Log lines go to stderr with a timestamp.

Return only the script, in one ```bash fenced block. It must pass `shellcheck` with no findings.
