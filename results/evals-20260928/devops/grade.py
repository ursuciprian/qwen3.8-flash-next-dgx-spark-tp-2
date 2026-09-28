#!/usr/bin/env python3
"""Grade DevOps task responses with real validators plus deterministic rubric checks.

  uv run --with pyyaml python grade.py <build> [<build> ...]     (or: grade.py --selftest)

Validators used: terraform (fmt, init -backend=false, validate), kubeconform -strict, actionlint,
shellcheck, bash -n plus a functional test with a stub pg_dump, helm template, hadolint, promtool.
Every check is pass/fail; a task's score is passed/total and it is "clean" when every check passes.
Writes graded/<build>.json and prints a per-task table.
"""
import json, os, pathlib, re, shutil, subprocess, sys, tempfile

import yaml

HERE = pathlib.Path(__file__).resolve().parent
TASKS = HERE / "tasks"
K8S = "1.30.0"


# ---------- helpers ----------
def blocks(text):
    """Fenced blocks as (lang, code, header) where header is the '### name' line just before it."""
    out = []
    for m in re.finditer(r"(?:^|\n)(?:#+\s*`?([\w./-]+)`?\s*\n+)?```([\w+-]*)[^\n]*\n(.*?)\n```", text, re.S):
        out.append(((m.group(2) or "").lower(), m.group(3), m.group(1)))
    return out


def first(text, langs):
    for lang, code, _ in blocks(text):
        if lang in langs:
            return code
    b = blocks(text)
    return b[-1][1] if b else ""


def sh(cmd, cwd=None, inp=None, env=None, timeout=300):
    p = subprocess.run(cmd, cwd=cwd, input=inp, capture_output=True, text=True, timeout=timeout,
                       shell=isinstance(cmd, str), env=env)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def tmpdir():
    return pathlib.Path(tempfile.mkdtemp(prefix="devops-grade-"))


def ydocs(code):
    try:
        return [d for d in yaml.safe_load_all(code) if d is not None]
    except yaml.YAMLError:
        return []


def kubeconform(code, d):
    f = d / "manifest.yaml"
    f.write_text(code)
    rc, out = sh(["kubeconform", "-strict", "-summary", "-kubernetes-version", K8S, str(f)])
    return rc == 0 and "Invalid: 0" in out and "Errors: 0" in out, out


def qty(s):
    """Kubernetes quantity -> float (bytes or cores)."""
    s = str(s).strip()
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3}
    for u, f in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(u):
            return float(s[: -len(u)]) * f
    return float(s)


def terraform_checks(files, d):
    for name, code in files.items():
        (d / name).write_text(code)
    c = []
    rc, out = sh(["terraform", "fmt", "-check", "-diff"], cwd=d)
    c.append(("terraform fmt -check", rc == 0, out))
    rc, out = sh(["terraform", "init", "-backend=false", "-input=false", "-no-color"], cwd=d, timeout=600)
    rc2, out2 = sh(["terraform", "validate", "-no-color"], cwd=d) if rc == 0 else (1, out)
    c.append(("terraform validate", rc2 == 0, out2))
    return c


def rx(text, pattern, flags=re.I | re.S):
    return re.search(pattern, text, flags) is not None


# ---------- tasks ----------
def t01(resp):
    files = {}
    for lang, code, hdr in blocks(resp):
        if hdr and hdr.endswith(".tf"):
            files[os.path.basename(hdr)] = code
    d = tmpdir()
    c = [("four named files", set(files) >= {"versions.tf", "variables.tf", "main.tf", "outputs.tf"}, str(sorted(files)))]
    if not files:
        files = {"main.tf": "\n".join(code for lang, code, _ in blocks(resp) if lang in ("hcl", "terraform", "tf"))}
    c += terraform_checks(files, d)
    allc = "\n".join(files.values())
    pab = re.search(r'resource\s+"aws_s3_bucket_public_access_block".*?\n}', allc, re.S)
    c.append(("public access block: all 4 true", bool(pab) and all(
        rx(pab.group(0), rf"{k}\s*=\s*true") for k in
        ("block_public_acls", "block_public_policy", "ignore_public_acls", "restrict_public_buckets")), ""))
    c.append(("SSE aws:kms + kms key var + bucket key", rx(allc, r'sse_algorithm\s*=\s*"aws:kms"') and
              rx(allc, r"kms_master_key_id\s*=\s*var\.kms_key_arn") and rx(allc, r"bucket_key_enabled\s*=\s*true"), ""))
    c.append(("versioning Enabled", rx(allc, r'aws_s3_bucket_versioning') and rx(allc, r'status\s*=\s*"Enabled"'), ""))
    c.append(("ownership BucketOwnerEnforced", rx(allc, r'object_ownership\s*=\s*"BucketOwnerEnforced"'), ""))
    c.append(("lifecycle: noncurrent expiry var + abort MPU 7d",
              rx(allc, r"noncurrent_days\s*=\s*var\.noncurrent_days") and
              rx(allc, r"abort_incomplete_multipart_upload\s*{[^}]*days_after_initiation\s*=\s*7"), ""))
    c.append(("TLS-only deny policy", rx(allc, r"aws:SecureTransport") and rx(allc, r'"?Deny"?'), ""))
    c.append(("outputs bucket_id + bucket_arn", rx(allc, r'output\s+"bucket_id"') and rx(allc, r'output\s+"bucket_arn"'), ""))
    c.append(("no deprecated inline blocks/acl", not re.search(
        r'resource\s+"aws_s3_bucket"\s+"\w+"\s*{[^}]*(versioning\s*{|acl\s*=|server_side_encryption_configuration\s*{|lifecycle_rule\s*{)',
        allc, re.S), ""))
    return c


def t02(resp):
    code = first(resp, ("hcl", "terraform", "tf"))
    d = tmpdir()
    c = terraform_checks({"main.tf": code}, d)
    c.append(("cidrsubnet uses count.index", rx(code, r"cidrsubnet\([^)]*count\.index\s*\)"), ""))
    c.append(("vpc_id = aws_vpc.main.id", rx(code, r"vpc_id\s*=\s*aws_vpc\.main\.id") and "main.vpc_id" not in code, ""))
    c.append(("output uses splat/for over subnets", rx(code, r"aws_subnet\.private\[\*\]\.id|for\s+\w+\s+in\s+aws_subnet\.private"), ""))
    c.append(("SG cidr_blocks = subnet CIDR list", rx(code, r"cidr_blocks\s*=\s*(aws_subnet\.private\[\*\]\.cidr_block|\[\s*for\s+\w+\s+in\s+aws_subnet\.private)"), ""))
    c.append(("explains errors (>=4 listed)", len(re.findall(r"^\s*(?:\d+[.)]|[-*])\s", resp.split("```")[0], re.M)) >= 4, ""))
    return c


def t03(resp):
    code = first(resp, ("yaml", "yml"))
    d = tmpdir()
    ok, out = kubeconform(code, d)
    c = [("kubeconform -strict (k8s 1.30)", ok, out)]
    docs = ydocs(code)
    dep = next((x for x in docs if isinstance(x, dict) and x.get("kind") == "Deployment"), {}) or {}
    spec = dep.get("spec", {}) or {}
    sel = ((spec.get("selector") or {}).get("matchLabels") or {})
    lbl = (((spec.get("template") or {}).get("metadata") or {}).get("labels") or {})
    ctr = ((((spec.get("template") or {}).get("spec") or {}).get("containers") or [{}])[0]) or {}
    c.append(("apiVersion apps/v1", dep.get("apiVersion") == "apps/v1", ""))
    c.append(("selector matches template labels", bool(sel) and all(lbl.get(k) == v for k, v in sel.items()), f"{sel} vs {lbl}"))
    ports = ctr.get("ports") or [{}]
    c.append(("containerPort int 8080", ports[0].get("containerPort") == 8080, ""))
    env = {e.get("name"): e.get("value") for e in ctr.get("env") or []}
    c.append(("DB_PORT value is a string", env.get("DB_PORT") == "5432", repr(env.get("DB_PORT"))))
    probe_port = (((ctr.get("readinessProbe") or {}).get("httpGet")) or {}).get("port")
    named = {p.get("name"): p.get("containerPort") for p in ports if p.get("name")}
    c.append(("readiness probe on 8080", probe_port == 8080 or named.get(probe_port) == 8080, repr(probe_port)))
    res = ctr.get("resources") or {}
    try:
        rq, lm = res.get("requests") or {}, res.get("limits") or {}
        ok = all(qty(lm[k]) >= qty(rq[k]) for k in ("cpu", "memory") if k in lm and k in rq) and bool(rq)
    except Exception as e:
        ok = False
    c.append(("limits >= requests", ok, json.dumps(res)))
    return c


def t04(resp):
    code = first(resp, ("yaml", "yml"))
    c = [("names OOMKilled / exit 137", rx(resp, r"OOM") or rx(resp, r"\b137\b"), ""),
         ("heap (-Xmx 1g) vs 512Mi limit identified", rx(resp, r"Xmx") and rx(resp, r"512\s*Mi"), "")]
    docs = ydocs(code)
    ctr = None
    for x in docs:
        cand = x
        if isinstance(cand, dict) and "containers" in cand:
            cand = cand["containers"]
        if isinstance(cand, list) and cand:
            cand = cand[0]
        if isinstance(cand, dict) and "resources" in cand:
            ctr = cand
    c.append(("container spec YAML parses", ctr is not None, ""))
    ok, detail = False, ""
    if ctr:
        try:
            lim = qty(ctr["resources"]["limits"]["memory"])
            req = qty(ctr["resources"].get("requests", {}).get("memory", "0"))
            envtxt = json.dumps(ctr.get("env", []))
            m = re.search(r"-Xmx(\d+)([gGmM])", envtxt)
            heap = (int(m.group(1)) * (2**30 if m.group(2) in "gG" else 2**20)) if m else None
            pct = re.search(r"MaxRAMPercentage=(\d+(?:\.\d+)?)", envtxt)
            if heap:
                ok = heap <= 0.8 * lim
            elif pct:
                ok = float(pct.group(1)) <= 80
            detail = f"limit={lim/2**20:.0f}Mi heap={heap and heap/2**20} pct={pct and pct.group(1)}"
            c.append(("request == limit (memory)", req == lim, detail))
        except Exception as e:
            detail = repr(e)
            c.append(("request == limit (memory)", False, detail))
    else:
        c.append(("request == limit (memory)", False, ""))
    c.append(("heap fits in limit with >=20% headroom", ok, detail))
    return c


def t05(resp):
    code = first(resp, ("yaml", "yml"))
    d = tmpdir()
    (d / ".github/workflows").mkdir(parents=True)
    f = d / ".github/workflows/ci.yml"
    f.write_text(code)
    rc, out = sh(["actionlint", "-no-color", str(f)], cwd=d)
    c = [("actionlint clean", rc == 0, out)]
    doc = (ydocs(code) or [{}])[0] or {}
    jobs = doc.get("jobs") or {}
    test, dep = jobs.get("test") or {}, jobs.get("deploy") or {}
    mx = ((test.get("strategy") or {}).get("matrix") or {})
    vals = next((v for v in mx.values() if isinstance(v, list)), [])
    c.append(("matrix versions quoted strings", [str(v) for v in vals] == ["3.10", "3.11", "3.12"] and all(isinstance(v, str) for v in vals), repr(vals)))
    c.append(("test job has runs-on", "runs-on" in test, ""))
    c.append(("uses $GITHUB_OUTPUT, no set-output", "GITHUB_OUTPUT" in code and "::set-output" not in code, ""))
    c.append(("no secrets in if:", not rx(str(dep.get("if", "")), r"secrets\."), ""))
    c.append(("secret passed via env, not argv", not rx(code, r"run:[^\n]*\$\{\{\s*secrets\.", re.I), ""))
    c.append(("permissions block present", "permissions" in doc or any("permissions" in (j or {}) for j in jobs.values()), ""))
    c.append(("deploy only on push to main", rx(str(dep.get("if", "")), r"refs/heads/main") and rx(str(dep.get("if", "")) + code, r"push"), str(dep.get("if"))))
    return c


def t06(resp):
    code = first(resp, ("yaml", "yml"))
    d = tmpdir()
    (d / ".github/workflows").mkdir(parents=True)
    f = d / ".github/workflows/terraform.yml"
    f.write_text(code)
    rc, out = sh(["actionlint", "-no-color", str(f)], cwd=d)
    c = [("actionlint clean", rc == 0, out)]
    c.append(("id-token: write", rx(code, r"id-token:\s*write"), ""))
    c.append(("contents: read (no write-all)", rx(code, r"contents:\s*read") and not rx(code, r"write-all"), ""))
    c.append(("configure-aws-credentials with role-to-assume", rx(code, r"aws-actions/configure-aws-credentials@") and
              rx(code, r"role-to-assume:\s*['\"]?arn:aws:iam::123456789012:role/gha-terraform"), ""))
    c.append(("no static AWS keys", not rx(code, r"aws-access-key-id|aws-secret-access-key|AWS_SECRET_ACCESS_KEY"), ""))
    c.append(("plan on PRs", rx(code, r"pull_request") and rx(code, r"terraform\s+plan"), ""))
    c.append(("apply gated: environment production + main push", rx(code, r"environment:\s*(\n\s*name:\s*)?['\"]?production") and
              rx(code, r"terraform\s+apply") and rx(code, r"refs/heads/main|github\.event_name\s*==\s*'push'"), ""))
    c.append(("concurrency guard", rx(code, r"concurrency:"), ""))
    c.append(("non-interactive (-input=false)", rx(code, r"-input=false"), ""))
    return c


def t07(resp):
    code = first(resp, ("json",))
    c = []
    try:
        pol = json.loads(code)
        st = pol["Statement"] if isinstance(pol["Statement"], list) else [pol["Statement"]]
        c.append(("policy JSON valid", True, ""))
    except Exception as e:
        return [("policy JSON valid", False, repr(e))] + [(n, False, "") for n in (
            "no service wildcards", "no Resource *", "no iam:PassRole", "s3 read scoped to incoming/", "s3 write scoped to acme-thumbs",
            "dynamodb scoped to table", "kms scoped to key", "logs scoped to function", "findings flag PassRole escalation")]
    lst = lambda v: v if isinstance(v, list) else [v]
    allow = [s for s in st if s.get("Effect") == "Allow"]
    acts = [a for s in allow for a in lst(s.get("Action", []))]
    res = [r for s in allow for r in lst(s.get("Resource", []))]
    def has(action_rx, res_rx):
        return any(any(re.fullmatch(action_rx, a, re.I) for a in lst(s.get("Action", []))) and
                   all(re.search(res_rx, r) for r in lst(s.get("Resource", []))) for s in allow)
    c.append(("no service wildcards", not any(a.endswith(":*") or a == "*" for a in acts), str(acts)))
    c.append(("no Resource *", "*" not in res, ""))
    c.append(("no iam:PassRole", not any(a.lower().startswith("iam:") for a in acts), ""))
    c.append(("s3 read scoped to incoming/", has(r"s3:GetObject", r"^arn:aws:s3:::acme-uploads/incoming/\*$"), ""))
    c.append(("s3 write scoped to acme-thumbs", has(r"s3:PutObject", r"^arn:aws:s3:::acme-thumbs/"), ""))
    c.append(("dynamodb scoped to table", has(r"dynamodb:PutItem", r"^arn:aws:dynamodb:eu-west-1:123456789012:table/thumbnails$"), ""))
    c.append(("kms scoped to key", has(r"kms:(Decrypt|GenerateDataKey)", r"^arn:aws:kms:eu-west-1:123456789012:key/0a1b2c3d"), ""))
    c.append(("logs scoped to function", has(r"logs:(PutLogEvents|CreateLogStream)", r"log-group:/aws/lambda/thumbnailer"), ""))
    c.append(("findings flag PassRole escalation", rx(resp, r"PassRole.{0,200}(escalat|privilege)|(escalat|privilege).{0,200}PassRole"), ""))
    return c


def t08(resp):
    code = first(resp, ("bash", "sh", "shell"))
    d = tmpdir()
    s = d / "pg-backup.sh"
    s.write_text(code)
    s.chmod(0o755)
    rc, out = sh(["shellcheck", "-S", "style", str(s)])
    c = [("shellcheck clean", rc == 0, out)]
    rc, out = sh(["bash", "-n", str(s)])
    c.append(("bash -n", rc == 0, out))
    c.append(("set -euo pipefail", rx(code, r"set -[a-z]*e[a-z]*u[a-z]*o pipefail|set -euo pipefail|set -o errexit.*nounset.*pipefail"), ""))
    stub = d / "bin"
    stub.mkdir()
    (stub / "pg_dump").write_text('#!/bin/bash\n[ "${FAIL_DUMP:-0}" = 1 ] && { echo boom >&2; exit 3; }\necho "-- dump of $*"\n')
    (stub / "pg_dump").chmod(0o755)
    env = dict(os.environ, PATH=f"{stub}:{os.environ['PATH']}")
    out_dir = d / "out"
    rc_h, _ = sh([str(s), "--help"], env=env, timeout=30)
    rc_bad, _ = sh([str(s), "--bogus"], env=env, timeout=30)
    rc_nodb, _ = sh([str(s), "--out-dir", str(out_dir)], env=env, timeout=30)
    c.append(("--help exits 0; bad option / missing --db exit 2", rc_h == 0 and rc_bad == 2 and rc_nodb == 2, f"{rc_h},{rc_bad},{rc_nodb}"))
    rc_dry, _ = sh([str(s), "--db", "shop", "--out-dir", str(out_dir), "--dry-run"], env=env, timeout=30)
    c.append(("--dry-run writes nothing", rc_dry == 0 and not any(out_dir.glob("*")), ""))
    import time
    ok_runs = True
    for _ in range(5):
        r, _o = sh([str(s), "--db", "shop", "--out-dir", str(out_dir), "--keep", "3"], env=env, timeout=30)
        ok_runs &= r == 0
        time.sleep(1.1)
    files = sorted(out_dir.glob("shop-*.sql.gz")) if out_dir.exists() else []
    gz_ok = bool(files) and sh(f"gzip -dc '{files[-1]}' | grep -q 'dump of'")[0] == 0
    c.append(("real run: gzip dump named NAME-ts.sql.gz", ok_runs and gz_ok and all(
        re.fullmatch(r"shop-\d{8}T\d{6}\.sql\.gz", f.name) for f in files), str([f.name for f in files])))
    c.append(("--keep 3 retention after 5 runs", len(files) == 3, str(len(files))))
    before = set(out_dir.glob("*")) if out_dir.exists() else set()
    r_fail, _ = sh([str(s), "--db", "shop", "--out-dir", str(out_dir)], env=dict(env, FAIL_DUMP="1"), timeout=30)
    after = set(out_dir.glob("*")) if out_dir.exists() else set()
    c.append(("failed dump: non-zero exit, no partial file", r_fail != 0 and after == before, f"rc={r_fail} new={[p.name for p in after - before]}"))
    shutil.rmtree(d, ignore_errors=True)
    return c


def t09(resp):
    code = first(resp, ("yaml", "yml"))
    d = tmpdir()
    shutil.copytree(TASKS / "09-helm-values-fix/chart/webapp", d / "webapp")
    (d / "values.yaml").write_text(code)
    rc, out = sh(["helm", "template", "webapp", str(d / "webapp"), "-f", str(d / "values.yaml")])
    c = [("helm template renders", rc == 0, out[-500:])]
    rendered = out if rc == 0 else ""
    ok, kout = kubeconform(rendered, d) if rendered else (False, "")
    c.append(("kubeconform -strict on render", ok, kout))
    docs = ydocs(rendered)
    kinds = {x.get("kind"): x for x in docs if isinstance(x, dict)}
    dep, svc, ing = kinds.get("Deployment", {}), kinds.get("Service", {}), kinds.get("Ingress", {})
    try:
        ctr = dep["spec"]["template"]["spec"]["containers"][0]
    except Exception:
        ctr = {}
    c.append(("image tag exactly 2.40", ctr.get("image") == "ghcr.io/acme/webapp:2.40", str(ctr.get("image"))))
    try:
        port = svc["spec"]["ports"][0]["port"]
    except Exception:
        port = None
    c.append(("service port 80", port == 80, repr(port)))
    r = ctr.get("resources") or {}
    c.append(("resources limits+requests as asked", (r.get("limits") or {}) == {"cpu": "500m", "memory": "256Mi"} and
              (r.get("requests") or {}) == {"cpu": "250m", "memory": "128Mi"}, json.dumps(r)))
    try:
        rule = ing["spec"]["rules"][0]
        path = rule["http"]["paths"][0]
        ok = rule.get("host") == "webapp.example.com" and path.get("pathType") == "Prefix"
    except Exception:
        ok = False
    c.append(("ingress host list + pathType Prefix", ok, ""))
    return c


def t10(resp):
    code = first(resp, ("dockerfile", "docker"))
    d = tmpdir()
    f = d / "Dockerfile"
    f.write_text(code)
    rc, out = sh(["hadolint", "--no-color", "--failure-threshold", "warning", "--ignore", "DL3008", "--ignore", "DL3018", str(f)])
    c = [("hadolint (no warnings/errors; DL3008/DL3018 ignored)", rc == 0, out)]
    froms = re.findall(r"^FROM\s+(\S+)", code, re.M | re.I)
    c.append(("multi-stage", len(froms) >= 2, str(froms)))
    aliases = {a.lower() for a in re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", code, re.M | re.I)}
    bases = [x for x in froms if x.lower() not in aliases]
    c.append(("pinned base (tag or digest, not latest)", bool(bases) and all(
        ("@sha256:" in x or ":" in x.split("/")[-1]) and not x.endswith(":latest") for x in bases), str(bases)))
    last = code[code.upper().rfind("FROM "):] if froms else ""
    c.append(("non-root USER in final stage", rx(last, r"^USER\s+(?!root\b|0\b)\S+", re.M | re.I), ""))
    c.append(("npm token via BuildKit secret", rx(code, r"--mount=type=secret,[^\n]*id=npm_token") and "npm_abc123secret" not in code
              and not rx(code, r"^\s*(ENV|ARG)\s+NPM_TOKEN", re.M | re.I), ""))
    c.append(("npm ci + production deps", rx(code, r"npm ci") and rx(code, r"--omit=dev|--only=production|--production|npm prune"), ""))
    c.append(("exec-form CMD node dist/server.js", rx(code, r'^CMD\s*\[\s*"node"\s*,\s*"dist/server\.js"\s*\]', re.M), ""))
    c.append(("no remote ADD, no curl install", not rx(code, r"^ADD\s+https?://", re.M | re.I) and not rx(code, r"install[^\n]*\bcurl\b"), ""))
    return c


def t11(resp):
    return [
        ("counts 2 add / 2 change / 2 destroy", rx(resp, r"2 to add,\s*2 to change,\s*2 to destroy"), ""),
        ("RDS instance replaced = destroy + recreate", rx(resp, r"(replac|destroy|recreat)") and rx(resp, r"aws_db_instance|RDS|database"), ""),
        ("flags data loss: skip_final_snapshot true + deletion_protection false", rx(resp, r"skip_final_snapshot") and rx(resp, r"deletion_protection"), ""),
        ("verdict: do not apply", rx(resp, r"do not apply|don.t apply|not safe|unsafe|NO[- ]GO|block(ed)? (the )?apply|must not be applied|should not be applied"), ""),
        ("safe path: revert identifier / snapshot / prevent_destroy", rx(resp, r"prevent_destroy") and rx(resp, r"snapshot") and rx(resp, r"revert|keep the (old|existing) identifier|identifier\s*=\s*\"orders-prod\"|modify|rename in place|apply_immediately|AWS (console|CLI)|modify-db-instance"), ""),
        ("notes endpoint/address change breaks clients", rx(resp, r"(endpoint|address|DNS|hostname).{0,120}(chang|break|new)|(chang|new).{0,80}(endpoint|address|hostname)"), ""),
        ("notes max_connections needs reboot", rx(resp, r"pending-reboot|reboot"), ""),
    ]


def t12(resp):
    code = first(resp, ("yaml", "yml"))
    d = tmpdir()
    ok, out = kubeconform(code, d)
    c = [("kubeconform -strict", ok, out)]
    pols = [x for x in ydocs(code) if isinstance(x, dict) and x.get("kind") == "NetworkPolicy"]
    for p in pols:
        p.setdefault("spec", {})
    c.append(("all in namespace shop", bool(pols) and all((p.get("metadata") or {}).get("namespace") == "shop" for p in pols), ""))
    deny = [p for p in pols if (p["spec"].get("podSelector") in ({}, None)) and set(p["spec"].get("policyTypes") or []) >= {"Ingress", "Egress"}
            and not p["spec"].get("ingress") and not p["spec"].get("egress")]
    c.append(("default deny ingress+egress", bool(deny), ""))
    chk = [p for p in pols if ((p["spec"].get("podSelector") or {}).get("matchLabels") or {}).get("app") == "checkout"]
    ing = [r for p in chk for r in p["spec"].get("ingress") or []]
    egr = [r for p in chk for r in p["spec"].get("egress") or []]
    ns = lambda peer, n: ((peer.get("namespaceSelector") or {}).get("matchLabels") or {}).get("kubernetes.io/metadata.name") == n
    pod = lambda peer, k, v: ((peer.get("podSelector") or {}).get("matchLabels") or {}).get(k) == v
    ports = lambda r: {(q.get("protocol", "TCP"), q.get("port")) for q in r.get("ports") or []}
    c.append(("ingress 8080 only from ingress-nginx", any(r.get("from") and all(ns(f, "ingress-nginx") for f in r["from"]) and ports(r) == {("TCP", 8080)} for r in ing)
              and all(all(ns(f, "ingress-nginx") for f in r.get("from") or [{}]) for r in ing), ""))
    c.append(("egress 5432 to app=postgres AND ns data (same peer)", any(any(ns(f, "data") and pod(f, "app", "postgres") for f in r.get("to") or [])
              and ports(r) == {("TCP", 5432)} for r in egr) and not any(any(ns(f, "data") and "podSelector" not in f for f in r.get("to") or []) for r in egr), ""))
    c.append(("DNS UDP+TCP 53 to kube-dns in kube-system", any(any(ns(f, "kube-system") and pod(f, "k8s-app", "kube-dns") for f in r.get("to") or [])
              and {("UDP", 53), ("TCP", 53)} <= ports(r) for r in egr), ""))
    c.append(("checkout policies declare policyTypes", bool(chk) and all(p["spec"].get("policyTypes") for p in chk), ""))
    return c


def t13(resp):
    code = first(resp, ("yaml", "yml"))
    d = tmpdir()
    f = d / "rules.yaml"
    f.write_text(code)
    rc, out = sh(["promtool", "check", "rules", str(f)])
    c = [("promtool check rules", rc == 0, out)]
    rules = [r for g in ((ydocs(code) or [{}])[0] or {}).get("groups", []) for r in g.get("rules", [])]
    rec = next((r for r in rules if r.get("record") == "service:http_requests_5xx:ratio_rate5m"), {})
    e = rec.get("expr", "")
    c.append(("recording rule: ratio of rates by service", bool(rec) and rx(e, r'status\s*=~\s*"5(\.\.|\[0-9\]\{2\}|\d\d|xx)"') and
              len(re.findall(r"rate\(\s*http_requests_total", e)) >= 2 and len(re.findall(r"by\s*\(\s*service\s*\)", e)) >= 2, e))
    her = next((r for r in rules if r.get("alert") == "HighErrorRate"), {})
    c.append(("HighErrorRate > 0.05 for 10m, severity page", rx(str(her.get("expr", "")), r"ratio_rate5m\s*>\s*0?\.05") and
              str(her.get("for")) == "10m" and (her.get("labels") or {}).get("severity") == "page", ""))
    c.append(("summary names service + percent", rx(str((her.get("annotations") or {}).get("summary", "")), r"\$labels\.service") and
              rx(str((her.get("annotations") or {}).get("summary", "")), r"humanizePercentage|\* ?100"), ""))
    pcl = next((r for r in rules if r.get("alert") == "PodCrashLooping"), {})
    c.append(("PodCrashLooping increase[15m] > 3, severity ticket", rx(str(pcl.get("expr", "")), r"increase\(\s*kube_pod_container_status_restarts_total[^\[]*\[15m\]\s*\)\s*>\s*3")
              and (pcl.get("labels") or {}).get("severity") == "ticket", str(pcl.get("expr"))))
    return c


def t14(resp):
    return [
        ("12 x 50 = 600 > 500 max_connections", rx(resp, r"600") and rx(resp, r"500") and rx(resp, r"12\s*(pods|replicas|×|x|\*)"), ""),
        ("trigger = 14:02 scale 4 -> 12", rx(resp, r"14:02") and rx(resp, r"4\s*(->|→|to)\s*12"), ""),
        ("not DB capacity (CPU 31%)", rx(resp, r"31\s*%|CPU.{0,40}(normal|fine|not|low|healthy)"), ""),
        ("mitigation: cut pool size or scale back", rx(resp, r"(pool|MAXIMUM_POOL_SIZE).{0,80}(\b[1-3]\d\b|reduc|lower|decreas)|(scale|roll).{0,40}back|replicas.{0,20}\b4\b"), ""),
        ("long-term: pooler (PgBouncer / RDS Proxy) or connection budget", rx(resp, r"pgbouncer|rds proxy|connection budget|pooler"), ""),
    ]


GRADERS = {"01": t01, "02": t02, "03": t03, "04": t04, "05": t05, "06": t06, "07": t07,
           "08": t08, "09": t09, "10": t10, "11": t11, "12": t12, "13": t13, "14": t14}


def grade_build(build):
    rows = []
    for f in sorted((HERE / "responses" / build).glob("*.json")):
        r = json.loads(f.read_text())
        try:
            if not r["content"].strip():  # no answer (runaway thinking): nothing to grade, every check fails
                checks = [("answer present (runaway thinking)", False, "no answer")]
            else:
                checks = GRADERS[r["task"][:2]](r["content"])
        except Exception as e:
            checks = [("grader exception", False, repr(e))]
        passed = sum(1 for _, ok, _ in checks if ok)
        rows.append(dict(task=r["task"], repeat=r["repeat"], passed=passed, total=len(checks), clean=passed == len(checks),
                         total_s=r["total_s"], think_s=r["t_first_answer_s"], finish=r["finish_reason"],
                         runaway=r["finish_reason"] == "length" and not r["content"].strip(),
                         truncated_answer=r["finish_reason"] == "length" and bool(r["content"].strip()),
                         completion_tokens=(r.get("usage") or {}).get("completion_tokens"),
                         reasoning_tokens=((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                         checks=[dict(name=n, ok=ok, detail=det[-400:] if not ok else "") for n, ok, det in checks]))
    out = HERE / "graded"
    out.mkdir(exist_ok=True)
    (out / f"{build}.json").write_text(json.dumps(rows, indent=1))
    (out / f"{build}.md").write_text(markdown(rows))
    return rows


def markdown(rows):
    """Per-task table: mean check score, clean runs, runaways, median/max wall-clock, median thinking time."""
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    lines = ["| Task | Mean checks passed | Clean runs | Runaway thinking | Median time (s) | Max time (s) | Median time to answer (s) | Median completion tokens | Failed checks (all runs) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == t]
        fails = sorted({c["name"] for r in rs for c in r["checks"] if not c["ok"]})
        think = [r["think_s"] for r in rs if r["think_s"] is not None]
        lines.append(f"| {t} | {sum(r['passed'] / r['total'] for r in rs) / len(rs) * 100:.0f}% | {sum(r['clean'] for r in rs)}/{len(rs)} | "
                     f"{sum(r['runaway'] for r in rs)}/{len(rs)} | {med([r['total_s'] for r in rs]):.0f} | {max(r['total_s'] for r in rs):.0f} | "
                     f"{med(think) if think else '-'} | {med([r['completion_tokens'] or 0 for r in rs])} | {'; '.join(fails) or '-'} |")
    n = len(rows)
    ra = sum(r["runaway"] for r in rows)
    lines.append(f"| **All ({n} runs)** | **{sum(r['passed'] / r['total'] for r in rows) / max(n, 1) * 100:.1f}%** | "
                 f"**{sum(r['clean'] for r in rows)}/{n}** | **{ra}/{n} ({100 * ra / max(n, 1):.0f}%)** | "
                 f"{med([r['total_s'] for r in rows]):.0f} | {max(r['total_s'] for r in rows):.0f} | | | |")
    return "\n".join(lines) + "\n"


def selftest():
    good = '### main.tf\n```hcl\nx\n```\n\n```yaml\na: 1\n```'
    b = blocks(good)
    assert b[0] == ("hcl", "x", "main.tf") and b[1][0] == "yaml", b
    assert qty("512Mi") == 512 * 2**20 and qty("250m") == 0.25 and qty("1") == 1.0
    assert t14("12 pods x 50 = 600 > 500; at 14:02 scaled 4 -> 12; CPU 31%; reduce pool to 30; add PgBouncer")[0][1]
    print("grade selftest OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit()
    for b in sys.argv[1:]:
        rows = grade_build(b)
        ra = sum(r['runaway'] for r in rows)
        print(f"== {b}: runaway thinking {ra}/{len(rows)} ({100 * ra / max(len(rows), 1):.0f}%), "
              f"truncated answers {sum(r['truncated_answer'] for r in rows)}")
        print(f"== {b}: {sum(r['clean'] for r in rows)}/{len(rows)} clean, "
              f"mean score {sum(r['passed'] / r['total'] for r in rows) / max(len(rows), 1):.3f}")
        for r in rows:
            fails = "; ".join(c["name"] for c in r["checks"] if not c["ok"])
            print(f"  {r['task']:<26} r{r['repeat']} {r['passed']:>2}/{r['total']:<2} {r['total_s']:>7.1f}s  {fails}")
