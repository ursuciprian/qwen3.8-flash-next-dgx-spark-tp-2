#!/usr/bin/env python3
"""Thunderdome verdict for one Spark (#83): arm vs control, paired boots on the same node.

Usage: thunderdome_report.py <node dir> [--acc-only]
       thunderdome_report.py --selftest

<node dir> holds ctl-p1, arm-p1, arm-p2, ctl-p2 from scripts/thunderdome.sh. Pass K pairs arm-pK with
ctl-pK (same prompts, same Spark). Fixed rules:

  probe cells   fresh-c1 fresh-c4 fresh-c8 d16k-c4 count-c8 (depth_decode_probe.py, T=0)
                delta = mean paired decode tok/s change (paired_decode_ab.compare, stalls excluded)
                noise = max(1 %, |control pass 2 vs pass 1 mean change|, half-width of the delta's 95% CI)
  wall cell     d16k-c8 (1 rep, cut at 480 s): the 6 GiB KV pool preempts and recomputes there, so per-request
                tok/s means nothing; speed = 1 / wall time of the cell. A run cut by the timeout counts as
                the cut; an errored run is missing
                delta = mean over passes of (control wall / arm wall - 1)
                noise = max(1 %, |control boot 2 vs boot 1 wall change|)
  benchy cells  pp2048 (c1), tg512 (c1), tg512 (c8) (llama-benchy t/s total)
                delta = arm mean vs control mean
                noise = max(1 %, control spread between boots, mean reported sd), in % of the control mean
  acceptance    accepted/drafted per draft position, pooled over the probe cells both sides finished

  KILL          a cell worse than 2x its noise, acceptance moved by more than 0.03 at any shared position,
                an arm boot failed, the arm server died during a boot (DIED), or an arm cell did not finish
                (timeout or error) in a pass where the control's did
  PROMOTE       every cell measured, none worse than 1x noise, at least one better than 1x noise
  INCONCLUSIVE  anything else: control boot failed, a cold boot, missing cells, a cell worse than
                1x but not 2x noise, or nothing better than noise

Logits (logits_equiv.py captured twice per boot) are reported, not judged.
--acc-only checks acceptance on the boots done so far and exits 3 when that alone means KILL.
Last line: VERDICT=KILL|PROMOTE|INCONCLUSIVE.
"""
import glob, json, os, re, shutil, statistics, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_decode_ab import compare, load  # noqa: E402

NOISE_FLOOR, KILL_X, ACC_TOL = 1.0, 2.0, 0.03
PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "count-c8")
WALL = ("d16k-c8",)
BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c8)")
ROW = re.compile(r"^\|\s*[^|]+\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)")
POS = re.compile(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)')


def boots(d, who):
    """{K: dir} for <who>-pK boots that came up (no FAILED / COLD marker)."""
    return {int(p[-1]): p for p in glob.glob(os.path.join(d, f"{who}-p[0-9]"))
            if not any(os.path.exists(os.path.join(p, m)) for m in ("FAILED", "COLD"))}


def marked(d, marker):
    return sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(d, "*-p[0-9]", marker)))


def probe(p, cell):
    f = os.path.join(p, f"probe-{cell}.json")
    return load(f, "inf") if os.path.exists(f) else None


def probe_cell(C, A, cell):
    ctl = {k: probe(p, cell) for k, p in C.items()}
    arm = {k: probe(p, cell) for k, p in A.items()}
    a, b = {}, {}
    for k in sorted(set(ctl) & set(arm)):
        if ctl[k] and arm[k]:
            a.update({(k, i): v for i, v in ctl[k].items()})
            b.update({(k, i): v for i, v in arm[k].items()})
    ok = [k for k in sorted(ctl) if ctl[k]]
    if not a or len(ok) < 2:
        return None
    res, own = compare(a, b), compare(ctl[ok[0]], ctl[ok[1]]).get("tok_s")
    r = res.get("tok_s")
    if not r or not own:
        return None
    lo, hi = r["ci95"]
    noise = max(NOISE_FLOOR, abs(own["mean_pct"]), (hi - lo) / 2)
    st = res.get("step_ms")
    info = f"n={r['n']} CI [{lo:+.1f},{hi:+.1f}] control boot-to-boot {own['mean_pct']:+.2f}%"
    return r["mean_pct"], noise, info + (f", step {st['mean_pct']:+.2f}%" if st else "")


def timing(p, cell):
    """(rc, seconds, cut) from time-<cell>.txt, or None."""
    try:
        rc, w, cut = open(os.path.join(p, f"time-{cell}.txt")).read().split()[:3]
        return int(rc), float(w), float(cut)
    except (OSError, ValueError):
        return None


def wall(p, cell):
    """Wall seconds of a cell; a run cut by `timeout` (124/137) counts as the cut, an errored run as missing."""
    t = timing(p, cell)
    if t is None:
        return None
    rc, w, cut = t
    return cut if rc in (124, 137) else (None if rc else min(w, cut))


def arm_unfinished(C, A, cell):
    """Passes where the arm's cell timed out or failed while the control's finished."""
    out = []
    for k in sorted(set(C) & set(A)):
        c, a = timing(C[k], cell), timing(A[k], cell)
        # on the wall cell a timeout is a measured outcome (scored as the cut), not a failure
        if c and a and c[0] == 0 and a[0] != 0 and not (cell in WALL and a[0] in (124, 137)):
            out.append(k)
    return out


def wall_cell(C, A, cell):
    ctl = {k: wall(p, cell) for k, p in C.items()}
    arm = {k: wall(p, cell) for k, p in A.items()}
    pairs = [(ctl[k], arm[k]) for k in sorted(set(ctl) & set(arm)) if ctl[k] and arm[k]]
    own = [ctl[k] for k in sorted(ctl) if ctl[k]]
    if not pairs or len(own) < 2:
        return None
    delta = statistics.mean((x / y - 1) * 100 for x, y in pairs)
    noise = max(NOISE_FLOOR, abs(own[1] / own[0] - 1) * 100)
    show = lambda d: "/".join(f"{d[k]:.0f}" if d[k] else "-" for k in sorted(d))
    return delta, noise, f"wall s control {show(ctl)} arm {show(arm)} (cut counts as the time)"


def benchy(p):
    try:
        return {m.group(1): (float(m.group(2)), float(m.group(3))) for m in map(ROW.match, open(os.path.join(p, "task.csv"))) if m}
    except OSError:
        return {}


def benchy_cell(C, A, cell):
    c = [r[cell] for r in map(benchy, C.values()) if cell in r]
    a = [r[cell][0] for r in map(benchy, A.values()) if cell in r]
    if len(c) < 2 or not a:
        return None
    cm, am = statistics.mean(x for x, _ in c), statistics.mean(a)
    noise = max(NOISE_FLOOR, (max(x for x, _ in c) - min(x for x, _ in c)) / cm * 100, statistics.mean(s for _, s in c) / cm * 100)
    return (am / cm - 1) * 100, noise, f"control {cm:.1f} arm {am:.1f} t/s (boots {len(c)}/{len(a)})"


def counts(p, cell):
    def rd(f):
        out = {}
        for line in open(f):
            m = POS.match(line)
            if m:
                out[(m.group(1), int(m.group(2)))] = float(m.group(3))
        return out
    try:
        x, y = rd(os.path.join(p, f"pos-{cell}.before")), rd(os.path.join(p, f"pos-{cell}.after"))
    except OSError:
        return {}
    return {k: v - x.get(k, 0.0) for k, v in y.items()}


def acceptance(C, A):
    """Per-position rates (control, arm), pooled over cells and passes that finished on both sides."""
    tot = ({}, {})
    for k in sorted(set(C) & set(A)):
        for cell in PROBES + WALL:
            if probe(C[k], cell) is None or probe(A[k], cell) is None:
                continue
            for t, p in zip(tot, (C[k], A[k])):
                for key, v in counts(p, cell).items():
                    t[key] = t.get(key, 0.0) + v
    return [{pos: t[("accepted", pos)] / dr for (kind, pos), dr in t.items()
             if kind == "draft" and dr and ("accepted", pos) in t} for t in tot]


def logit_pair(x, y):
    same, dl = 0, []
    for p, q in zip(x["records"], y["records"]):
        n = next((i for i, (s, t) in enumerate(zip(p["tokens"], q["tokens"])) if s != t), None)
        if n is None:
            same, n = same + 1, min(len(p["tokens"]), len(q["tokens"]))
        dl += [abs(s - t) for s, t in zip(p["token_logprobs"][:n], q["token_logprobs"][:n]) if s is not None and t is not None]
    return f"{same}/{len(x['records'])} prompts identical, |dlogprob| mean {statistics.mean(dl) if dl else 0:.4f} max {max(dl) if dl else 0:.4f}"


def logits(C, A):
    def rd(p, x):
        try:
            return json.load(open(os.path.join(p, f"logits-{x}.json")))
        except (OSError, ValueError):
            return None
    for who, B in (("ctl", C), ("arm", A)):
        for k, p in sorted(B.items()):
            a, b = rd(p, "a"), rd(p, "b")
            print(f"  self  {who}-p{k} a/b: {logit_pair(a, b) if a and b else 'missing'}")
    for k in sorted(set(C) & set(A)):
        a, b = rd(C[k], "a"), rd(A[k], "a")
        print(f"  cross ctl-p{k}/arm-p{k}: {logit_pair(a, b) if a and b else 'missing'}")


def main(d, acc_only=False):
    C, A = boots(d, "ctl"), boots(d, "arm")
    kill, worse, missing, better = [], [], [], []
    for p in marked(d, "FAILED"):
        (kill if p.startswith("arm") else missing).append(f"{p}: boot failed")
    for p in marked(d, "COLD"):
        missing.append(f"{p}: backbone compiled at boot (cold): bake the image first")
    for p in marked(d, "DIED"):
        (kill if p.startswith("arm") else missing).append(f"{p}: server died during the boot's measurements")
    print("== acceptance per draft position (T=0 probe cells, pooled)")
    ca, aa = acceptance(C, A)
    if not ca or not aa:
        missing.append("acceptance: no paired data")
    for pos in sorted(set(ca) ^ set(aa)):   # different draft counts: compare the shared positions only
        print(f"  pos {pos}: only on the {'control' if pos in ca else 'arm'} side (not judged)")
    for pos in sorted(set(ca) & set(aa)):
        c, a = ca[pos], aa[pos]
        bad = abs(a - c) > ACC_TOL
        print(f"  pos {pos}: control {c:.3f} arm {a:.3f} shift {a - c:+.3f}{'  > 0.03' if bad else ''}")
        if bad:
            kill.append(f"acceptance pos {pos} moved {a - c:+.3f}")
    if acc_only:
        print("ACC=" + ("KILL" if kill else "OK"))
        return 3 if kill else 0
    print("== cells: arm vs control % (noise = the cell's noise band, %)")
    for kind, cells, fn in (("probe", PROBES, probe_cell), ("wall", WALL, wall_cell), ("benchy", BENCHY, benchy_cell)):
        for cell in cells:
            bad = arm_unfinished(C, A, cell)
            if bad:
                kill.append(f"{kind} {cell}: arm did not finish in pass {','.join(map(str, bad))}, control did")
            r = fn(C, A, cell)
            if r is None:
                print(f"  {kind:6s} {cell:12s} missing")
                missing.append(f"{kind} {cell}: missing")
                continue
            delta, noise, info = r
            line = f"{kind} {cell} {delta:+.2f}% (noise {noise:.2f}%)"
            if delta < -KILL_X * noise:
                tag = "WORSE>2x"; kill.append(line)
            elif delta < -noise:
                tag = "worse"; worse.append(line)
            elif delta > noise:
                tag = "better"; better.append(line)
            else:
                tag = ""
            print(f"  {kind:6s} {cell:12s} {delta:+6.2f}%  noise {noise:5.2f}%  {tag:8s}  {info}")
    print("== logits_equiv (reported, not judged)")
    logits(C, A)
    if kill:
        verdict = "KILL"
    elif missing or worse or not better:
        verdict = "INCONCLUSIVE"
        if not (missing or worse):
            missing.append("no cell better than its noise")
    else:
        verdict = "PROMOTE"
    for x in kill + worse + missing:
        print("reason:", x)
    print("better:", "; ".join(better) or "none")
    print(f"VERDICT={verdict}")
    return 0


def selftest():
    def make(d, arm=1.0, cell_arm=None, acc=0.0, fail=None, arm_wall=(200, 205), ctl_wall=(300, 310)):
        """4 boots; arm tok/s x arm (or x cell_arm[cell]); control boot 2 is 0.3 % faster than boot 1;
        d16k-c8 wall: control 300/310 s, arm arm_wall (480 = cut)."""
        for who, k in (("ctl", 1), ("arm", 1), ("arm", 2), ("ctl", 2)):
            p = os.path.join(d, f"{who}-p{k}")
            os.makedirs(p)
            if fail == f"{who}-p{k}":
                open(os.path.join(p, "FAILED"), "w").write("failed")
                continue
            f = 1.003 if (who, k) == ("ctl", 2) else 1.0
            w = arm_wall[k - 1] if who == "arm" else ctl_wall[k - 1]   # 480 = cut by timeout, < 0 = errored
            rc = 124 if w >= 480 else 1 if w < 0 else 0
            open(os.path.join(p, "time-d16k-c8.txt"), "w").write(f"{rc} {abs(w)} 480\n")
            for cell in PROBES + WALL:
                g = (cell_arm or {}).get(cell, arm) if who == "arm" else f
                rows = [{"index": i, "inf": {"gap_ms_p50": 20.0 / g, "accept_rate": 0.6, "stalls": 0,
                                              "decode_tps_no_stalls": 50.0 * g * (1 + 0.002 * i)}} for i in range(4)]
                json.dump({"rows": rows}, open(os.path.join(p, f"probe-{cell}.json"), "w"))
                for when, n in (("before", 0), ("after", 1000)):
                    shift = acc if who == "arm" else 0.0
                    open(os.path.join(p, f"pos-{cell}.{when}"), "w").write("".join(
                        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{q}"}} {n * (0.8 - 0.1 * q + shift)}\n'
                        f'vllm:spec_decode_num_draft_tokens_per_pos_total{{engine="0",position="{q}"}} {n}\n' for q in range(4)))
            g = arm if who == "arm" else f
            open(os.path.join(p, "task.csv"), "w").write("".join(
                f"| qwen3.8-flash-next | {c} | {v * g:.2f} ± {v * 0.004:.2f} | x |\n"
                for c, v in (("pp2048 (c1)", 1700.0), ("tg512 (c1)", 55.0), ("tg512 (c8)", 130.0))))

    def run(**kw):
        d = tempfile.mkdtemp()
        try:
            make(d, **kw)
            out = os.path.join(d, "out.txt")
            so, sys.stdout = sys.stdout, open(out, "w")
            try:
                main(d)
                rc = main(d, acc_only=True)
            finally:
                sys.stdout.close()
                sys.stdout = so
            text = open(out).read()
            return text.split("VERDICT=")[1].split()[0], rc
        finally:
            shutil.rmtree(d)

    assert run(arm=1.05) == ("PROMOTE", 0), run(arm=1.05)
    assert run(arm=1.0, arm_wall=(300, 310))[0] == "INCONCLUSIVE"              # nothing better than noise
    assert run(arm=1.05, cell_arm={"d16k-c4": 0.95})[0] == "KILL"               # -5 % > 2x noise
    assert run(arm=1.05, cell_arm={"fresh-c1": 0.985})[0] == "INCONCLUSIVE"     # -1.5 %: between 1x and 2x noise
    assert run(arm=1.05, acc=-0.05) == ("KILL", 3)                             # acceptance moved 0.05
    assert run(arm=1.05, acc=0.02) == ("PROMOTE", 0)                           # within 0.03
    assert run(arm=1.05, fail="arm-p2")[0] == "KILL"
    assert run(arm=1.05, fail="ctl-p2")[0] == "INCONCLUSIVE"
    assert run(arm=1.05, arm_wall=(480, 480))[0] == "KILL"                     # cut both times: -37 % on the wall cell
    assert run(arm=1.05, arm_wall=(300, 320))[0] == "PROMOTE"                  # wall cell flat, the rest better
    assert run(arm=1.05, arm_wall=(-5, 205))[0] == "KILL"                      # arm errored where the control finished
    assert run(arm=1.05, ctl_wall=(300, 470), arm_wall=(480, 300))[0] != "KILL"  # arm cut once: scored, not killed
    assert run(arm=1.0, ctl_wall=(-5, 310), arm_wall=(300, 310))[0] == "INCONCLUSIVE"  # control crash is not the cut
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) in (2, 3):
        sys.exit(main(sys.argv[1], sys.argv[2:] == ["--acc-only"]))
    else:
        sys.exit(__doc__)
