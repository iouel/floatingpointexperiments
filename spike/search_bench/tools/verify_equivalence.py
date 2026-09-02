#!/usr/bin/env python3
"""Gates 1 and 3: constants check and cross-binary per-z hits equivalence.

Run inside WSL:  python3 tools/verify_equivalence.py <build_dir>

Step 1 (gate 1 completion)
    Each binary's `--constants` output is compared key-by-key against
    results/constants_expected.txt, the record written by gen_constants.py.
    A binary is required to agree on every key it prints, and to print the
    whole common set.  Only search_fx64 prints alpha_fixed samples, because
    only FX64 has an alpha_fixed.

Step 2 (gate 3)
    `--dump-hits` from all three binaries plus `--dump-ref` from one of them
    are diffed against each other and against the reference, over all 100002
    candidates.  Writes results/hits.csv and results/equivalence.json.

Any mismatch is a hard stop: the script exits non-zero and no timing is
produced.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

BINARIES = [("FP-A", "search_fp"), ("FP-B", "search_fp_b"), ("FX64", "search_fx64")]
COMMON_KEYS = ["N", "T", "eps_P", "eps_Q", "num_candidates", "eps_double_bits",
               "one_minus_eps_double_bits", "two_eps_double_bits",
               "W_eps", "two_W_eps"]


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        if " = " not in line:
            continue
        k, v = line.split(" = ", 1)
        out[k.strip()] = v.strip()
    return out


def load_dump(path):
    """z -> hits, from a `z,hits` CSV."""
    out = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["z", "hits"], header
        for row in r:
            out[int(row[0])] = int(row[1])
    return out


def parse_provenance(stderr):
    """`best_z=48 best_D=0 best_D_bits=0x... checksum=... candidates=...`"""
    out = {}
    for token in stderr.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def main():
    build = Path(sys.argv[1]).expanduser()
    root = Path(__file__).resolve().parent.parent
    res = root / "results"
    res.mkdir(exist_ok=True)
    tmp = res / "_dumps"
    tmp.mkdir(exist_ok=True)

    report = {"constants": {}, "dumps": {}, "diffs": {}, "provenance": {}}
    failures = []

    # ---------------------------------------------------------------- step 1
    expected = parse_kv((res / "constants_expected.txt").read_text())
    for label, exe in BINARIES:
        p = subprocess.run([str(build / exe), "--constants"],
                           capture_output=True, text=True, check=True)
        got = parse_kv(p.stdout)
        bad = {k: (v, expected.get(k)) for k, v in got.items()
               if k != "impl" and expected.get(k) != v}
        missing = [k for k in COMMON_KEYS if k not in got]
        report["constants"][label] = dict(keys_checked=len(got) - 1,
                                          mismatched=bad, missing=missing)
        if bad or missing:
            failures.append("constants mismatch in %s: %s %s" % (label, bad, missing))

    # ---------------------------------------------------------------- step 2
    dumps = {}
    for label, exe in BINARIES:
        path = tmp / ("hits_%s.csv" % label.replace("-", "_"))
        p = subprocess.run([str(build / exe), "--dump-hits", str(path)],
                           capture_output=True, text=True, check=True)
        dumps[label] = load_dump(path)
        report["provenance"][label] = parse_provenance(p.stderr.strip())

    p = subprocess.run([str(build / "search_fp"), "--dump-ref", str(tmp / "hits_REF.csv")],
                       capture_output=True, text=True, check=True)
    dumps["REF"] = load_dump(tmp / "hits_REF.csv")
    report["provenance"]["REF"] = parse_provenance(p.stderr.strip())

    n_expected = int(expected["num_candidates"])
    for label, d in dumps.items():
        report["dumps"][label] = dict(rows=len(d), sum_hits=sum(d.values()),
                                      min_hits=min(d.values()), max_hits=max(d.values()))
        if len(d) != n_expected:
            failures.append("%s dumped %d rows, expected %d" % (label, len(d), n_expected))

    pairs = [("FP-A", "REF"), ("FX64", "REF"), ("FP-B", "REF"),
             ("FP-A", "FX64"), ("FP-A", "FP-B")]
    for a, b in pairs:
        da, db = dumps[a], dumps[b]
        diff = [z for z in da if da[z] != db.get(z)]
        report["diffs"]["%s_vs_%s" % (a, b)] = dict(
            compared=len(da), mismatches=len(diff), first_mismatch=(diff[0] if diff else None),
            examples=[dict(z=z, a=da[z], b=db.get(z)) for z in diff[:10]])
        if diff:
            failures.append("%s vs %s: %d mismatching candidates" % (a, b, len(diff)))

    # provenance agreement: best_z, best_D bits, checksum, candidate count
    prov_keys = ["best_z", "best_D_bits", "checksum", "candidates"]
    base = report["provenance"]["REF"]
    prov_bad = {}
    for label, pv in report["provenance"].items():
        d = {k: (pv.get(k), base.get(k)) for k in prov_keys if pv.get(k) != base.get(k)}
        if d:
            prov_bad[label] = d
    report["provenance_mismatches"] = prov_bad
    if prov_bad:
        failures.append("provenance mismatch: %s" % prov_bad)

    # --------------------------------------------------------------- hits.csv
    with (res / "hits.csv").open("w", newline="\n") as f:
        f.write("z,hits_fp_a,hits_fp_b,hits_fx64,hits_ref\n")
        for z in sorted(dumps["REF"]):
            f.write("%d,%d,%d,%d,%d\n" % (z, dumps["FP-A"][z], dumps["FP-B"][z],
                                          dumps["FX64"][z], dumps["REF"][z]))

    report["result"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    report["total_candidates"] = n_expected
    (res / "equivalence.json").open("w", newline="\n").write(json.dumps(report, indent=1))

    for label in ("FP-A", "FP-B", "FX64", "REF"):
        d = report["dumps"][label]
        print("%-5s rows=%d sum_hits=%d hits range [%d,%d]"
              % (label, d["rows"], d["sum_hits"], d["min_hits"], d["max_hits"]))
    for k, v in report["diffs"].items():
        print("%-14s compared=%d mismatches=%d" % (k, v["compared"], v["mismatches"]))
    print("best_z / best_D_bits / checksum agreement:",
          "OK" if not prov_bad else prov_bad)
    print("RESULT:", report["result"])
    for m in failures:
        print("  FAILURE:", m)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
