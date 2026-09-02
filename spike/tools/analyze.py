#!/usr/bin/env python3
"""Analysis in the order fixed by brief section 4.2.

    1 median(FP-A)   2 median(FP-B)   3 noise_floor_pct   4 pool FP-A+FP-B
    5 median(FP), stddev(FP)          6 decision threshold
    7 FX64_margin_pct                 8 FX128_margin_pct
    9 FX128_vs_FX64_margin_pct       10 verdict rules

`stddev` is the sample standard deviation of the per-invocation
`value_per_state` observations.

Threshold, per section 4.2:

    threshold(eps) = max(noise_floor_pct(eps),
                         3 * stddev(FP_pooled, eps) / median(FP_pooled, eps) * 100)

Verdict rules.  Section 4.3 classifies a margin as present when
|margin| > threshold.  That rule is reported here for every margin.  The
*primary* verdict additionally uses the signed form agreed with the brief
author, because the section 9 gate asks whether FX128 is faster than FP, and an
absolute-value rule would report "margin present" for an FX128 that is
decisively slower:

    primary: margin present  <=>  FX128_margin_pct > threshold   (FX128 faster)
"""

import csv
import json
import statistics as st
from pathlib import Path

EPS = ["1e-6", "1e-9", "1e-12"]
VARIANTS = ["FP-A", "FP-B", "FX64", "FX128"]


def main():
    root = Path(__file__).resolve().parent.parent
    res = root / "results"
    rows = list(csv.DictReader((res / "raw.csv").open()))
    assert len(rows) == 240, len(rows)

    def sel(eps, variant):
        return [float(r["value_per_state"]) for r in rows
                if r["eps"] == eps and r["build_variant"] == variant]

    def stats(v):
        return dict(n=len(v), median=st.median(v), minimum=min(v), stddev=st.stdev(v))

    out = {"per_eps": {}}
    for eps in EPS:
        a, b = sel(eps, "FP-A"), sel(eps, "FP-B")
        fx64, fx128 = sel(eps, "FX64"), sel(eps, "FX128")
        assert all(len(v) == 20 for v in (a, b, fx64, fx128))

        ma, mb = st.median(a), st.median(b)                      # 1, 2
        noise = 100.0 * abs(ma - mb) / min(ma, mb)               # 3
        pooled = a + b                                           # 4
        mfp, sdfp = st.median(pooled), st.stdev(pooled)          # 5
        cv3 = 3.0 * sdfp / mfp * 100.0
        threshold = max(noise, cv3)                              # 6

        m64 = 100.0 * (mfp - st.median(fx64)) / mfp              # 7
        m128 = 100.0 * (mfp - st.median(fx128)) / mfp            # 8
        m128v64 = 100.0 * (st.median(fx64) - st.median(fx128)) / st.median(fx64)  # 9

        det = {v: sorted({int(r["detections"]) for r in rows
                          if r["eps"] == eps and r["build_variant"] == v})
               for v in VARIANTS}

        out["per_eps"][eps] = dict(
            fp_a=stats(a), fp_b=stats(b), fp_pooled=stats(pooled),
            fx64=stats(fx64), fx128=stats(fx128),
            noise_floor_pct=noise, cv3_pct=cv3, threshold_pct=threshold,
            fx64_margin_pct=m64, fx128_margin_pct=m128,
            fx128_vs_fx64_margin_pct=m128v64, detections=det)

    def med(key):
        return st.median([out["per_eps"][e][key] for e in EPS])

    out["decision_fx64_margin_pct"] = med("fx64_margin_pct")
    out["decision_fx128_margin_pct"] = med("fx128_margin_pct")
    out["decision_fx128_vs_fx64_margin_pct"] = med("fx128_vs_fx64_margin_pct")
    out["decision_threshold_pct"] = med("threshold_pct")
    out["decision_noise_floor_pct"] = med("noise_floor_pct")

    t = out["decision_threshold_pct"]

    def classify(margin, faster, slower):
        return dict(margin_pct=margin,
                    abs_rule=("margin present" if abs(margin) > t else "no margin"),
                    signed_rule=("margin present" if margin > t else "no margin"),
                    direction=(faster if margin > 0 else slower))

    out["verdicts"] = {
        "FP_vs_FX64": classify(out["decision_fx64_margin_pct"], "FX64 faster", "FP faster"),
        "FP_vs_FX128": classify(out["decision_fx128_margin_pct"], "FX128 faster", "FP faster"),
        "FX64_vs_FX128": classify(out["decision_fx128_vs_fx64_margin_pct"],
                                  "FX128 faster", "FX64 faster"),
    }
    out["primary_verdict"] = out["verdicts"]["FP_vs_FX128"]["signed_rule"]
    out["primary_gate_fx128_faster_than_fp"] = (
        out["decision_fx128_margin_pct"] > t)

    log = json.load((res / "run_log.json").open())
    busy = [r["sibling_busy_jiffies"] for r in log["runs"]]
    out["sibling"] = dict(runs=len(busy), zero_busy_runs=sum(1 for x in busy if x == 0),
                          max_busy_jiffies=max(busy), total_busy_jiffies=sum(busy),
                          retries=len(log["retries"]))

    (res / "analysis.json").open("w", newline="\n").write(json.dumps(out, indent=1))

    for eps in EPS:
        d = out["per_eps"][eps]
        print("eps=%s" % eps)
        for k in ("fp_a", "fp_b", "fp_pooled", "fx64", "fx128"):
            s = d[k]
            print("  %-9s n=%2d med=%.9f min=%.9f sd=%.9f"
                  % (k, s["n"], s["median"], s["minimum"], s["stddev"]))
        print("  noise=%.4f%%  3*CV=%.4f%%  threshold=%.4f%%"
              % (d["noise_floor_pct"], d["cv3_pct"], d["threshold_pct"]))
        print("  FX64_margin=%.4f%%  FX128_margin=%.4f%%  FX128_vs_FX64=%.4f%%"
              % (d["fx64_margin_pct"], d["fx128_margin_pct"],
                 d["fx128_vs_fx64_margin_pct"]))
        print("  detections %s" % d["detections"])
    print()
    print("Decision noise_floor_pct          = %.4f" % out["decision_noise_floor_pct"])
    print("Decision threshold                = %.4f" % out["decision_threshold_pct"])
    print("Decision FX64_margin_pct          = %.4f" % out["decision_fx64_margin_pct"])
    print("Decision FX128_margin_pct         = %.4f  (PRIMARY)"
          % out["decision_fx128_margin_pct"])
    print("Decision FX128_vs_FX64_margin_pct = %.4f" % out["decision_fx128_vs_fx64_margin_pct"])
    for k, v in out["verdicts"].items():
        print("  %-14s margin=%9.4f  abs-rule=%-14s signed-rule=%-14s %s"
              % (k, v["margin_pct"], v["abs_rule"], v["signed_rule"], v["direction"]))
    print("PRIMARY VERDICT (signed, FP vs FX128): %s" % out["primary_verdict"])
    print("sibling:", out["sibling"])


if __name__ == "__main__":
    main()
