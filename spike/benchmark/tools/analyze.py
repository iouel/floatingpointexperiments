#!/usr/bin/env python3
"""FP vs FX64 analysis.

Order of operations - the noise floor is computed before any FP-versus-FX64
comparison is inspected:

    1 median(FP-A)   2 median(FP-B)   3 noise_floor_pct   4 pool FP-A+FP-B
    5 median(FP), stddev(FP)          6 threshold         7 FX64_margin_pct
    8 verdict

    noise_floor_pct(eps) = 100 * |median(FP-A) - median(FP-B)|
                                 / min(median(FP-A), median(FP-B))
    threshold(eps)       = max(noise_floor_pct(eps),
                               3 * stddev(FP_pooled) / median(FP_pooled) * 100)
    FX64_margin_pct(eps) = 100 * (median(FP) - median(FX64)) / median(FP)

    Decision <x> = median over the three eps values.

Verdict: `Decision FX64_margin_pct > Decision threshold` -> margin present
(FX64 faster); otherwise no margin.  The rule is signed, not absolute: a
negative margin means FP is faster and is reported as "no margin".

`stddev` is the sample standard deviation of the per-invocation
`value_per_state` observations.
"""

import csv
import json
import statistics as st
from pathlib import Path

EPS = ["1e-6", "1e-9", "1e-12"]
VARIANTS = ["FP-A", "FP-B", "FX64"]


def main():
    root = Path(__file__).resolve().parent.parent
    res = root / "results"
    rows = list(csv.DictReader((res / "raw.csv").open()))
    assert len(rows) == 180, len(rows)

    def sel(eps, variant):
        return [float(r["value_per_state"]) for r in rows
                if r["eps"] == eps and r["build_variant"] == variant]

    def stats(v):
        return dict(n=len(v), median=st.median(v), minimum=min(v), stddev=st.stdev(v))

    out = {"per_eps": {}}
    for eps in EPS:
        a, b = sel(eps, "FP-A"), sel(eps, "FP-B")
        fx = sel(eps, "FX64")
        assert all(len(v) == 20 for v in (a, b, fx))

        ma, mb = st.median(a), st.median(b)                     # 1, 2
        noise = 100.0 * abs(ma - mb) / min(ma, mb)              # 3
        pooled = a + b                                          # 4
        mfp, sdfp = st.median(pooled), st.stdev(pooled)         # 5
        cv3 = 3.0 * sdfp / mfp * 100.0
        threshold = max(noise, cv3)                             # 6
        margin = 100.0 * (mfp - st.median(fx)) / mfp            # 7

        det = {v: sorted({int(r["detections"]) for r in rows
                          if r["eps"] == eps and r["build_variant"] == v})
               for v in VARIANTS}

        out["per_eps"][eps] = dict(
            fp_a=stats(a), fp_b=stats(b), fp_pooled=stats(pooled), fx64=stats(fx),
            noise_floor_pct=noise, cv3_pct=cv3, threshold_pct=threshold,
            fx64_margin_pct=margin,
            fp_over_fx64_ratio=mfp / st.median(fx), detections=det)

    def med(key):
        return st.median([out["per_eps"][e][key] for e in EPS])

    out["decision_noise_floor_pct"] = med("noise_floor_pct")
    out["decision_threshold_pct"] = med("threshold_pct")
    out["decision_fx64_margin_pct"] = med("fx64_margin_pct")
    out["decision_fp_over_fx64_ratio"] = med("fp_over_fx64_ratio")
    out["verdict"] = ("margin present"
                      if out["decision_fx64_margin_pct"] > out["decision_threshold_pct"]
                      else "no margin")
    out["faster"] = "FX64" if out["decision_fx64_margin_pct"] > 0 else "FP"

    log = json.load((res / "run_log.json").open())
    busy = [r["sibling_busy_jiffies"] for r in log["runs"]]
    out["sibling"] = dict(runs=len(busy), zero_busy_runs=sum(1 for x in busy if x == 0),
                          max_busy_jiffies=max(busy), total_busy_jiffies=sum(busy),
                          tsc_aux_rejections=len(log["rejections"]))

    (res / "analysis.json").open("w", newline="\n").write(json.dumps(out, indent=1))

    for eps in EPS:
        d = out["per_eps"][eps]
        print("eps=%s" % eps)
        for k in ("fp_a", "fp_b", "fp_pooled", "fx64"):
            s = d[k]
            print("  %-9s n=%2d med=%.9f min=%.9f sd=%.9f"
                  % (k, s["n"], s["median"], s["minimum"], s["stddev"]))
        print("  noise=%.4f%%  3*CV=%.4f%%  threshold=%.4f%%  margin=%.4f%%  FP/FX64=%.4f"
              % (d["noise_floor_pct"], d["cv3_pct"], d["threshold_pct"],
                 d["fx64_margin_pct"], d["fp_over_fx64_ratio"]))
        print("  detections %s" % d["detections"])
    print()
    print("Decision noise_floor_pct = %.4f" % out["decision_noise_floor_pct"])
    print("Decision threshold       = %.4f" % out["decision_threshold_pct"])
    print("Decision FX64_margin_pct = %.4f" % out["decision_fx64_margin_pct"])
    print("FP/FX64 measured ratio   = %.4f" % out["decision_fp_over_fx64_ratio"])
    print("Verdict: %s (%s faster)" % (out["verdict"], out["faster"]))
    print("sibling / validity:", out["sibling"])


if __name__ == "__main__":
    main()
