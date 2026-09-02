#!/usr/bin/env python3
"""Locate FP/FX64/FX128 detection-boundary divergences (brief section 7.3).

Runs all three scalar models offline - outside any timed region - and compares
the iteration indices at which each detects, not merely the totals.
"""

import json
import struct
from pathlib import Path

TWO64 = 1 << 64
ITERS = 10_000_000

def d(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]

def main():
    root = Path(__file__).resolve().parent.parent
    import sys
    sys.path.insert(0, str(root / "tools"))
    hpp = (root / "src" / "constants.hpp").read_text()
    def grab(name):
        import re
        m = re.search(r"%s\s*=\s*(0x[0-9a-f]+)ull" % name, hpp)
        return int(m.group(1), 16)
    def grab_arr(name):
        import re
        body = re.search(r"%s\[3\] = \{(.*?)\};" % name, hpp, re.S).group(1)
        return [int(x, 16) for x in re.findall(r"(0x[0-9a-f]+)ull", body)]

    x0b, dlb = grab("kX0Bits"), grab("kDeltaBits")
    X0, D64 = grab("kX0Fx"), grab("kD64Fx")
    W = grab_arr("kWEps"); EB = grab_arr("kEpsBits"); OB = grab_arr("kOneMinusEpsBits")
    labels = ["1e-6", "1e-9", "1e-12"]

    MASK128 = (1 << 128) - 1
    X0_128 = (grab("kX0_128_hi") << 64) | grab("kX0_128_lo")
    D_128 = (grab("kD_128_hi") << 64) | grab("kD_128_lo")
    W128 = [(h << 64) | l for h, l in zip(grab_arr("kW128Hi"), grab_arr("kW128Lo"))]
    T2_128 = [(h << 64) | l for h, l in zip(grab_arr("kTwoW128Hi"), grab_arr("kTwoW128Lo"))]

    report = {}
    for i, lab in enumerate(labels):
        delta, eps, ome = d(dlb), d(EB[i]), d(OB[i])
        x = d(x0b)
        fp_hits = []
        for t in range(ITERS):
            x += delta
            if x >= 1.0:
                x -= 1.0
            if x < eps or x >= ome:
                fp_hits.append(t)
        u = X0
        twoW = 2 * W[i]
        fx64_hits = []
        for t in range(ITERS):
            u = (u + D64) & 0xFFFFFFFFFFFFFFFF
            if ((u + W[i]) & 0xFFFFFFFFFFFFFFFF) < twoW:
                fx64_hits.append(t)
        v = X0_128
        w128, t2 = W128[i], T2_128[i]
        fx128_hits = []
        for t in range(ITERS):
            v = (v + D_128) & MASK128
            if ((v + w128) & MASK128) < t2:
                fx128_hits.append(t)

        sets = {"FP": set(fp_hits), "FX64": set(fx64_hits), "FX128": set(fx128_hits)}
        pairs = {}
        for a in ("FP", "FX64", "FX128"):
            for b in ("FP", "FX64", "FX128"):
                if a < b:
                    only_a = sorted(sets[a] - sets[b])
                    only_b = sorted(sets[b] - sets[a])
                    pairs["%s_vs_%s" % (a, b)] = dict(
                        n_only_first=len(only_a), n_only_second=len(only_b),
                        only_first_iterations=only_a[:20],
                        only_second_iterations=only_b[:20])
        report[lab] = dict(
            fp_per_lane=len(fp_hits), fx64_per_lane=len(fx64_hits),
            fx128_per_lane=len(fx128_hits),
            fp_total_16=len(fp_hits) * 16, fx64_total_16=len(fx64_hits) * 16,
            fx128_total_16=len(fx128_hits) * 16,
            first_hit=fp_hits[:5], last_hit=fp_hits[-5:] if fp_hits else [],
            pairwise=pairs)
        print(lab, "FP=%d FX64=%d FX128=%d" % (len(fp_hits), len(fx64_hits),
                                               len(fx128_hits)),
              {k: (v["n_only_first"], v["n_only_second"]) for k, v in pairs.items()},
              flush=True)

    (root / "results" / "divergence.json").open("w", newline="\n").write(
        json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
