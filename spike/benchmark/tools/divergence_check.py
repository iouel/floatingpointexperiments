#!/usr/bin/env python3
"""Locate FP / FX64 detection-boundary divergences.

Replays both scalar models offline - outside any timed region - and compares the
iteration indices at which each detects, not merely the totals.  Writes
results/divergence.json.
"""

import json
import re
import struct
from pathlib import Path

ITERS = 10_000_000
MASK64 = (1 << 64) - 1


def d(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def main():
    root = Path(__file__).resolve().parent.parent
    hpp = (root / "src" / "constants.hpp").read_text()

    def grab(name):
        return int(re.search(r"%s\s*=\s*(0x[0-9a-f]+)ull" % name, hpp).group(1), 16)

    def grab_arr(name):
        body = re.search(r"%s\[3\] = \{(.*?)\};" % name, hpp, re.S).group(1)
        return [int(x, 16) for x in re.findall(r"(0x[0-9a-f]+)ull", body)]

    x0b, dlb = grab("kX0Bits"), grab("kDeltaBits")
    X0, D64 = grab("kX0Fx"), grab("kD64Fx")
    W = grab_arr("kWEps")
    EB, OB = grab_arr("kEpsBits"), grab_arr("kOneMinusEpsBits")
    labels = ["1e-6", "1e-9", "1e-12"]

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
        fx_hits = []
        for t in range(ITERS):
            u = (u + D64) & MASK64
            if ((u + W[i]) & MASK64) < twoW:
                fx_hits.append(t)

        only_fp = sorted(set(fp_hits) - set(fx_hits))
        only_fx = sorted(set(fx_hits) - set(fp_hits))
        report[lab] = dict(
            fp_per_lane=len(fp_hits), fx64_per_lane=len(fx_hits),
            fp_total_16=len(fp_hits) * 16, fx64_total_16=len(fx_hits) * 16,
            n_only_fp=len(only_fp), n_only_fx64=len(only_fx),
            only_fp_iterations=only_fp[:20], only_fx64_iterations=only_fx[:20],
            first_hits=fp_hits[:5], last_hits=fp_hits[-5:] if fp_hits else [])
        print(lab, "FP=%d FX64=%d only_fp=%d only_fx64=%d"
              % (len(fp_hits), len(fx_hits), len(only_fp), len(only_fx)), flush=True)

    (root / "results" / "divergence.json").open("w", newline="\n").write(
        json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
