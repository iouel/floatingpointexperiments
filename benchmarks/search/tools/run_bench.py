#!/usr/bin/env python3
"""Execute the 30 benchmark process invocations and write results/raw_times.csv.

Run inside WSL:  python3 tools/run_bench.py <build_dir>

Ordering: the 30 invocations (10 each of FP-A, FP-B and FX64) are shuffled with
a fixed recorded seed, so no implementation runs contiguously and the order is
reproducible.  The resulting order is written to results/run_order.txt.

Every invocation is pinned with `taskset -c 6` (core 3, SMT sibling 7 left
idle), matching the pinning used by the accumulator spike.  Sibling occupancy
is sampled from /proc/stat immediately before and after each run and recorded
in results/run_log.json, so a contaminated run is visible after the fact rather
than silently averaged in.

Each binary prints its own impl_name in column 1; the driver checks that
against the build variant it invoked (FP-A and FP-B must both report FP, FX64
must report FX64) and substitutes the build-variant label into raw_times.csv.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

SEED = 20260902
PIN_CPU = 6           # core 3, first SMT thread
SIBLING_CPU = 7       # SMT sibling of PIN_CPU
RUNS_PER_VARIANT = 10
VARIANTS = [("FP-A", "search_fp", "FP"),
            ("FP-B", "search_fp_b", "FP"),
            ("FX64", "search_fx64", "FX64")]


def sibling_jiffies(cpu):
    """Non-idle and total jiffies for one logical CPU from /proc/stat."""
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu%d " % cpu):
                v = [int(x) for x in line.split()[1:]]
                idle = v[3] + v[4]          # idle + iowait
                return sum(v) - idle, sum(v)
    return None, None


def parse_provenance(stderr):
    out = {}
    for token in stderr.split():
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    return out


def main():
    build = Path(sys.argv[1]).expanduser()
    root = Path(__file__).resolve().parent.parent
    results = root / "results"
    results.mkdir(exist_ok=True)

    schedule = []
    for label, _, _ in VARIANTS:
        for r in range(RUNS_PER_VARIANT):
            schedule.append((label, r))
    random.Random(SEED).shuffle(schedule)

    exe_of = {label: exe for label, exe, _ in VARIANTS}
    impl_of = {label: impl for label, _, impl in VARIANTS}

    with (results / "run_order.txt").open("w", newline="\n") as f:
        f.write("# Fixed run order, seed=%d\n" % SEED)
        f.write("# pinned logical CPU = %d, SMT sibling = %d (left idle)\n"
                % (PIN_CPU, SIBLING_CPU))
        f.write("# position\tbuild_variant\trun_id\n")
        for i, (label, r) in enumerate(schedule):
            f.write("%d\t%s\t%d\n" % (i, label, r))
    # no implementation runs contiguously more than this many times
    longest = 1
    cur = 1
    for i in range(1, len(schedule)):
        cur = cur + 1 if schedule[i][0] == schedule[i - 1][0] else 1
        longest = max(longest, cur)

    rows, run_log = [], []
    for i, (label, run_id) in enumerate(schedule):
        cmd = ["taskset", "-c", str(PIN_CPU), str(build / exe_of[label]), str(run_id)]
        busy0, tot0 = sibling_jiffies(SIBLING_CPU)
        p = subprocess.run(cmd, capture_output=True, text=True)
        busy1, tot1 = sibling_jiffies(SIBLING_CPU)
        if p.returncode != 0:
            print("FATAL: run %d (%s) failed rc=%d: %s"
                  % (i, label, p.returncode, p.stderr.strip()))
            return 1

        impl, got_run_id, time_ms = p.stdout.strip().split(",")
        if impl != impl_of[label]:
            print("FATAL: %s binary reported impl=%s, expected %s"
                  % (label, impl, impl_of[label]))
            return 1
        assert got_run_id == str(run_id)

        rows.append((label, run_id, float(time_ms)))
        run_log.append(dict(position=i, build_variant=label, run_id=run_id,
                            impl_name=impl, time_ms=float(time_ms),
                            sibling_busy_jiffies=busy1 - busy0,
                            sibling_total_jiffies=tot1 - tot0,
                            provenance=parse_provenance(p.stderr.strip())))
        print("  %2d/%d  %-5s run %d  %10.3f ms" % (i + 1, len(schedule), label,
                                                    run_id, float(time_ms)),
              flush=True)

    with (results / "raw_times.csv").open("w", newline="\n") as f:
        f.write("implementation,run_id,time_ms\n")
        for label, run_id, t in rows:
            f.write("%s,%d,%.6f\n" % (label, run_id, t))

    # every run must agree on the answer; a differing best_z would invalidate
    # the campaign even though equivalence passed earlier
    prov_keys = ["best_z", "best_D_bits", "checksum", "candidates"]
    base = run_log[0]["provenance"]
    disagree = [r["position"] for r in run_log
                if any(r["provenance"].get(k) != base.get(k) for k in prov_keys)]

    with (results / "run_log.json").open("w", newline="\n") as f:
        json.dump(dict(seed=SEED, pin_cpu=PIN_CPU, sibling_cpu=SIBLING_CPU,
                       runs_per_variant=RUNS_PER_VARIANT,
                       longest_contiguous_same_variant=longest,
                       answer_disagreements=disagree,
                       answer=base, runs=run_log), f, indent=1)

    print("\nwrote %s (%d rows), %s, %s"
          % (results / "raw_times.csv", len(rows), results / "run_order.txt",
             results / "run_log.json"))
    print("longest contiguous same-variant streak: %d" % longest)
    print("runs disagreeing on the answer: %d" % len(disagree))
    print("answer: %s" % base)
    busy = [r["sibling_busy_jiffies"] for r in run_log]
    print("sibling cpu%d busy jiffies: zero in %d/%d runs, max %d"
          % (SIBLING_CPU, sum(1 for x in busy if x == 0), len(busy), max(busy)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
