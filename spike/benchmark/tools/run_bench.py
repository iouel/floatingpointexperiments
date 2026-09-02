#!/usr/bin/env python3
"""Execute the 180 benchmark process invocations and write results/raw.csv.

Run inside WSL:  python3 tools/run_bench.py <build_dir>

Ordering: for each eps the 60 invocations across FP-A, FP-B and FX64 are
shuffled with a fixed recorded seed, so no implementation runs contiguously and
the order is reproducible.  The resulting order is written to
results/run_order.txt.

A run is valid only when the harness accepts it: both RDTSCP reads must return
the same TSC_AUX and its low 12 bits must equal sched_getcpu().  Rejected runs
are retried up to MAX_RETRIES and every rejection is logged to
results/run_log.json.  All valid runs are recorded; none is discarded.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

SEED = 20260902
PIN_CPU = 6          # core 3, first SMT thread
SIBLING_CPU = 7      # SMT sibling of PIN_CPU
RUNS_PER_VARIANT = 20
EPS_LABELS = ["1e-6", "1e-9", "1e-12"]
VARIANTS = [("FP-A", "fp_a"), ("FP-B", "fp_b"), ("FX64", "fx64")]
MAX_RETRIES = 3


def sibling_jiffies(cpu):
    """Non-idle and total jiffies for one logical CPU from /proc/stat."""
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu%d " % cpu):
                v = [int(x) for x in line.split()[1:]]
                idle = v[3] + v[4]          # idle + iowait
                return sum(v) - idle, sum(v)
    return None, None


def main():
    build = Path(sys.argv[1])
    root = Path(__file__).resolve().parent.parent
    results = root / "results"
    results.mkdir(exist_ok=True)

    schedule = []
    for eps_index, eps in enumerate(EPS_LABELS):
        block = []
        for variant, _ in VARIANTS:
            for r in range(RUNS_PER_VARIANT):
                block.append((eps_index, eps, variant, r))
        random.Random(SEED + eps_index).shuffle(block)
        schedule.extend(block)

    order_path = results / "run_order.txt"
    with order_path.open("w", newline="\n") as f:
        f.write("# Fixed run order. seed=%d (per-eps seed = %d + eps_index)\n"
                % (SEED, SEED))
        f.write("# pinned logical CPU = %d, SMT sibling = %d\n" % (PIN_CPU, SIBLING_CPU))
        f.write("# position\teps\tbuild_variant\trun_id\n")
        for i, (_, eps, variant, r) in enumerate(schedule):
            f.write("%d\t%s\t%s\t%d\n" % (i, eps, variant, r))

    rows = []
    run_log = []
    rejections = []
    for i, (eps_index, eps, variant, run_id) in enumerate(schedule):
        exe = dict(VARIANTS)[variant]
        cmd = ["taskset", "-c", str(PIN_CPU), str(build / exe),
               str(eps_index), variant, str(run_id)]
        for attempt in range(MAX_RETRIES):
            busy0, tot0 = sibling_jiffies(SIBLING_CPU)
            p = subprocess.run(cmd, capture_output=True, text=True)
            busy1, tot1 = sibling_jiffies(SIBLING_CPU)
            if p.returncode == 0:
                break
            rejections.append(dict(position=i, eps=eps, variant=variant,
                                   run_id=run_id, attempt=attempt,
                                   rc=p.returncode, stderr=p.stderr.strip()))
        if p.returncode != 0:
            print("FATAL: TSC_AUX validation failed persistently at position %d: %s"
                  % (i, p.stderr.strip()))
            return 1
        rows.append(p.stdout.strip())
        run_log.append(dict(position=i, eps=eps, variant=variant, run_id=run_id,
                            sibling_busy_jiffies=busy1 - busy0,
                            sibling_total_jiffies=tot1 - tot0,
                            harness_stderr=p.stderr.strip()))
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(schedule)), flush=True)

    header = ("implementation,build_variant,run_id,iterations,logical_states,eps,"
              "measured_value,unit,value_per_state,detections")
    with (results / "raw.csv").open("w", newline="\n") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")

    with (results / "run_log.json").open("w", newline="\n") as f:
        json.dump(dict(seed=SEED, pin_cpu=PIN_CPU, sibling_cpu=SIBLING_CPU,
                       max_retries=MAX_RETRIES, rejections=rejections,
                       runs=run_log), f, indent=1)

    print("wrote %s (%d rows), %s, %s"
          % (results / "raw.csv", len(rows), order_path, results / "run_log.json"))
    print("TSC_AUX rejections: %d" % len(rejections))
    return 0


if __name__ == "__main__":
    sys.exit(main())
