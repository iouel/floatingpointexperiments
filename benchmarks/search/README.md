# Search benchmark (Kronecker-sequence parameter search)

The follow-on experiment gated by the accumulator benchmark's verdict (see
`docs/briefs/accumulator-brief-v3.md`, §9): instead of just accumulating and
detecting, this searches for the integer parameter that makes the detection
statistic closest to its target value.

For `alpha = z/N`, `x_0 = 0`, `x_{t+1} = (x_t + alpha) mod 1`:

```
hits(z) = |{ t in 1..T : x_t in [0,eps) ∪ [1-eps,1) }|
D(z)    = | hits(z)/T - 2*eps |
best_z  = argmin D(z) over z in 1..N-1 with gcd(z,N) = 1
```

Two implementations of the same search are compared: a wrapped `double`
accumulator (`FP`, in two link/layout variants `search_fp`/`search_fp_b`) and
a 64-bit fixed-point accumulator (`FX64`, `search_fx64`). Both are checked
against an exact integer reference (`src/reference.cpp`, no floating-point
rounding anywhere) rather than merely against each other.

## Layout

* `src/search_common.hpp`, `src/search_constants.hpp` — shared metric
  definition and problem constants.
* `src/search_fp.cpp`, `src/search_fx64.cpp` — the two timed
  implementations.
* `src/reference.cpp` — exact-integer ground truth, linked into every
  binary.
* `src/padding.cpp` — layout-only translation unit for the `_b` link-order
  variant.
* `src/harness.cpp` — argument handling and mode dispatch only; the timed
  search itself lives in each implementation TU.
* `tools/` — constant generation, correctness/equivalence verification, run
  orchestration, and disassembly analysis.
* `results/` — an example run captured on the original development machine.
  **Not** a performance claim for any other machine.

## Building

```sh
cmake -S . -B build
cmake --build build -j
```

Produces `search_fp`, `search_fp_b`, `search_fx64`.

## Correctness checks (safe anywhere, no timing)

```sh
./build/search_fx64 --verify   # own hits vs exact reference, this binary
python3 tools/verify_equivalence.py build   # all binaries vs each other + reference
```

`verify_equivalence.py` also checks every binary's `--constants` output
against the recorded `results/constants_expected.txt`. It prints `RESULT:
PASS`/`RESULT: FAIL` and exits non-zero on any mismatch; this is what CI runs
as a smoke test.

## Timed runs (pinned machine only — do not run in CI)

```sh
python3 tools/run_bench.py build
```

Pins to a specific logical CPU (matching the accumulator spike's pinning),
leaves its SMT sibling idle, and records sibling occupancy so a contaminated
run is visible after the fact. Designed for a quiet, dedicated machine, not
shared CI runners.

## Regenerating `results/`

```sh
python3 tools/gen_constants.py            # -> src/search_constants.hpp, results/constants*.txt
cmake -S . -B build && cmake --build build -j
python3 tools/verify_equivalence.py build  # -> results/hits.csv, equivalence.json
python3 tools/run_bench.py build           # -> results/raw_times.csv, run_order.txt, run_log.json
python3 tools/analyze.py                   # -> results/analysis.json, summary.md
python3 tools/extract_asm.py build         # -> results/asm_*.txt
python3 tools/asm_norm.py                  # -> results/asm_check.md
```

Kept under version control as a worked example of the pipeline and output
format, not as a claim that reproduces identically on other hardware.
