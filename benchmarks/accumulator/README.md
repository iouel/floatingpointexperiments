# Accumulator benchmark (FX64 vs FX128 vs wrapped FP)

Implements `docs/briefs/accumulator-brief-v3.md`: does a 64-bit or 128-bit
fixed-point modular accumulator cost fewer ticks per state than a wrapped
AVX2 `double` accumulator, when a boundary-detection predicate is included
in the timed region?

All three variants run the same recurrence, `x[t+1] = (x[t] + Delta) mod 1`,
over 16 independent lanes packed into AVX2 YMM registers, and share the same
mathematically-defined `x0`/`Delta`/`eps` constants (generated once, exactly,
outside any timed region — see `tools/gen_constants.py`).

## Layout

* `src/constants.hpp` — generated constants shared by every variant.
* `src/spike.hpp`, `src/fp_double.cpp`, `src/fixed_u64.cpp`,
  `src/fixed_u128.cpp` — the three timed implementations (FP, FX64, FX128).
* `src/padding.cpp` — layout-only translation unit linked into the `*_b`
  binaries to check sensitivity to link/layout order.
* `src/harness.cpp` — argument handling, CPU pinning/validation, CSV
  emission, and the `--verify`/`--constants` correctness modes. Contains no
  part of the timed workload.
* `tools/` — constant generation, run orchestration, and analysis scripts.
* `results/` — an example run captured on the original development machine
  (see "Regenerating results" below). **Not** a performance claim for any
  other machine.

## Building

```sh
cmake -S . -B build
cmake --build build -j
```

Produces four executables: `fp_bench_a`, `fp_bench_b`, `fx64_bench`,
`fx128_bench`.

## Correctness checks (safe anywhere, no timing)

Each binary can verify its vectorised detection logic against an
independent scalar reference model:

```sh
./build/fx64_bench  --verify 0   # eps index 0..2 (1e-6, 1e-9, 1e-12)
./build/fx128_bench --verify 0
./build/fp_bench_a  --verify 0
```

Expected output ends in `MATCH`. This is what CI runs as a smoke test.

## Timed runs (pinned machine only — do not run in CI)

```sh
python3 tools/run_bench.py build
```

`run_bench.py` pins to a specific logical CPU, requires its SMT sibling to
be idle, and runs 240 shuffled invocations with a fixed recorded seed. It is
designed for a quiet, dedicated benchmarking machine, not for shared CI
runners, and will behave inconsistently (or refuse to run) anywhere else.

## Regenerating `results/`

The committed `results/` files were produced by, in order:

```sh
python3 tools/gen_constants.py        # -> src/constants.hpp, results/constants.md
cmake -S . -B build && cmake --build build -j
python3 tools/run_bench.py build      # -> results/raw.csv, run_order.txt, run_log.json
python3 tools/analyze.py              # -> results/analysis.json, summary.md
python3 tools/divergence_check.py     # -> results/divergence.json
```

`asm_fp_a.txt`, `asm_fp_b.txt`, `asm_fx64.txt` and `asm_fx128.txt` are
`objdump -d` disassembly of each binary's hot loop, extracted by hand and
compared with `tools/loop_stats.py` (loop-shape summary) and
`tools/asm_norm.py` (normalised diff, ignoring absolute addresses); see
`results/asm_check.md` for the recorded comparison.

These are kept under version control as a worked example of the full
pipeline and its output format, not as a claim that reproduces identically
on other hardware. Re-running the above on a different machine will
overwrite them with that machine's own numbers.
