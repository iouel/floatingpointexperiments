# Contributing

This repository holds focused performance-benchmarking spikes, not a
general-purpose library. Contributions are welcome, but please keep the
scope narrow: each benchmark exists to answer one specific question (see
`docs/briefs/`), and changes to the timed workloads should not silently
change what's being measured.

## Before you start

* Read the relevant brief under `docs/briefs/` and the benchmark's own
  `README.md` first. If a change would alter the timed workload (the code
  inside the measured region), explain why in the PR description — these
  benchmarks are deliberately pinned to specific compiler flags and code
  shapes so results stay comparable across runs.
* Prefer small, reviewable diffs. Do not reformat or restructure code you
  are not otherwise changing.

## Building

```sh
cmake -S . -B build
cmake --build build -j
```

Both benchmarks also build standalone from their own directories; see the
root `README.md`.

## Formatting

There is no repository-wide formatter or `.clang-format` yet. Match the
existing style of the file you're editing (spacing, brace style, comment
conventions) rather than introducing a new one.

## Testing / verifying changes

Every change to `src/` under either benchmark must still pass that
benchmark's deterministic correctness check before you open a PR:

```sh
# accumulator
./build/benchmarks/accumulator/fx64_bench  --verify 0
./build/benchmarks/accumulator/fx128_bench --verify 0
./build/benchmarks/accumulator/fp_bench_a  --verify 0

# search
python3 benchmarks/search/tools/verify_equivalence.py build/benchmarks/search
```

These are the same checks CI runs on every push/PR (see
`.github/workflows/ci.yml`).

## Benchmarking expectations

Do not commit new timed results (anything under a benchmark's `results/`
directory) from ad hoc runs on shared or virtualised machines — they are
not comparable to the existing pinned-machine data and will be misleading.
If you want to update `results/`, say in the PR how and where you
regenerated them (see each benchmark's README for the exact commands) and
note the hardware/OS used.
