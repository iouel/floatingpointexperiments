## Experiments with FP

Micro-benchmarking spikes investigating whether a fixed-point modular
accumulator is cheaper than a wrapped floating-point one on modern x86
(AVX2/FMA, tested against a Zen 3 target), including a downstream search
that uses the same accumulator recurrence.

## Status

**Experimental / research spike, not a library.** This repository captures a
narrowly scoped performance investigation together with the raw and derived
results collected while running it. It is not a general-purpose
fixed-point/SIMD library, and the code is intentionally written to answer one
question per benchmark rather than to be reused as-is.

Known limitations:

* Results committed under `results/` were captured on a specific, pinned
  development machine (see each benchmark's README) and are **not**
  performance guarantees for any other machine. Treat them as a worked
  example of the methodology, not as a universal verdict.
* The benchmarks use fixed, hand-picked compiler flags (`-O3 -mavx2 -mfma`,
  no `-march=native`, no `-ffast-math`, no LTO) to keep codegen comparable
  across runs; they are not tuned for peak throughput on any given machine.
* `benchmarks/search/` depends on the accumulator question being answered
  favourably before it is worth extending further; see
  `docs/briefs/accumulator-brief-v3.md` for the gating rationale.

## Repository layout

```
.
├── CMakeLists.txt              # root umbrella build (see below)
├── benchmarks/
│   ├── accumulator/            # FX64 vs FX128 vs wrapped-FP accumulator spike
│   │   ├── src/                # timed workload + harness
│   │   ├── tools/               # constant generation, analysis, run scripts
│   │   ├── results/             # committed example run (see its README)
│   │   └── CMakeLists.txt
│   └── search/                 # Kronecker-sequence parameter search spike
│       ├── src/
│       ├── tools/
│       ├── results/
│       └── CMakeLists.txt
├── docs/briefs/                 # the design briefs the spikes implement
└── .github/workflows/           # CI: configure + build + deterministic checks
```

`benchmarks/accumulator` supersedes an earlier iteration of the same spike
that did not yet include the 128-bit fixed-point variant (`FX128`); that
earlier iteration is described by `docs/briefs/accumulator-brief-v1.md` and
is not carried forward as separate code since it is fully subsumed by the
current one (still visible in git history prior to this cleanup).
`docs/briefs/accumulator-brief-v3.md` is the brief the current code
implements.

## Prerequisites

* A C++20 compiler with AVX2/FMA codegen support (developed against GCC on
  Linux; the CI workflow builds with the default GCC on `ubuntu-latest`).
* CMake >= 3.20.
* An x86-64 CPU (or emulation) with AVX2 and FMA, to *run* the resulting
  binaries. Configuring and building does not require it, but running any of
  the executables on a CPU lacking AVX2/FMA will crash with `SIGILL`.
* Python 3 (no third-party packages) for the analysis/regeneration scripts
  under each benchmark's `tools/`.

## Building

From the repository root, both benchmarks configure and build together:

```sh
cmake -S . -B build
cmake --build build -j
```

Each benchmark also remains a fully standalone CMake project and can be
built directly from its own directory, exactly as before this reorganisation:

```sh
cmake -S benchmarks/accumulator -B build-accumulator
cmake --build build-accumulator -j

cmake -S benchmarks/search -B build-search
cmake --build build-search -j
```

Set `-DBUILD_ACCUMULATOR_BENCHMARK=OFF` or `-DBUILD_SEARCH_BENCHMARK=OFF` at
the root to skip either one.

## Running and interpreting results

Each benchmark directory has its own README with the exact invocation for
timing runs, correctness/equivalence checks, and how the files under its
`results/` directory were produced and can be regenerated. In short:

* `benchmarks/accumulator/README.md` — FX64 vs FX128 vs FP accumulator spike.
* `benchmarks/search/README.md` — Kronecker-sequence search spike.

Both benchmarks separate a **deterministic correctness/equivalence check**
(safe to run anywhere, no timing claims) from the **timed run** (pinned
CPU affinity, many repetitions, intended to be run on a quiet, dedicated
machine — not CI). CI only exercises the former; see
`.github/workflows/ci.yml`.

## Contributing

See `CONTRIBUTING.md` for formatting, build, and benchmark expectations
before proposing changes.

## License

Apache License 2.0, see `LICENSE`.
