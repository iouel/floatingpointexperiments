# summary.md — Multi-Lane Fixed-Point vs Single-Lane Fixed-Point vs Wrapped FP

**Primary decision:** `Decision FX128_margin_pct = -18.34` against
`Decision threshold = 5.11` — **no margin**.

The sign matters: FX128 is *slower* than FP by 18.34%, not faster. Under the
section 9 gate, FX128 shows no advantage over FP, so the full-search experiment
should not be run. See section 9 for the verdict rule actually applied and
section 10 deviation 1 for why.

Secondary: `Decision FX64_margin_pct = 64.02` (**margin present**, FX64 faster
than FP); `Decision FX128_vs_FX64_margin_pct = -230.52` (FX64 faster than FX128
by a factor of about 3.3).

---

## 1. Environment

```text
CPU model                  AMD Ryzen 7 5800X 8-Core Processor
                           (Zen 3, family 25, model 33, stepping 0, AuthenticAMD)
physical core ID           3
pinned logical CPU ID      6
SMT sibling logical CPU ID 7
SMT sibling state          SMT enabled, 2 threads per core; sibling online
sibling online / idle      online; /proc/stat non-idle jiffies for cpu7 summed
                           over all 240 run intervals = 0 (see note below)
turbo / governor state     no cpufreq or governor interface exists inside the
                           WSL2 guest. Windows host active power scheme is
                           "High performance" (GUID 8c5e7fda-...-a6e23a8c635c),
                           PROCTHROTTLEMAX = 100% on AC and DC. PERFBOOSTMODE is
                           not exposed by the active scheme; AMD Precision Boost
                           was left at the host default, i.e. NOT disabled.
OS                         Ubuntu 26.04 LTS, kernel 6.18.33.2-microsoft-standard-WSL2
                           on Windows 11 Home build 26200
WSL2                       yes

compiler                   g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0
compile flags              -O3 -mavx2 -mfma -std=c++20
                           (-std=c++20 is contributed by CMake and is identical
                           for FP-A, FP-B, FX64 and FX128; CMAKE_BUILD_TYPE is
                           forced empty so CMake adds no optimisation flags of
                           its own, and IPO/LTO is off. No -march=native, no
                           -ffast-math.)
link                       FP-A:  harness.o fp_double.o
                           FP-B:  padding.o fp_double.o harness.o
                           FX64:  harness.o fixed_u64.o
                           FX128: harness.o fixed_u128.o
timing mechanism           LFENCE ; RDTSCP ; LFENCE, see below
warm-up iteration count    500000 (5e5) full iterations of the identical timed
                           code path, in the same process, before timing
affinity                   taskset -c 6, identical for all four variants
build system               cmake 4.2.3, GNU binutils 2.46 (objdump)
```

Sibling-activity note: `/proc/stat` has jiffy resolution and a single invocation
takes roughly 15 ms, so this establishes that the sibling carried no sustained
load during the campaign, not that it was quiescent at sub-jiffy granularity. No
run was discarded on account of sibling activity.

### Timing mechanism

```asm
lfence
rdtscp
lfence

        timed workload

lfence
rdtscp
lfence
```

Implemented in `src/spike.hpp`:

```cpp
static inline uint64_t rdtscp_fenced(uint32_t* aux) {
    uint32_t lo, hi, a;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtscp\n\t"
        "lfence"
        : "=a"(lo), "=d"(hi), "=c"(a)
        :
        : "memory");
    *aux = a;
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
```

* **CPU ordering** is explicit: both `LFENCE` instructions bracketing `RDTSCP`
  are emitted literally inside the asm template.
* **Compiler ordering** is explicit: `__asm__ __volatile__` with a `"memory"`
  clobber. `__volatile__` forbids deleting the statement or reordering it against
  other volatile/asm statements; `"memory"` forbids moving any memory access
  across it. The timed workload is a call to `run_loop`, declared
  `__attribute__((noinline, noclone))`, which reads and writes a `Ctx` object in
  memory — it is itself a memory access and cannot be hoisted above the opening
  barrier or sunk below the closing one.
* **Constraints**: `"=a"(lo)` and `"=d"(hi)` capture the TSC value from EAX/EDX;
  `"=c"(a)` captures `TSC_AUX` from ECX. This differs from the example sequence
  offered in brief section 5 — see section 10, deviation 4.
* **Unit**: results are reported as `ticks`, the raw TSC delta. TSC ticks are not
  core cycles and no conversion to cycles was attempted anywhere in this report.

### TSC_AUX validation

Linux programs `IA32_TSC_AUX` as `(node << 12) | cpu`, and this is preserved
under the WSL2 hypervisor (verified: `taskset -c N` yields `TSC_AUX == N` for
N = 0, 3, 6, 11). The harness rejects a run unless both `RDTSCP` reads return the
same `TSC_AUX` **and** its low 12 bits equal `sched_getcpu()`, the pinned logical
CPU 6. All 240 runs passed; 0 rejections, 0 retries, 0 runs discarded.

---

## 2. Canonical Constants

Generated by `tools/gen_constants.py`, which parses the hex-float literals
exactly, scales by `2^64` and `2^128` using `fractions.Fraction` and Python
integers, and applies explicit round-to-nearest-even. No `long double` is used
and no `double` representation of `2^64` or `2^128` is ever constructed. The FP
binary loads `eps_d` and `one_minus_eps_d` by bit pattern, so the values it uses
are provably the recorded ones; `bench --constants` prints them back and they
match for every binary and eps.

Shared by all three eps values:

```text
x0                = 0x1.123456789abcp-4
x0_double_bits    = 0x3fb123456789abc0
Delta             = 0x1.6a09e667f3bccp-7
Delta_double_bits = 0x3f86a09e667f3bcc
X0 (64-bit)       = 0x1123456789abc000
D64               = 0x02d413cccfe77980
X0_128_lo         = 0x0000000000000000
X0_128_hi         = 0x1123456789abc000
D_128_lo          = 0x0000000000000000
D_128_hi          = 0x02d413cccfe77980
```

### eps = 1e-6

```text
eps                       = 1e-6 (exact decimal 1/1000000)
eps_double_bits           = 0x3eb0c6f7a0b5ed8d   (0x1.0c6f7a0b5ed8dp-20)
one_minus_eps_double_bits = 0x3feffffde7210be9   (0x1.ffffde7210be9p-1)
W_eps (64-bit)            = 0x000010c6f7a0b5ee   (18446744073710)
2*W_eps (64-bit)          = 0x0000218def416bdc
W_eps_128_lo              = 0x8d36b4c7f3493858
W_eps_128_hi              = 0x000010c6f7a0b5ed
2*W_eps_128_lo            = 0x1a6d698fe69270b0
2*W_eps_128_hi            = 0x0000218def416bdb
```

### eps = 1e-9

```text
eps                       = 1e-9 (exact decimal 1/1000000000)
eps_double_bits           = 0x3e112e0be826d695   (0x1.12e0be826d695p-30)
one_minus_eps_double_bits = 0x3fefffffff768fa1   (0x1.fffffff768fa1p-1)
W_eps (64-bit)            = 0x000000044b82fa0a   (18446744074)
2*W_eps (64-bit)          = 0x000000089705f414
W_eps_128_lo              = 0xb5a52cb98b405448
W_eps_128_hi              = 0x000000044b82fa09
2*W_eps_128_lo            = 0x6b4a59731680a890
2*W_eps_128_hi            = 0x000000089705f413
```

### eps = 1e-12

```text
eps                       = 1e-12 (exact decimal 1/1000000000000)
eps_double_bits           = 0x3d719799812dea11   (0x1.19799812dea11p-40)
one_minus_eps_double_bits = 0x3fefffffffffdcd1   (0x1.fffffffffdcd1p-1)
W_eps (64-bit)            = 0x0000000001197998   (18446744)
2*W_eps (64-bit)          = 0x000000000232f330
W_eps_128_lo              = 0x12dea11197f27f0f
W_eps_128_hi              = 0x0000000001197998
2*W_eps_128_lo            = 0x25bd42232fe4fe1e
2*W_eps_128_hi            = 0x000000000232f330
```

Constraint checks, all satisfied for every eps: `0 <= X0 < 2^64`,
`0 <= D64 < 2^64`, `0 < 2*W_eps < 2^64`, `0 <= X0_128 < 2^128`,
`0 <= D_128 < 2^128`, `0 < 2*W_eps_128 < 2^128`. `W_eps_128_hi` is nonzero at
every eps, as the brief notes it must be. All three implementations are
initialised from the same mathematical `x0` and `Delta`; no separately chosen
constants were substituted.

### The zero low lanes — a structural finding

**`X0_128_lo` and `D_128_lo` are both exactly zero.** `x0` and `Delta` are IEEE
doubles, so their significands occupy at most 53 bits and their exact scalings by
`2^128` have no bits below `2^-64`:
`x0 = 0x1123456789abc x 2^-52` and `Delta = 0x16a09e667f3bcc x 2^-59`.

Consequences, none of which the brief anticipates:

* The FX128 low lane holds zero at every iteration, and the update carry is
  always zero. The carry is still generated and consumed every iteration as
  section 2.3 requires, and the sequence is branchless, so its **cost** is
  unaffected and the timing comparison remains valid.
* FX128 therefore carries no information that FX64 does not: its high lane is bit
  for bit the FX64 state. Section 1.5's unknown "whether 128-bit precision
  provides measurable benefit over 64-bit after carry overhead" is answered
  trivially and unfavourably — with these constants there is no precision benefit
  at all to weigh against the overhead, because `Delta` has no bits below
  `2^-64` to carry.
* The constants must be made runtime-opaque or the compiler folds `D_lo == 0` and
  deletes the mandated carry chain outright. `src/fixed_u128.cpp` passes every
  constant through an `asm volatile("" : "+x")` barrier for this reason.

Section 1.2 forbids substituting separately chosen constants, so this was
implemented as specified and is reported rather than corrected. A future revision
of the brief that genuinely wants to exercise 128-bit precision needs a `Delta`
with nonzero bits below `2^-64`, which by construction cannot come from a
`double`.

---

## 3. Emitted Implementation

Full listings and the dependency argument are in `results/asm_check.md`; complete
`objdump` extracts are in `results/asm_fp_a.txt`, `results/asm_fp_b.txt`,
`results/asm_fx64.txt` and `results/asm_fx128.txt`.

Hot-loop shape, from `tools/loop_stats.py`:

```text
name    range         insns vector  stack  loads
FP-A    1660-1713    39     36      0      0
FP-B    16a0-1753    39     36      0      0
FX64    1660-16bf    23     20      0      0
FX128   16e0-1839    71     68     20     20
```

**Loop-carried dependency, all variants.** Each state vector lives in a register
inside the loop and appears as both source and destination of its update, so the
value produced at iteration `t` is consumed at `t+1`. The induction variable
`%rax` is only incremented and compared against the trip count `%rsi`; it never
feeds any state, so no recurrence is recomputed from it and no closed form
`x0 + t*Delta` was substituted. No `vmulpd` or FMA against an index appears in
the FP body despite `-mfma` being enabled. No loop is unrolled. The four (FP,
FX64) or eight (FX128) chains were not collapsed despite identical initial
values, because `init_ctx` passes every vector through an opaque
`asm volatile("" : "+x"(v))` barrier before storing it.

**FP wrap.** Branchless and present as section 2.4 requires
(`vcmpge_oqpd` / `vandpd` / `vsubpd` per state vector); the only control transfer
in the body is the loop back-edge, and detection reads the post-wrap register.

**FX64 wrap.** No wrap instruction; modulo-`2^64` is inherent in `vpaddq`.
Unsigned comparison by sign-bit biasing, folded into the addend outside timing.

**FX128 carry propagation — confirmed.** `vpcmpgtq %ymm0,%ymm4,%ymm0` at `0x16ef`
produces the carry out of the low lane as a mask, and `vpsubq %ymm0,%ymm8,%ymm8`
at `0x1710` consumes it into the high lane (subtracting `-1` adds one). The pair
appears four times per iteration, once per state pair, with no deferral or
batching. The detection add `y = state + W_128` carries its own second carry the
same way. The high lane depends on the low lane of the same iteration, so the two
lanes demonstrably do not evolve independently. No floating-point instruction and
no reconstruction to `double` occurs anywhere in the FX128 body.

**Register pressure and spills — explicit statement.** **No live state is
spilled.** All eight FX128 state vectors (`lo` in `%ymm13 %ymm12 %ymm10 %ymm9`,
`hi` in `%ymm8 %ymm7 %ymm6 %ymm5`) are loaded once in the prologue, remain in
registers for the whole loop, and are stored once in the epilogue directly from
those registers. The section 8 stop condition on state spills is not triggered.
What *is* in memory is the four accumulators (read-modify-write each iteration)
and three of the eight constants, used as folded memory operands — 20 stack
references per iteration. This is arithmetically forced: FX128 needs 8 state +
4 accumulator + 8 constant vectors = 20 live YMM values against 16 architectural
registers, where FP and FX64 need 12 and 13 and have zero stack traffic. An
earlier FX128 version that used explicit `vpxor` biasing did spill `hi[3]`, which
would have been a stop condition; folding the sign bias into pre-biased constants
(the same fold FX64 uses) freed the register. See `asm_check.md` section 4.1.

**FP-A / FP-B instruction-equivalence check (section 4.2 precondition).** The hot
loop moves from `0x1660` to `0x16a0`, confirming link layout changed. Comparing
the streams with `tools/asm_norm.py`, which normalises away the absolute address
column and rewrites branch targets as relative offsets:

```text
instruction count: 60 vs 60
RESULT: MATCH instruction-for-instruction (mnemonics + operands; absolute
addresses and link-layout offsets normalised away)
```

**Precondition satisfied.** No stop condition was triggered by any variant.

**Detection correctness cross-check** (outside any timed region): each binary's
vector detection total was compared against an independent scalar model of one
lane — `double` for FP, `uint64_t` for FX64, `unsigned __int128` for FX128 — run
for the same 1e7 iterations and multiplied by 16. All nine implementation/eps
combinations returned `MATCH`.

---

## 4. Measured Result

`value_per_state = measured_value / (1e7 x 16)`, in **ticks per state**.
`stddev` is the sample standard deviation of the per-invocation observations.

### eps = 1e-6

| group | runs | median | minimum | stddev |
| --- | --- | --- | --- | --- |
| FP-A | 20 | 0.732513875 | 0.715765625 | 0.016649920 |
| FP-B | 20 | 0.718660275 | 0.712669338 | 0.015298488 |
| FP pooled | 40 | 0.727251169 | 0.712669338 | 0.016045998 |
| FX64 | 20 | 0.262838400 | 0.249946187 | 0.006728205 |
| FX128 | 20 | 0.859284078 | 0.845986875 | 0.014884221 |

### eps = 1e-9

| group | runs | median | minimum | stddev |
| --- | --- | --- | --- | --- |
| FP-A | 20 | 0.716041957 | 0.711875850 | 0.004486132 |
| FP-B | 20 | 0.716123775 | 0.713152888 | 0.001077646 |
| FP pooled | 40 | 0.716115225 | 0.711875850 | 0.003238407 |
| FX64 | 20 | 0.257657100 | 0.251693000 | 0.003974311 |
| FX128 | 20 | 0.851616219 | 0.843367725 | 0.006895075 |

### eps = 1e-12

| group | runs | median | minimum | stddev |
| --- | --- | --- | --- | --- |
| FP-A | 20 | 0.716121163 | 0.711282813 | 0.016659698 |
| FP-B | 20 | 0.715773225 | 0.710868850 | 0.002267994 |
| FP pooled | 40 | 0.715945887 | 0.710868850 | 0.012206628 |
| FX64 | 20 | 0.255671712 | 0.249217300 | 0.006248361 |
| FX128 | 20 | 0.847223300 | 0.842818863 | 0.014757253 |

240 process invocations (3 eps x 80), all valid, none discarded. Run order was
randomised per eps with a fixed recorded seed and is listed in
`results/run_order.txt`; no implementation ran contiguously. Per-invocation rows
are in `results/raw.csv`.

---

## 5. Noise Floor and Decision Threshold

Computed before any FP-versus-FX comparison, in the order fixed by section 4.2.

```text
noise_floor_pct = 100 * |median(FP-A) - median(FP-B)| / min(median(FP-A), median(FP-B))
threshold(eps)  = max(noise_floor_pct(eps),
                      3 * stddev(FP_pooled, eps) / median(FP_pooled, eps) * 100)
```

| eps | FP-A median | FP-B median | noise_floor_pct | stddev(FP pooled) | 3 x CV % | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-6 | 0.732513875 | 0.718660275 | 1.9277 | 0.016045998 | 6.6192 | 6.6192 |
| 1e-9 | 0.716041957 | 0.716123775 | 0.0114 | 0.003238407 | 1.3567 | 1.3567 |
| 1e-12 | 0.716121163 | 0.715773225 | 0.0486 | 0.012206628 | 5.1149 | 5.1149 |

```text
Decision noise_floor_pct = median(1.9277, 0.0114, 0.0486) = 0.0486
Decision threshold       = median(6.6192, 1.3567, 5.1149) = 5.1149
```

The `3 x CV` term dominates the link-layout noise floor at all three eps values,
so the threshold is set by ordinary run-to-run variability of the FP baseline,
not by link layout.

---

## 6. Margins

```text
FX64_margin_pct(eps)         = 100 * (median(FP)   - median(FX64))  / median(FP)
FX128_margin_pct(eps)        = 100 * (median(FP)   - median(FX128)) / median(FP)
FX128_vs_FX64_margin_pct(eps)= 100 * (median(FX64) - median(FX128)) / median(FX64)
```

Positive means the first-named variant is faster.

| eps | FX64_margin_pct | FX128_margin_pct | FX128_vs_FX64_margin_pct | threshold |
| --- | --- | --- | --- | --- |
| 1e-6 | 63.8586 | -18.1551 | -226.9249 | 6.6192 |
| 1e-9 | 64.0202 | -18.9217 | -230.5231 | 1.3567 |
| 1e-12 | 64.2890 | -18.3362 | -231.3715 | 5.1149 |

Decision numbers:

```text
Decision FX64_margin_pct          =  64.0202
Decision FX128_margin_pct         = -18.3362   (PRIMARY)
Decision FX128_vs_FX64_margin_pct = -230.5231
Decision threshold                =   5.1149
```

Classification. The section 4.3 rule as literally written compares
`|margin|` against the threshold; the primary verdict additionally applies the
signed form (see section 9 and deviation 1). Both are shown:

| comparison | Decision margin | \|margin\| > threshold | signed: margin > threshold | which is faster |
| --- | --- | --- | --- | --- |
| FP vs FX64 | 64.0202 | margin present | margin present | FX64 |
| FP vs FX128 | -18.3362 | margin present | **no margin** | FP |
| FX64 vs FX128 | -230.5231 | margin present | no margin | FX64 |

All three absolute margins clear the threshold by a wide factor, so no comparison
is ambiguous with respect to measurement noise. The only question the sign
resolves is *direction*, and in both FX128 comparisons FX128 is the slower side.

---

## 7. Boundary Divergences

Detection counts are not required to match. In this run **none of the three
representations diverged, at any eps**:

| eps | FP-A | FP-B | FX64 | FX128 |
| --- | --- | --- | --- | --- |
| 1e-6 | 288 | 288 | 288 | 288 |
| 1e-9 | 0 | 0 | 0 | 0 |
| 1e-12 | 0 | 0 | 0 | 0 |

Every one of the 240 invocations reported exactly the value shown; detection is
deterministic and reproduced identically across all runs of a given
implementation and eps. Totals are over all 16 logical states and all 1e7 timed
iterations. All 16 lanes carry the same trajectory (deviation 2), so 288 is
18 detections per lane x 16.

`tools/divergence_check.py` performs a stronger check than equality of totals: it
replays all three scalar models for the full 1e7 iterations and compares the
**iteration indices** at which each detects.

```text
eps = 1e-6    FP 18, FX64 18, FX128 18 hits/lane
              FP vs FX64 (0, 0)   FP vs FX128 (0, 0)   FX64 vs FX128 (0, 0)
eps = 1e-9    FP  0, FX64  0, FX128  0 hits/lane; all pairwise differences empty
eps = 1e-12   FP  0, FX64  0, FX128  0 hits/lane; all pairwise differences empty
```

There is no iteration at which one representation detects and another does not.
Per-index detail is in `results/divergence.json`.

Why divergence was expected but did not appear, and where it would show first:

* **FX64 vs FX128.** These two cannot diverge in this experiment for a structural
  reason, not a numerical coincidence: `D_128_lo` is zero, so the FX128 low lane
  never leaves zero and the FX128 high lane is bit for bit the FX64 state. The
  windows differ only in bits the trajectory never occupies. See section 2.
* **FP vs the fixed-point pair.** At `eps = 1e-6` the window half-width is about
  `1.8447e13` units of `2^-64` while the `double` grid spacing just below `1.0`
  is `2^-53`, i.e. 2048 such units, so the window edges are far from coincident;
  no trajectory point landed between them in 1e7 steps.
* At `eps = 1e-9` and `1e-12` no representation registered any detection at all,
  so agreement there is trivially satisfied and carries no information. The zero
  counts follow from the chosen `Delta` and iteration count and are not a defect;
  no algorithm was altered to change them.

---

## 8. Mechanism

**Primary: vector execution-resource pressure.**

The argument uses measured ratios and emitted operation counts only, and needs no
absolute frequency, so it is independent of the tick-to-cycle relationship. Per
iteration the three loops issue 36 (FP), 20 (FX64) and 68 (FX128) vector
operations. All three bodies are dominated by 256-bit vector work contending for
the same Zen 3 vector pipes. Comparing predicted-by-op-count against measured:

| ratio | from vector op counts | measured (median of the three eps) | excess |
| --- | --- | --- | --- |
| FX128 / FX64 | 3.40 | 3.31 | -3% |
| FX128 / FP | 1.89 | 1.19 | -37% |
| FP / FX64 | 1.80 | 2.78 | +55% |

FX128's cost tracks its operation count almost exactly — 3.31 measured against
3.40 predicted, a 3% shortfall — which is the signature of a loop limited by how
many vector operations it must issue rather than by any dependency. FP is the
outlier in the other direction, running 55% slower than its operation count alone
predicts. So FX128 loses to FP despite the two being far closer in dependency
depth (FX128 carries 2 dependent ops on the high lane and 1 on the low lane; FP
carries 4) purely because 128-bit arithmetic costs 3.4x the vector operations of
64-bit and there is no dependency slack left for those extra operations to hide
in.

**Secondary contributions:**

* **Carry-propagation overhead.** The mandated carry costs 3 of the 17 vector
  operations per state pair in the update (`vpaddq` for the biased sum,
  `vpcmpgtq` to form the carry, `vpsubq` to consume it), and the detection add
  `y = state + W_128` carries a second one. Roughly a fifth of FX128's work is
  carry machinery. It lengthens the high-lane chain from 1 to 2 dependent
  operations, but as the table shows FX128 is not dependency-limited, so the
  carry's cost is felt as issue slots, not as latency.
* **Register pressure / spills.** FX128 needs 20 live YMM values against 16
  registers. No live *state* spilled, but all four accumulators and three
  constants live in stack slots, adding 20 stack references per iteration
  including four explicit `vmovdqa` stores. FP and FX64 have zero stack traffic.
  This is a real component of FX128's operation count, not an artefact.
* **Dependency-chain depth.** This is the mechanism behind the *secondary*
  FP-vs-FX64 result, where FP's 4-deep `vaddpd -> vcmpge_oqpd -> vandpd ->
  vsubpd` recurrence, created by its explicit wrap, makes it 55% slower than its
  operation count predicts while FX64's 1-deep `vpaddq` runs at its issue limit.

Stated limitation: section 3.1 forbids update-only and detect-only runs, so these
attributions rest on measured ratios plus emitted structure rather than on direct
decomposition, and hardware performance counters were not used.

---

## 9. Verdict

**Primary (FP vs FX128):**

```text
no margin
```

`Decision FX128_margin_pct = -18.34`, `Decision threshold = 5.11`. FX128 is
18.34% slower than FP, so it does not clear the threshold in the direction that
would constitute an advantage.

Verdict rule applied. Section 4.3 as literally written classifies a margin as
present when `|Decision margin_pct| > Decision threshold`, which would return
"margin present" here for a variant that is decisively *worse*. That reading
contradicts the section 9 gate, which asks whether FX128 shows an advantage over
FP. The signed one-sided rule was adopted for the primary verdict with the
brief author's agreement (deviation 1): margin present only when
`Decision FX128_margin_pct > Decision threshold`. Both classifications are
reported in section 6 so the literal rule remains recoverable.

**Gate answer (section 9):** FX128 shows no advantage over FP in this
microbenchmark — it is slower than FP and 3.3x slower than FX64. On the brief's
own gate condition, the full-search experiment should not be run on the basis of
FX128.

**Secondary verdicts:**

```text
FP vs FX64:     margin present   (FX64 faster, Decision FX64_margin_pct = 64.02)
FX64 vs FX128:  no margin under the signed rule; FX64 is faster by 230.52%,
                i.e. FX128 costs about 3.3x the ticks per state of FX64.
```

FX64 remains decisively the fastest of the three, reproducing the earlier
two-variant result under the stricter threshold and the longer warm-up.

---

## 10. Deviations

1. **Signed primary verdict rule.** Section 4.3 defines the verdict on
   `|Decision margin_pct|`. Because the section 9 gate asks whether FX128 is
   *faster* than FP, and the section 1.1 prior explicitly allows FX128 to be
   slower, the absolute-value rule would report "margin present" for a decisively
   worse variant. The one-sided signed rule was adopted for the primary verdict
   after raising this with the brief author, who chose it. Section 6 reports both
   classifications for every margin, so nothing is lost.
2. **Unified 128-bit detection window.** Section 2.3 specifies an explicit
   two-window form (separate bottom and top comparisons), while section 2.2 lets
   FX64 use the unified `y = x + W`, `y <u 2W` trick. Implementing that asymmetry
   literally would have handicapped FX128 by roughly 3 of 19 vector operations per
   state relative to FX64 for no reason connected to the question. On raising
   this, the brief author chose the unified form. FX128 therefore computes
   `y = state + W_128` with full carry propagation and tests `y <u 2*W_128` as
   `(y_hi <u T2_hi) || (y_hi == T2_hi && y_lo <u T2_lo)`. The detected set is
   identical to the two-window form, `W_eps_128_hi` is nonzero at every eps and no
   `W_hi == 0` simplification was used, and the result was verified against an
   independent `unsigned __int128` scalar model.
3. **Zero low lanes.** `X0_128_lo` and `D_128_lo` are exactly zero, a consequence
   of `x0` and `Delta` being doubles. Fully documented in section 2. Implemented
   as specified, since section 1.2 forbids substituting different constants; the
   constants are made runtime-opaque so the mandated carry chain survives `-O3`.
4. **Timestamp primitive differs from the example in section 5.** The brief offers
   a sequence that declares `"=r"` outputs while also listing `eax`, `edx` and
   `ecx` as clobbers, which GCC rejects as a conflict between an output register
   and the clobber list. The implementation here uses the direct register
   constraints `"=a"`, `"=d"`, `"=c"` with a `"memory"` clobber, which satisfies
   every requirement the brief states — literal `LFENCE`/`RDTSCP`/`LFENCE`,
   explicit compiler barrier, `TSC_AUX` captured from both reads — and is
   documented in section 1 as required.
5. **All 16 lanes carry the same trajectory.** Section 1.2 requires all
   implementations to be initialised from the same mathematical `x0` and `Delta`
   and forbids separately chosen constants, so every lane starts at `x0` and
   advances by `Delta`. The 16 states are independent as dependency chains —
   four (FP, FX64) or eight (FX128) separate YMM registers, each loop-carried,
   never collapsed — but not as distinct trajectories. This is why detection
   counts are exact multiples of 16.
6. **FX128 accumulator and constant spills.** Reported, not corrected. They are
   forced by the register budget and are part of the measured cost; see
   section 3, section 8 and `asm_check.md` section 4.1. No *state* spill occurred,
   so no stop condition was triggered.
7. **FP-B uses both permitted noise-floor mechanisms.** Section 4.2 offers
   "different link order; or a padding translation unit". FP-B uses both. The
   instruction-equivalence precondition was still satisfied exactly.
8. **`-std=c++20` beyond the mandated flags.** Contributed by CMake; not an
   optimisation-changing flag and byte-identical across all four binaries.
   `CMAKE_BUILD_TYPE` is forced empty so no `-O`/`-DNDEBUG` is added behind it,
   and IPO/LTO is disabled.
9. **Sign-bit bias folded into constants, FX64 and FX128 alike.** Section 2.2
   permits "sign-bit biasing or an equivalent AVX2 sequence". Because
   `x + 2^63 == x ^ 2^63` modulo `2^64`, the bias is applied to the constants once
   outside the timed region, so the loops emit `vpaddq`/`vpcmpgtq` rather than
   `vpxor`/`vpcmpgtq`. Predicates are unchanged and were verified against
   independent scalar models. For FX128 this fold is what keeps all eight state
   vectors in registers.
10. **Extra `tools/` directory and extra `results/` files.** The tree in section 7
    was produced in full (`src/fp_double.cpp`, `src/fixed_u64.cpp`,
    `src/fixed_u128.cpp`, `src/harness.cpp`, `CMakeLists.txt`,
    `results/asm_check.md`, `results/raw.csv`, `results/summary.md`). Alongside it
    are `tools/` (constant generation, run driver, assembly normaliser, loop-shape
    reporter, divergence check) and supporting result files (`constants.md`,
    `run_order.txt`, `run_log.json`, `analysis.json`, `divergence.json`, the four
    `asm_*.txt` extracts), plus generated `src/constants.hpp`, `src/spike.hpp` and
    `src/padding.cpp`. Additions, not substitutions.
11. **Warm-up is exactly the specified minimum**, 5e5 iterations, raised from the
    1e5 used in the earlier two-variant run. Warm-up and timed run share one
    `noinline, noclone` function so they exercise the same instructions; state and
    accumulators are reset between them, outside timing, so the reported detection
    count covers exactly the 1e7 timed iterations.
12. **Turbo could not be disabled from inside the guest.** WSL2 exposes no cpufreq
    or governor interface; boost is host-controlled and was left enabled. Recorded
    as an environment observation per section 5.1 rather than corrected for.
13. **Invalid-run policy.** The driver was prepared to retry a run up to three
    times if `TSC_AUX` disagreed between the two reads or did not match the pinned
    CPU. Zero such runs occurred; none was retried and none discarded, including
    for sibling activity.
