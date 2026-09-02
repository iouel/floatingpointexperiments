# Minimal Spike Brief v3 (as executed) — Multi-Lane Fixed-Point vs Single-Lane Fixed-Point vs Wrapped FP Accumulator

> **Status.** This is the v3 brief with the draft's errors corrected to match the
> workflow that was actually executed. Every correction is itemised in
> **Appendix A**, so this document can be diffed against the original draft.
> Measured outcomes live in `results/summary.md`; this file is the specification
> only.

## Question

Does a multi-lane fixed-point modular accumulator (128-bit, two 64-bit lanes) cost fewer **ticks per state** than either a single-lane 64-bit fixed-point accumulator or a wrapped AVX2 `double` accumulator on Zen 3 when detection is included in the timed region?

**Final decision:** one primary number + one primary verdict. Secondary margins are reported but are not the primary decision.

---

# 1. Analytic Prior

## 1.1 Architecture Hypothesis

Zen 3 executes AVX2 floating-point and AVX2 integer SIMD through its vector execution resources. Any difference is therefore expected to arise from the required operations and their dependency structure, not from assuming a separate integer execution domain.

* **FX64:** modulo-`2^64` unsigned addition; wrap is inherent in unsigned arithmetic.
* **FX128:** true modulo-`2^128` addition via two 64-bit lanes with **mandatory carry propagation**; no implicit independent-lane assumption.
* **FP:** floating-point update plus explicit branchless wrap into `[0,1)`.
* **All:** vector detection, mask extraction, and accumulation are part of the measured workload.

The critical-path hypothesis is qualitative and deliberately cautious:

* **FP:** update add → compare → mask → subtract forms a serial dependency chain.
* **FX64:** the update is a single add; the biased add and compare that implement detection hang off the state as a side chain and are not loop-carried.
* **FX128:** update requires low-lane add, **explicit carry generation**, and high-lane add that consumes the carry. Detection requires a second carry-propagated add and a cross-lane comparison.
  The carry step lengthens the loop-carried chain relative to FX64 and raises the operation count substantially. This cost is real and may erase or reverse any advantage over both FX64 and FP. No assumption is made that FX128 will be faster.

No numerical latency, throughput, or port figures are asserted before measurement.

## 1.2 Fixed Workload

**16 independent scalar states** in all implementations.

**FP and FX64:** 4 × 256-bit YMM state vectors, each containing four independent scalar streams.

**FX128:** 8 × 256-bit YMM state vectors — 4 for high lanes and 4 for low lanes — each pair of YMM registers representing four independent 128-bit logical states.

Additional YMM registers may be used for constants, masks, temporaries, and accumulation.
**Register pressure is a first-class concern.** With 8 YMM state registers already allocated, any spills of live state will be reported and will invalidate claims of a clean comparison. See §4.1 for the mitigations that must be attempted before a spill is treated as a stop condition.

"Independent" here means **independent dependency chains**, not distinct trajectories. All implementations are initialized from the same mathematical `x0` and `Delta` and no separately chosen per-lane constants may be substituted, so every lane follows the same trajectory. Each lane must nevertheless remain a separate loop-carried chain in the emitted code; the initialization must pass every state vector through an opaque barrier so the compiler cannot prove the vectors equal and collapse them. A consequence is that detection totals are exact multiples of 16.

Mathematical initialization:

```text
x0    = 0x1.123456789abcp-4
Delta = 0x1.6a09e667f3bccp-7
```

Both are exactly representable in `double`.

Canonical FX constants are generated once, outside timing.

For FX64:

```text
X0    = round_nearest_even(x0    × 2^64)
D64   = round_nearest_even(Delta × 2^64)
W_eps = round_nearest_even(eps   × 2^64)
```

For FX128:

```text
X0_128    = round_nearest_even(x0    × 2^128)
D_128     = round_nearest_even(Delta × 2^128)
W_eps_128 = round_nearest_even(eps   × 2^128)
```

The FX128 state is split into two 64-bit lanes:

```text
lo = X0_128 mod 2^64
hi = floor(X0_128 / 2^64)
```

Similarly for `D_128` and `W_eps_128`, and for `2*W_eps_128`, which the detection predicate of §2.3 requires.

### 1.2.1 Zero low lanes — mandatory acknowledgement

`x0` and `Delta` are IEEE doubles, so their significands occupy at most 53 bits:

```text
x0    = 0x1123456789abc    × 2^-52
Delta = 0x16a09e667f3bcc   × 2^-59
```

Their exact scalings by `2^128` therefore have **no bits below `2^-64`**, and

```text
X0_128_lo = 0
D_128_lo  = 0
```

exactly. Three consequences follow and must be recorded rather than engineered around:

1. The FX128 low lane holds zero at every iteration and the update carry is always zero. The mandated sequence is branchless, so its **cost** is unaffected and the timing comparison remains valid.
2. FX128 carries no information that FX64 does not: its high lane is bit for bit the FX64 state. The §1.5 unknown "whether 128-bit precision provides measurable benefit over 64-bit" is therefore answered trivially and unfavourably for this constant set — there is no precision benefit to weigh against the carry overhead.
3. **The FX128 constants must be made runtime-opaque.** If `D_128_lo` reaches the optimizer as a compile-time zero, the carry chain mandated by §2.3 and §6 is folded away entirely and disappears from the emitted loop. Every FX128 constant must therefore pass through an opaque barrier (`asm volatile("" : "+x"(v))` or equivalent) before entering the timed loop.

A future revision that genuinely wants to exercise 128-bit precision needs a `Delta` with nonzero bits below `2^-64`, which by construction cannot come from a `double`. Changing `x0` or `Delta` is **out of scope for this brief**: §1.2 forbids substituting separately chosen constants.

### 1.2.2 Canonical generation requirements

The canonical calculation must:

* parse the mathematical values exactly;
* perform scaling and round-to-nearest-even conversion using integer/arbitrary-precision arithmetic or an equivalently exact mechanism;
* **not** depend on host-specific `long double` precision;
* **not** use an intermediate `double` representation of `2^64` or `2^128`.

For each `eps`, record:

```text
x0
x0_double_bits
Delta
Delta_double_bits
X0 (64-bit)
D64
X0_128_lo
X0_128_hi
D_128_lo
D_128_hi
eps
eps_double_bits
one_minus_eps_double_bits
W_eps (64-bit)
2*W_eps (64-bit)
W_eps_128_lo
W_eps_128_hi
2*W_eps_128_lo
2*W_eps_128_hi
```

The FP implementation uses:

```text
eps_d           = correctly rounded double representation of eps
one_minus_eps_d = correctly rounded double result of (1.0 - eps_d)
```

The FP binary must load these by bit pattern from the canonical record, and must be able to print back the bit patterns it actually used, so that the recorded values are provably the executed values.

Require:

```text
0 <= X0 < 2^64
0 <= D64 < 2^64
0 < 2*W_eps < 2^64
0 <= X0_128 < 2^128
0 <= D_128 < 2^128
0 < 2*W_eps_128 < 2^128
```

All implementations must be initialized from the same mathematical `x0` and `Delta`. Do not substitute separately chosen constants.

## 1.3 Recurrence

```text
x[t+1] = (x[t] + Delta) mod 1
```

The timed workload detects the **updated state**:

```text
detect(x[t+1])
```

Each state vector is loop-carried: the value produced at iteration `t` must be consumed as the corresponding state input at iteration `t+1`.

No numerical cycles-per-iteration or cycles-per-state bound is asserted before measurement.

## 1.4 Hypothesis

```text
FP:
    loop-carried floating-point update + explicit branchless wrap
    loop-carried chain: add → compare → mask → subtract   (4 dependent ops)

FX64:
    loop-carried modulo-add state update
    loop-carried chain: add                                (1 dependent op)
    detection is a side chain: biased add → compare

FX128:
    loop-carried true modulo-2^128 addition via two lanes with mandatory carry
    loop-carried chain: low add                            (1 dependent op)
                        high add → subtract-carry          (2 dependent ops),
                        the carry arriving from the low lane of the same iteration
    detection is a side chain, but a long one: a second carry-propagated add
                        followed by a cross-lane comparison
    The carry step is expected to lengthen the chain relative to FX64 and to
    raise the operation count; it may eliminate any advantage over FP.

All:
    update + detection + mask extraction + accumulation
    measured as one complete timed workload
```

## 1.5 Unknowns

The experiment determines:

* actual cost of the emitted update sequences;
* effect of the loop-carried dependencies (including carry);
* interaction between update and detection;
* cost of mask extraction and accumulation;
* whether 128-bit precision provides measurable benefit over 64-bit after carry overhead — see §1.2.1, which constrains what this experiment can answer;
* presence or absence of register spills;
* actual ticks per logical state.

Do not infer the winner from instruction counts, published latency, published throughput, or nominal port availability.

**The measured result is the answer.**

---

# 2. Recurrence and Detection

| Variant | Source | State | Update | Wrap | Detection |
|---------|--------|-------|--------|------|-----------|
| FP | `src/fp_double.cpp` | `double` in `[0,1)` | `vaddpd` | branchless subtract | two compares |
| FX64 | `src/fixed_u64.cpp` | `uint64_t` fraction of `2^64` | `vpaddq` | implicit modulo-`2^64` | add + unsigned compare |
| FX128 | `src/fixed_u128.cpp` | two `uint64_t` lanes per state, true 128-bit | `vpaddq` + **mandatory carry** | implicit modulo-`2^128` | carry-propagated add + cross-lane unsigned compare |

## 2.1 FP Detection

The implementation must use the post-wrap state:

```text
detect(x) := (x < eps_d) || (x >= one_minus_eps_d)
```

Detection operates directly on the vector state after the mandatory branchless wrap.

Keep the result as a vector mask until accumulation. Do not scalarize the state before accumulation.

**Per-iteration accumulation:** each logical 64-bit lane's boolean detection result is converted to a 0/1 value *within the vector domain* (mask AND against a one-vector) and added into a per-lane running vector accumulator via `vpaddq`. This happens on every timed iteration, for all 16 logical lanes. No scalar extraction, no per-iteration `POPCNT`, and no per-iteration horizontal reduction occur inside the timed region.

Horizontal reduction of the running vector accumulator to a single scalar population count happens **exactly once, after timing**, immediately before the `volatile` sink.

## 2.2 FX64 Detection

```text
y = x + W_eps
detect(x) := y < 2*W_eps
```

The window is:

```text
[0, W_eps) ∪ [2^64 - W_eps, 2^64)
```

The addition is ordinary unsigned `uint64_t` arithmetic:

```cpp
uint64_t twoW = W_eps * 2;
uint64_t y    = x + W_eps;   // modulo-2^64 wrap
```

AVX2 has no unsigned 64-bit compare. Use sign-bit biasing or an equivalent AVX2 sequence. Because `x + 2^63 == x ^ 2^63` modulo `2^64`, the bias may be folded into the constants once, outside the timed region, so that the loop emits `vpaddq`/`vpcmpgtq` rather than `vpxor`/`vpcmpgtq`. This fold is explicitly permitted and must be applied consistently across FX64 and FX128.

**Per-iteration accumulation:** identical to §2.1. Horizontal reduction to a scalar population count happens exactly once, after timing.

The implementation must not accidentally count 8-bit or 32-bit sublanes. The final reduction must represent exactly **16 logical 64-bit detection results** across the four YMM state vectors.

## 2.3 FX128 Detection

Each logical state is represented as two 64-bit lanes:

```text
state = (hi << 64) + lo
```

interpreted as a 128-bit fixed-point value on `[0, 1)`.

**Update (true 128-bit addition modulo 2^128 with mandatory carry):**

Let `D_lo` and `D_hi` be the low and high 64-bit lanes of `D_128`.

```text
sum_lo = lo + D_lo               // 64-bit unsigned addition, may wrap
carry  = (sum_lo < lo) ? 1 : 0   // carry out of low lane (must be computed)
lo'    = sum_lo                  // modulo 2^64 automatically
hi'    = hi + D_hi + carry       // 64-bit unsigned addition that consumes carry
```

This performs a genuine 128-bit addition with carry propagation. The two lanes do **not** evolve independently.
The carry must be generated and consumed on every timed iteration for every logical state. Deferred or batched carry is forbidden.

Note that `sum_lo < lo` is equivalent to `sum_lo < D_lo`. The second form is preferred because it compares against a constant, which permits the §2.2 bias fold and reduces register pressure. Both are the same carry.

**Detection (unified window, mirroring §2.2 exactly):**

Let `W_lo`/`W_hi` be the lanes of `W_eps_128` and `T2_lo`/`T2_hi` the lanes of `2*W_eps_128`.

```text
y      = state + W_eps_128        // modulo 2^128, with its own carry propagation
detect := y <u 2*W_eps_128
```

evaluated across lanes as:

```text
y_lo   = lo + W_lo
carry2 = (y_lo <u W_lo)
y_hi   = hi + W_hi + carry2
detect = (y_hi <u T2_hi) || (y_hi == T2_hi && y_lo <u T2_lo)
```

The window is:

```text
[0, W_eps_128) ∪ [2^128 - W_eps_128, 2^128)
```

which is the same pair of half-open intervals FX64 detects, evaluated on the `2^128` grid. This is the same unified formulation §2.2 grants FX64, so neither variant carries a detection-shape handicap the other does not.

`W_eps_128_hi` is **nonzero** for all tested `eps` values (1e-6, 1e-9, 1e-12), so any simplification that assumes `W_hi == 0` is invalid and must not be used. The high-lane comparison and the equality test are both mandatory.

All operations are pure integer SIMD:

* unsigned 64-bit comparisons (sign-bit biasing or equivalent, with the bias folded into constants per §2.2),
* equality comparison against a constant,
* logical AND / OR.

**No conversion to `double`. No floating-point operations in the hot loop. No reconstruction of the 128-bit value to a single scalar.**

**Per-iteration accumulation:** identical to §2.1. Horizontal reduction to a scalar population count happens exactly once, after timing.

The implementation must not accidentally count 8-bit or 32-bit sublanes. The final reduction must represent exactly **16 logical 128-bit detection results** across the eight YMM state vectors.

## 2.4 Required FP Wrap

The FP state must be wrapped after every update:

```cpp
x = _mm256_add_pd(x, delta);
__m256d ge = _mm256_cmp_pd(x, one, _CMP_GE_OQ);
x = _mm256_sub_pd(x, _mm256_and_pd(ge, one));
```

Unwrapped FP is not permitted.

If branchless wrapping cannot be achieved, **stop and report back**.

## 2.5 Sensitivity Values

Run exactly:

```text
eps = 1e-6
eps = 1e-9
eps = 1e-12
```

`eps` is a sensitivity parameter only; it does not gate or otherwise change the workload. It is selected at run time from a table indexed by an argument, so that a single binary per variant serves all three values and the emitted hot loop is provably identical across them.

---

# 3. Timed Workload

For every process invocation:

```text
setup / constants / allocation                          outside timing
            |
            v
update 16 states
    -> detect
    -> convert mask to 0/1 vector lanes
    -> vector-accumulate (vpaddq) into running accumulator
            |
            | repeat exactly 1e7 iterations
            v
horizontal reduce running accumulator -> population count  outside timing
final accumulator -> volatile sink                          outside timing
```

## 3.1 Constraints

* No update-only run.
* No detect-only run.
* No state load/store variant.
* No detection-frequency sweep.
* Exactly `1e7` timed iterations.
* Exactly `16` logical states.
* `iterations × logical_states` must match between FP, FX64, and FX128.
* Each logical lane contributes exactly one detection Boolean, converted to a 0/1 vector lane and added into the running vector accumulator, on every timed iteration.
* Horizontal reduction of the accumulator to a scalar population count occurs exactly once, after timing — not per iteration.
* No scalar `POPCNT` (or equivalent) instruction may appear inside the timed region for any implementation.
* The timed workload contains the complete update, wrap, detection, mask-to-vector conversion, and vector-accumulation sequence.
* The compiler must not eliminate, duplicate, replace with a closed-form computation, or move the timed workload outside the timing boundaries.
* The timed loop must be a single `noinline, noclone` function, so that exactly one copy exists per binary and warm-up and the timed run exercise the same instructions.
* The final accumulator is consumed through a `volatile` sink **after timing**.

Define:

```text
measured_value = end_timestamp - start_timestamp
```

`measured_value` is the raw timestamp delta for the complete timed workload.

```text
value_per_state =
    measured_value / (iterations × logical_states)
```

```text
detections =
    total mask population across all 16 states
    and all 1e7 timed iterations
    (computed via the post-timing horizontal reduction)
```

Do **not** require FP, FX64, and FX128 detection counts to match.

Each binary must additionally provide an out-of-band verification mode that compares its vector detection total against an independent scalar model of one logical lane — `double` for FP, `uint64_t` for FX64, `unsigned __int128` for FX128 — run for the same `1e7` iterations and multiplied by 16. This check runs outside every timed region and must pass for all implementation/eps combinations before the campaign is run.

---

# 4. Mandatory Validation

## 4.1 Loop-Carried Dependency, Carry, and Emitted Instructions

The mathematical closed form is:

```text
x0 + t·Delta mod 1
```

A genuine recurrence dependency must survive `-O3`.

For each implementation, disassembly must demonstrate that:

1. each state vector produced by iteration `t` is consumed as the corresponding state input at iteration `t+1`;
2. the recurrence is not recomputed from an induction variable;
3. the state is not replaced by a single closed-form computation;
4. the dependency is not otherwise eliminated;
5. **for FX128 only:** the carry from the low-lane addition is computed and consumed by the high-lane addition on every iteration.

Because `D_128_lo` is exactly zero (§1.2.1), point 5 is only achievable if the FX128 constants are runtime-opaque. Verifying that the carry instructions are present in the disassembly is therefore also a verification that the opacity barrier worked.

Loop unrolling is permitted. The dependency check must establish that the recurrence remains loop-carried across the unrolled body and across loop iterations; unrolling must not replace it with independent closed-form computations.

Merely finding an update instruction inside the loop is insufficient.

Record the relevant hot-loop assembly in:

```text
results/asm_check.md
```

The recorded region must contain:

* state update (including carry generation and consumption for FX128);
* wrap/update mechanics;
* detection;
* mask-to-vector conversion and vector-accumulation;
* concise register/data-dependency observations;
* **explicit statement on register pressure / spills:** whether any live state (lo/hi vectors) was spilled to the stack inside the hot loop, together with an itemisation of whatever else does occupy stack slots (accumulators, constants) and its per-iteration instruction cost.

### 4.1.1 Mitigations required before declaring a spill stop condition

A state spill is a stop condition only if it survives the following, none of which changes the algorithm:

1. **Fold the sign bias into the constants** (§2.2), removing the explicit `vpxor` per biased value and the live unbiased constants it needs.
2. **Pass loop constants by pointer to an aligned constant block** rather than by value, so the register allocator may choose folded memory operands for them instead of holding them in registers.
3. **Prefer the constant-comparand form of the carry test** (`sum_lo <u D_lo`, §2.3).

If live state still spills after these, **stop and report back**. Accumulator and constant spills are not stop conditions; they are reported and counted as part of the measured cost.

Published microarchitectural figures may be cited only as post-disassembly context. They must not construct a numerical bound.

If any dependency chain fails, if FX128 lacks visible carry propagation, or if state spills cannot be eliminated by §4.1.1, **stop and report back**.

## 4.2 Noise Floor and Statistical Threshold

Build FP twice:

```text
FP-A
FP-B
```

Use either:

* different link order; or
* a padding translation unit that changes link layout without changing generated hot-loop instructions.

Using both together is permitted and was the approach executed.

**Precondition:** before computing `noise_floor_pct`, diff the FP-A and FP-B hot-loop instruction streams from `results/asm_check.md` — mnemonics and operands, ignoring absolute addresses/offsets introduced by link order or padding. They must match instruction-for-instruction. If they do not match, the FP-A/FP-B pair reflects a code-generation difference, not link-layout noise alone, and cannot be used as the noise-floor reference; **stop and report back**.

The diff must be mechanical, not visual: normalise away the symbol header and the absolute address column, and rewrite branch targets as offsets relative to the following instruction.

For each `eps`:

```text
FP-A:  20 process invocations
FP-B:  20 process invocations
FX64:  20 process invocations
FX128: 20 process invocations
```

Total:

```text
3 eps × 80 invocations = 240 process invocations
```

Perform noise-floor analysis **before** inspecting the FP-versus-FX comparisons.

Analysis order:

```text
1. median(FP-A)
2. median(FP-B)
3. noise_floor_pct
4. pool FP-A + FP-B
5. median(FP), stddev(FP)
6. compute decision threshold
7. FX64_margin_pct
8. FX128_margin_pct
9. FX128_vs_FX64_margin_pct
10. verdict rules
```

Noise floor (link-layout sensitivity):

```text
noise_floor_pct =
    100 × |median(FP-A) - median(FP-B)|
    / min(median(FP-A), median(FP-B))
```

Per-`eps` margins:

```text
FX64_margin_pct(eps) =
    100 × (median(FP, eps) - median(FX64, eps))
    / median(FP, eps)

FX128_margin_pct(eps) =
    100 × (median(FP, eps) - median(FX128, eps))
    / median(FP, eps)

FX128_vs_FX64_margin_pct(eps) =
    100 × (median(FX64, eps) - median(FX128, eps))
    / median(FX64, eps)
```

Positive margin means the first-named variant is faster.

**Decision threshold:**

```text
threshold(eps) = max(noise_floor_pct(eps),
                     3 × (stddev(FP_pooled, eps) / median(FP_pooled, eps) × 100))
```

where `stddev` is the sample standard deviation of `value_per_state` across all 40 pooled FP runs for that eps.

This threshold accounts both for link-layout-induced noise and for ordinary run-to-run variability of the FP baseline.

## 4.3 Single Decision Number and Verdict

The three `eps` values are sensitivity checks on the same benchmark.

Decision numbers:

```text
Decision FX64_margin_pct =
    median( FX64_margin_pct(1e-6), FX64_margin_pct(1e-9), FX64_margin_pct(1e-12) )

Decision FX128_margin_pct =
    median( FX128_margin_pct(1e-6), FX128_margin_pct(1e-9), FX128_margin_pct(1e-12) )

Decision FX128_vs_FX64_margin_pct =
    median( FX128_vs_FX64_margin_pct(1e-6),
            FX128_vs_FX64_margin_pct(1e-9),
            FX128_vs_FX64_margin_pct(1e-12) )

Decision threshold =
    median( threshold(1e-6), threshold(1e-9), threshold(1e-12) )
```

### Verdict rule — signed, one-sided

A margin is **present** only when it exceeds the threshold **in the direction that constitutes an advantage for the first-named alternative**:

```text
For each comparison:

    Decision margin_pct >  Decision threshold   -> margin present
    Decision margin_pct <= Decision threshold   -> no margin
```

The signed form is mandatory. An absolute-value rule would classify a variant that is *decisively slower* as "margin present", which inverts the meaning of the §9 gate. Direction must be stated explicitly alongside every verdict: which variant is faster, and by how much.

For completeness and auditability, `summary.md` must also report the absolute-value classification `|Decision margin_pct| > Decision threshold` for every margin, so that a reader applying the stricter-sounding but direction-blind rule can see what it would have returned.

**Primary verdict (answers the gate question):**

```text
FP vs FX128:
    Decision FX128_margin_pct >  Decision threshold  -> margin present (FX128 faster)
    Decision FX128_margin_pct <= Decision threshold  -> no margin
```

The three per-`eps` margins, thresholds, and noise floors must still be reported in `summary.md`; they do not constitute separate final decisions.

**Final decision output:** exactly one primary `Decision FX128_margin_pct` and one primary verdict, with the direction stated. Secondary margins (FX64 vs FP, FX128 vs FX64) are reported but are not the primary decision.

Do not use the FP-versus-FX comparison to select the noise-floor reference.

`stddev` is the **sample** standard deviation of the relevant per-invocation `value_per_state` observations.

---

# 5. Measurement

Use `RDTSCP` with explicit CPU fences:

```text
lfence
rdtscp
lfence

        timed workload

lfence
rdtscp
lfence
```

Each timestamp primitive must also contain an explicit compiler barrier that prevents the compiler from moving operations belonging to the timed workload across that timestamp.

The implementation used, for GCC/Clang:

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

* **CPU ordering** is explicit: both `LFENCE` instructions are emitted literally inside the asm template.
* **Compiler ordering** is explicit: `__asm__ __volatile__` prevents deletion and reordering against other volatile/asm statements; the `"memory"` clobber prevents any memory access from moving across the barrier. Combined with the §3.1 requirement that the timed loop be a `noinline, noclone` function operating on a context object in memory, the timed workload is itself a memory access and cannot cross either barrier.
* **Constraints**: `"=a"` and `"=d"` capture the 64-bit TSC value from EAX/EDX; `"=c"` captures `TSC_AUX` from ECX.

Do **not** write the primitive with `"=r"` outputs plus an `eax`/`edx`/`ecx` clobber list: GCC rejects that as a conflict between an output register and the clobber list. Use the direct register constraints above, or an equivalent that the toolchain accepts. Adjust for other compilers (e.g. MSVC `_ReadWriteBarrier` + `__rdtscp` intrinsic + `_mm_lfence`). The exact mechanism must be documented in `summary.md`.

Report the timing unit as **`ticks`**, not `cycles`. TSC ticks are not core cycles; no conversion to cycles may appear anywhere in the report, and no argument may depend on the tick-to-cycle ratio.

Capture `TSC_AUX` from both `RDTSCP` reads.

A run is valid only when:

1. both `TSC_AUX` values match; and
2. the value corresponds to the specified pinned logical CPU.

The harness enforces this itself and exits non-zero on violation. Under Linux, `IA32_TSC_AUX` is `(node << 12) | cpu`, so the low 12 bits are compared against `sched_getcpu()`. Verify that this holds on the host before the campaign — it is preserved under WSL2, but that is a property of the hypervisor, not a guarantee.

## 5.1 CPU Placement

Pin each process to one specified logical CPU on one physical core.

Record:

* CPU model;
* physical core ID;
* pinned logical CPU ID;
* SMT sibling logical CPU ID;
* SMT sibling state;
* whether the sibling is online/idle;
* turbo/governor state;
* OS;
* whether WSL2.

Use identical affinity for FP-A, FP-B, FX64, and FX128.

Record the SMT sibling state immediately before each run. Unexpected sibling activity during measurement is an environment observation; do not selectively discard runs because of it. Where sibling occupancy is sampled from `/proc/stat`, state the sampling resolution against the invocation duration rather than claiming quiescence.

**Warm-up (minimum specified).** Warm-up occurs entirely before timing. Run at least **5×10⁵** full iterations of the identical timed-workload code path (update → detect → vector-accumulate) immediately before the timed region begins, in the same process invocation, to warm the instruction cache, data TLB, branch predictor, and to reach steady-state behaviour under the higher register pressure of FX128. This warm-up is unconditionally excluded from `measured_value`. State and accumulators are reset after warm-up, outside timing, so that `detections` covers exactly the `1e7` timed iterations.

## 5.2 Build and Timing Metadata

Record:

* compiler and exact version;
* exact compile flags;
* timing mechanism;
* physical core ID;
* pinned logical CPU ID;
* SMT sibling state;
* turbo/governor state;
* warm-up iteration count used.

**Do not use hardware performance counters.**

The experiment does not depend on PMU data.

## 5.3 Run Ordering

For each `eps`, interleave or randomize the 80 invocations across:

```text
FP-A
FP-B
FX64
FX128
```

Use a fixed, recorded run order: a named seed, deterministically derived per eps, with the resulting order written to `results/run_order.txt`.

Do not run all invocations of one implementation contiguously.

Purpose: reduce systematic bias from thermal, frequency, and system-state drift.

---

# 6. Build Constraints

Use exactly:

```text
-O3 -mavx2 -mfma
```

Do not use:

```text
-march=native
-ffast-math
```

A language-standard flag contributed by the build system (e.g. `-std=c++20`) is permitted provided it is byte-identical across all four binaries and no optimisation-level flag is added behind it. `CMAKE_BUILD_TYPE` must be forced empty so the build system contributes no `-O`/`-DNDEBUG` of its own, and IPO/LTO must be off.

No other optimization-changing flags may differ between FP-A, FP-B, FX64, and FX128, except the deliberate link-order/padding change used for the FP noise-floor comparison.

The hot loop must retain:

```text
16 independent logical states
```

FP and FX64: 4 YMM state vectors.
FX128: 8 YMM state vectors (4 lo + 4 hi).

Additional registers may be used for constants, masks, temporaries, and accumulation.

No scalarized replacement of the state representation is permitted.

**FX128 carry propagation:** The generated code for the 128-bit addition must compute the carry from the low-lane addition and feed it into the high-lane addition on every iteration. The compiler is free to use any valid instruction sequence (e.g. `vpaddq` and `vpcmpgtq` for carry detection, `vpsubq` to consume an all-ones mask) as long as the semantics are correct and the carry is not deferred. Constants must be runtime-opaque per §1.2.1, or the carry will be folded away.

---

# 7. Output

```text
spike/
├── src/
│   ├── constants.hpp        (generated by tools/gen_constants.py)
│   ├── spike.hpp            (timing primitive, run contract, shared constants)
│   ├── fp_double.cpp
│   ├── fixed_u64.cpp
│   ├── fixed_u128.cpp
│   ├── padding.cpp          (link-layout padding TU, FP-B only)
│   └── harness.cpp
├── tools/
│   ├── gen_constants.py     (exact canonical constant generation)
│   ├── run_bench.py         (fixed-order campaign driver)
│   ├── asm_norm.py          (FP-A/FP-B instruction-stream diff)
│   ├── loop_stats.py        (hot-loop shape: instruction/vector/stack counts)
│   ├── analyze.py           (§4.2 analysis order, thresholds, verdicts)
│   └── divergence_check.py  (per-iteration detection-index comparison)
├── CMakeLists.txt
└── results/
    ├── asm_check.md
    ├── raw.csv
    ├── summary.md
    ├── constants.md
    ├── run_order.txt
    ├── run_log.json
    ├── analysis.json
    ├── divergence.json
    ├── asm_fp_a.txt
    ├── asm_fp_b.txt
    ├── asm_fx64.txt
    └── asm_fx128.txt
```

`asm_check.md`, `raw.csv` and `summary.md` are the required deliverables; the remainder are the supporting record that makes them reproducible.

## 7.1 `raw.csv`

Exactly one row per process invocation:

```text
implementation,build_variant,run_id,iterations,logical_states,eps,measured_value,unit,value_per_state,detections
```

No aggregates.

Required values:

```text
implementation = FP | FX64 | FX128
build_variant  = FP-A | FP-B | FX64 | FX128
logical_states = 16
iterations     = 10000000
unit           = ticks
```

Definitions:

```text
measured_value =
    end_timestamp - start_timestamp

value_per_state =
    measured_value / (iterations × logical_states)

detections =
    total mask population over all 16 states
    and all timed iterations
    (from the post-timing horizontal reduction)
```

## 7.2 `summary.md`

Use this exact order:

### 1. Environment

Exactly the fields required by Section 5, including warm-up iteration count, the timestamp primitive with its constraints, and the `TSC_AUX` validation result.

### 2. Canonical Constants

For each `eps`, record the list in §1.2.2. Include the §1.2.1 zero-low-lane finding and its three consequences.

### 3. Emitted Implementation

Relevant hot-loop assembly and dependency observations from Section 4.1, including:

* the FP-A/FP-B instruction-equivalence check result (§4.2 precondition);
* confirmation of FX128 carry propagation in the assembly;
* **explicit statement on register pressure / spills** (whether any live lo/hi state was spilled inside the hot loop, and an itemisation of what else occupies stack slots);
* the hot-loop shape table (instructions, vector instructions, stack references per variant);
* the detection cross-check result from §3.1.

### 4. Measured Result

For each `eps`, report median / minimum / sample standard deviation of `value_per_state` for:

* FP-A, 20 runs;
* FP-B, 20 runs;
* pooled FP, 40 runs;
* FX64, 20 runs;
* FX128, 20 runs.

### 5. Noise Floor and Decision Threshold

For each `eps`, report:

* FP-A median;
* FP-B median;
* `noise_floor_pct`;
* `stddev(FP_pooled)` (value_per_state);
* `3 × CV(FP_pooled)` as a percentage;
* `threshold` as defined in §4.2, and which of the two terms set it.

### 6. Margins

Report:

* `FX64_margin_pct(eps)`, `FX128_margin_pct(eps)`, `FX128_vs_FX64_margin_pct(eps)` for each `eps`;
* final `Decision FX64_margin_pct`;
* final `Decision FX128_margin_pct` (primary);
* final `Decision FX128_vs_FX64_margin_pct`;
* final `Decision threshold`.

For each margin, report **both** the signed classification of §4.3 and the absolute-value classification, together with which variant is faster.

### 7. Boundary Divergences

Report detection-count differences and where they occur. Comparing totals is insufficient: compare the **iteration indices** at which each representation detects, pairwise across FP, FX64 and FX128, and report the size of each one-sided difference. Where no divergence occurs, state where it would have appeared first and why it did not.

### 8. Mechanism

Choose exactly one primary mechanism from:

* vector execution-resource pressure
* dependency-chain depth
* carry-propagation overhead
* implicit modulo-`2^64` / `2^128` wrap
* cheaper detection
* register pressure / spills
* unclear

State the primary and list secondary contributions. The argument must be constructible from measured ratios and emitted operation counts alone, without reference to absolute frequency or to the tick-to-cycle ratio. State explicitly that §3.1 forbids the decomposition runs that would isolate the mechanism directly.

### 9. Verdict

Exactly one primary verdict:

```text
no margin
```

or:

```text
margin present
```

stated together with the direction (which variant is faster and by how much), the rule applied, and the resulting answer to the §9 gate. Secondary verdicts for FX64 vs FP and FX128 vs FX64 are also reported.

### 10. Deviations

Any deviation from this brief and why.

## 7.3 Boundary Divergences

FP, FX64, and FX128 detection counts are **not required to match**.

Possible causes include:

* different finite-precision state trajectories;
* `2^64` vs `2^128` grid representation;
* detection-window boundary representation;
* half-open endpoint semantics.

Note that FX64 and FX128 cannot diverge under the constant set of §1.2, for the structural reason given in §1.2.1: with `D_128_lo == 0` the FX128 high lane is bit for bit the FX64 state, and the two windows differ only in bits the trajectory never occupies. This is a property of the constants, not evidence that the two representations agree in general.

Record where and by how much.

Expected divergence is not a defect and must not be corrected by altering any algorithm.

---

# 8. Stop Conditions

Stop only if:

* the FP baseline cannot be made to wrap branchlessly;
* the dependency chain fails for any variant and cannot be fixed without changing the algorithm;
* the FX128 update lacks visible carry propagation or the carry is deferred;
* the FX128 detection cannot be implemented without reconstruction to `double`;
* live state (lo/hi vectors) is spilled inside the hot loop and the spill survives the mitigations of §4.1.1;
* the FP-A/FP-B hot-loop instruction streams do not match (§4.2).

Nothing else is a stop condition.

Accumulator and constant spills are recorded and counted as measured cost, not escalated.

Boundary divergence is recorded, not escalated.

Thermal, frequency, run-order, and ordinary system-state noise are handled by the prescribed measurement and threshold procedure.

Missing performance counters are irrelevant because hardware performance counters are not used.

---

# 9. Scope

This measures **ticks per raw state** for the update+detect loop.

It does **not** measure cost per *useful* search state, because the number of states required to produce a useful result depends on the detection predicate and the mathematics behind it.

This spike is a **preliminary gate**. If FX128 shows no advantage over FP in this microbenchmark — that is, if the primary verdict of §4.3 is "no margin" under the signed rule — the full-search experiment (separate brief) will not be run.

Note the limit established in §1.2.1: because `Delta` is a `double`, this spike measures the *cost* of 128-bit modular arithmetic with carry, but cannot measure any *benefit* of 128-bit precision, since the low lane carries no information. A negative gate result is therefore a statement about cost alone, and is sufficient to close the gate; a positive gate result would not by itself have established that the extra precision is useful.

The measured ticks/state is the answer to this spike.

It does not justify extrapolation to GPU or specialised hardware.

Executed outcome: see `results/summary.md`.

---

# Appendix A — Corrections from the v3 draft

Itemised so this document can be diffed against the original draft. Corrections 1
and 2 were agreed with the brief author before implementation; the remainder are
factual fixes or additions recording what the workflow actually required.

1. **§4.3 verdict rule is now signed and one-sided.** The draft classified a
   margin as present when `|Decision margin_pct| > threshold`. Because §9 asks
   whether FX128 is *faster* than FP, and §1.1 explicitly allows FX128 to be
   slower, the absolute-value rule returns "margin present" for a variant that
   decisively loses — inverting the gate. The signed rule is now mandatory, with
   direction stated, and §7.2 part 6 requires the absolute-value classification to
   be reported alongside so nothing is lost.
2. **§2.3 FX128 detection now uses the unified window.** The draft mandated an
   explicit two-window form (separate bottom and top comparisons) for FX128 while
   §2.2 granted FX64 the cheaper unified `y = x + W`, `y <u 2W` trick. That
   asymmetry handicapped FX128 by roughly three vector operations per state for
   reasons unrelated to the question. FX128 now mirrors FX64 exactly. The detected
   set is identical; `W_eps_128_hi` is still nonzero at every eps and no
   `W_hi == 0` simplification is permitted.
3. **§1.2.1 added — the zero low lanes.** The draft did not anticipate that
   `X0_128_lo` and `D_128_lo` are exactly zero, which follows unavoidably from
   `x0` and `Delta` being doubles. This bounds what the spike can answer (§9), and
   it forces the runtime-opacity requirement without which the mandated carry
   chain is folded away by the optimizer.
4. **§5 timestamp example replaced.** The draft's `read_tscp` declared `"=r"`
   outputs while listing `eax`, `edx` and `ecx` as clobbers; GCC rejects that as an
   output/clobber conflict. The working primitive is now given, with the failure
   mode called out so it is not reintroduced.
5. **§1.1 and §1.4 FX64 critical path corrected.** The draft described FX64's
   critical path as "add → biased-add → compare". Only the add is loop-carried;
   the biased add and compare are a side chain. The same correction is applied to
   the FX128 hypothesis, which distinguishes the loop-carried depth from the
   detection side chain.
6. **§1.2 clarifies what "independent" means.** Sixteen independent dependency
   chains, not sixteen distinct trajectories — the same-`x0`-and-`Delta` rule makes
   all lanes follow one trajectory. The opaque-barrier requirement that stops the
   compiler collapsing the chains is now stated, as is the consequence that
   detection totals are multiples of 16.
7. **§4.1.1 added — mitigations before a spill stop condition.** The first FX128
   implementation spilled `hi[3]` inside the hot loop, which the draft would have
   made an immediate stop condition. Folding the sign bias into the constants
   (§2.2) removed three `vpxor`s and two live constants per state and returned the
   vector to a register. The three mitigations that must be attempted are now
   specified, and accumulator/constant spills are explicitly excluded from the stop
   condition in §8.
8. **§2.2 makes the bias fold explicit and mandatory-consistent**, since
   `x + 2^63 == x ^ 2^63` modulo `2^64`. §2.3 adds the equivalent note that
   `sum_lo <u lo` and `sum_lo <u D_lo` are the same carry, the latter comparing
   against a constant.
9. **§3.1 adds two requirements** that the workflow depended on: the timed loop
   must be a single `noinline, noclone` function shared by warm-up and the timed
   run, and each binary must provide an out-of-band scalar cross-check of its
   detection total.
10. **§2.5 specifies runtime eps selection**, so one binary per variant serves all
    three eps values and the hot loop is provably identical across them.
11. **§4.2 requires the FP-A/FP-B diff to be mechanical**, with the normalisation
    described, rather than a visual comparison. Using both link reordering and a
    padding TU together is explicitly permitted.
12. **§5 forbids tick-to-cycle conversion anywhere in the report**, and §7.2 part 8
    requires the mechanism argument to be constructible without absolute frequency.
13. **§5.1 requires stating sibling-sampling resolution** against invocation
    duration rather than claiming quiescence.
14. **§5.3 requires a named, deterministic seed** and a written run-order file.
15. **§6 permits a build-system language-standard flag** under stated conditions,
    and requires `CMAKE_BUILD_TYPE` empty with IPO/LTO off.
16. **§7 output tree updated** to the tree actually produced, including `tools/`,
    the generated headers, the padding TU and the supporting `results/` files.
17. **§7.1 adds the `implementation` column's permitted values**, which the draft
    specified for `build_variant` only.
18. **§7.2 part 7 strengthened**: divergence must be compared at the level of
    iteration indices, pairwise, not merely as totals; and where none occurs, the
    report must say where it would have appeared first.
19. **§7.3 records that FX64 and FX128 cannot diverge under this constant set**,
    with the structural reason, so their agreement is not misread as evidence about
    the two representations in general.
20. **§9 records the cost/benefit asymmetry**: a negative gate result is a
    statement about cost alone and suffices to close the gate, whereas a positive
    one would not have established that the extra precision is useful.
