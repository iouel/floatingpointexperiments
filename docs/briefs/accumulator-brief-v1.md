# Minimal Spike Brief — Fixed-Point vs Wrapped FP Accumulator

## Question

Does a 64-bit fixed-point modular accumulator cost fewer **ticks per state** than a wrapped AVX2 `double` accumulator on Zen 3 when detection is included in the timed region?

**Final decision:** one number + one verdict. Do not add scope.

---

# 1. Analytic Prior

## 1.1 Architecture Hypothesis

Zen 3 executes AVX2 floating-point and AVX2 integer SIMD through its vector execution resources. Any difference is therefore expected to arise from the required operations and their dependency structure, not from assuming a separate integer execution domain.

* **FX:** modulo-`2^64` unsigned addition; wrap is inherent in unsigned arithmetic.
* **FP:** floating-point update plus explicit branchless wrap into `[0,1)`.
* **Both:** vector detection, mask extraction, and accumulation are part of the measured workload.

Do **not** assign numerical latency, throughput, or port figures before measurement. Published instruction-level figures may be consulted **after disassembly** only to interpret emitted instructions, not to construct a benchmark premise or numerical bound.

## 1.2 Fixed Workload

**16 independent scalar states = 4 × 256-bit YMM state vectors** in both implementations.

Each YMM state vector contains four independent scalar streams. Additional YMM registers may be used for constants, masks, temporaries, and accumulation.

Mathematical initialization:

```text
x0    = 0x1.123456789abcp-4
Delta = 0x1.6a09e667f3bccp-7
```

Both are exactly representable in `double`.

Canonical FX constants are generated once, outside timing:

```text
X0    = round_nearest_even(x0    × 2^64)
D64   = round_nearest_even(Delta × 2^64)
W_eps = round_nearest_even(eps   × 2^64)
```

The canonical calculation must:

* parse the mathematical values exactly;
* perform scaling and round-to-nearest-even conversion using integer/arbitrary-precision arithmetic or an equivalently exact mechanism;
* **not** depend on host-specific `long double` precision;
* **not** use an intermediate `double` representation of `2^64`.

For each `eps`, record:

```text
x0
x0_double_bits
Delta
Delta_double_bits
X0
D64
eps
eps_double_bits
one_minus_eps_double_bits
W_eps
```

Here `eps` denotes the exact decimal real value `10^-6`, `10^-9`, or `10^-12` for canonical FX generation.

The FP implementation uses:

```text
eps_d            = correctly rounded double representation of eps
one_minus_eps_d = correctly rounded double result of (1.0 - eps_d)
```

Record the bit patterns actually used by the FP binary.

Require:

```text
0 <= X0 < 2^64
0 <= D64 < 2^64
0 < 2*W_eps < 2^64
```

Both implementations must be initialized from the same mathematical `x0` and `Delta`. Do not substitute separately chosen constants.

## 1.3 Recurrence

```text
x[t+1] = (x[t] + Delta) mod 1
```

The timed workload detects the **updated state**:

```text
detect(x[t+1])
```

Each of the four YMM state vectors is loop-carried: the value produced at iteration `t` must be consumed as the corresponding state input at iteration `t+1`.

No numerical cycles-per-iteration or cycles-per-state bound is asserted before measurement.

## 1.4 Hypothesis

```text
FX:
    loop-carried modulo-add state update

FP:
    loop-carried floating-point update + explicit branchless wrap

Both:
    update + detection + mask extraction + accumulation
    measured as one complete timed workload
```

## 1.5 Unknowns

The experiment determines:

* actual cost of the emitted update sequences;
* effect of the loop-carried dependencies;
* interaction between update and detection;
* cost of mask extraction and accumulation;
* actual ticks per logical state.

Do not infer the winner from instruction counts, published latency, published throughput, or nominal port availability.

**The measured result is the answer.**

---

# 2. Recurrence and Detection

| Variant | Source              | State                         | Update   | Wrap                   | Detection              |
| ------- | -------------------- | ----------------------------- | -------- | ---------------------- | ---------------------- |
| FP      | `src/fp_double.cpp`  | `double` in `[0,1)`           | `vaddpd` | branchless subtract    | two compares            |
| FX      | `src/fixed_u64.cpp`  | `uint64_t` fraction of `2^64` | `vpaddq` | implicit modulo-`2^64` | add + unsigned compare  |

## 2.1 FP Detection

The implementation must use the post-wrap state:

```text
detect(x) := (x < eps_d) || (x >= one_minus_eps_d)
```

Detection operates directly on the vector state after the mandatory branchless wrap.

Keep the result as a vector mask until accumulation. Do not scalarize the state before accumulation.

**Per-iteration accumulation (v2 clarification):** each logical 64-bit lane's boolean detection result is converted to a 0/1 value *within the vector domain* (e.g., mask AND against a one-vector) and added into a per-lane running vector accumulator via `vpaddq`. This happens on every timed iteration, for all 16 logical lanes. No scalar extraction, no per-iteration `POPCNT`, and no per-iteration horizontal reduction occur inside the timed region.

Horizontal reduction of the running vector accumulator to a single scalar population count happens **exactly once, after timing**, immediately before the `volatile` sink.

## 2.2 FX Detection

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

AVX2 has no unsigned 64-bit compare. Use sign-bit biasing or an equivalent AVX2 sequence.

**Per-iteration accumulation (v2 clarification):** identical to §2.1 — the boolean detection result per lane is converted to a 0/1 vector value and accumulated via `vpaddq` on every timed iteration. Horizontal reduction to a scalar population count happens exactly once, after timing.

The implementation must not accidentally count 8-bit or 32-bit sublanes. The final reduction must represent exactly **16 logical 64-bit detection results** across the four YMM state vectors.

## 2.3 Required FP Wrap

The FP state must be wrapped after every update:

```cpp
x = _mm256_add_pd(x, delta);
__m256d ge = _mm256_cmp_pd(x, one, _CMP_GE_OQ);
x = _mm256_sub_pd(x, _mm256_and_pd(ge, one));
```

Unwrapped FP is not permitted.

If branchless wrapping cannot be achieved, **stop and report back**.

## 2.4 Sensitivity Values

Run exactly:

```text
eps = 1e-6
eps = 1e-9
eps = 1e-12
```

`eps` is a sensitivity parameter only; it does not gate or otherwise change the workload.

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
* `iterations × logical_states` must match between FP and FX.
* Each logical 64-bit lane contributes exactly one detection Boolean, converted to a 0/1 vector lane and added into the running vector accumulator, on every timed iteration.
* Horizontal reduction of the accumulator to a scalar population count occurs exactly once, after timing — not per iteration.
* No scalar `POPCNT` (or equivalent) instruction may appear inside the timed region for either implementation.
* The timed workload contains the complete update, wrap, detection, mask-to-vector conversion, and vector-accumulation sequence.
* The compiler must not eliminate, duplicate, replace with a closed-form computation, or move the timed workload outside the timing boundaries.
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

Do **not** require FP and FX detection counts to match.

---

# 4. Mandatory Validation

## 4.1 Loop-Carried Dependency and Emitted Instructions

The mathematical closed form is:

```text
x0 + t·Delta mod 1
```

A genuine recurrence dependency must survive `-O3`.

For each implementation, disassembly must demonstrate that:

1. each of the four YMM state vectors produced by iteration `t` is consumed as the corresponding state input at iteration `t+1`;
2. the recurrence is not recomputed from an induction variable;
3. the state is not replaced by a single closed-form computation;
4. the dependency is not otherwise eliminated.

Loop unrolling is permitted. The dependency check must establish that the recurrence remains loop-carried across the unrolled body and across loop iterations; unrolling must not replace it with independent closed-form computations.

Merely finding an update instruction inside the loop is insufficient.

Record the relevant hot-loop assembly in:

```text
results/asm_check.md
```

The recorded region must contain:

* state update;
* wrap/update mechanics;
* detection;
* mask-to-vector conversion and vector-accumulation;
* concise register/data-dependency observations.

Published microarchitectural figures may be cited only as post-disassembly context. They must not construct a numerical bound.

If either dependency chain fails and cannot be fixed without changing the algorithm, **stop and report back**.

## 4.2 Noise Floor

Build FP twice:

```text
FP-A
FP-B
```

Use either:

* different link order; or
* a padding translation unit that changes link layout without changing generated hot-loop instructions.

**Precondition (v2 addition):** before computing `noise_floor_pct`, diff the FP-A and FP-B hot-loop instruction streams from `results/asm_check.md` — mnemonics and operands, ignoring absolute addresses/offsets introduced by link order or padding. They must match instruction-for-instruction. If they do not match, the FP-A/FP-B pair reflects a code-generation difference, not link-layout noise alone, and cannot be used as the noise-floor reference; **stop and report back**.

For each `eps`:

```text
FP-A: 20 process invocations
FP-B: 20 process invocations
FX:   20 process invocations
```

Total:

```text
3 eps × 60 invocations = 180 process invocations
```

Perform noise-floor analysis **before** inspecting the FP-versus-FX comparison.

Analysis order:

```text
1. median(FP-A)
2. median(FP-B)
3. noise_floor_pct
4. pool FP-A + FP-B
5. median(FP)
6. FX_margin_pct
7. fixed verdict rule
```

Noise floor:

```text
noise_floor_pct =
    100 × |median(FP-A) - median(FP-B)|
    / min(median(FP-A), median(FP-B))
```

Per-`eps` FX margin:

```text
FX_margin_pct(eps) =
    100 × (median(FP, eps) - median(FX, eps))
    / median(FP, eps)
```

Positive `FX_margin_pct(eps)` means FX is faster.

## 4.3 Single Decision Number and Verdict

The three `eps` values are sensitivity checks on the same benchmark.

Decision number:

```text
Decision FX_margin_pct =
    median(
        FX_margin_pct(1e-6),
        FX_margin_pct(1e-9),
        FX_margin_pct(1e-12)
    )
```

Decision noise floor:

```text
Decision noise_floor_pct =
    median(
        noise_floor_pct(1e-6),
        noise_floor_pct(1e-9),
        noise_floor_pct(1e-12)
    )
```

Final verdict:

```text
Decision FX_margin_pct <= Decision noise_floor_pct
    -> no margin

Decision FX_margin_pct > Decision noise_floor_pct
    -> margin present
```

The three per-`eps` margins and noise floors must still be reported in `summary.md`; they do not constitute separate final decisions.

**Final decision output:** exactly one `Decision FX_margin_pct` and one verdict.

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

Document the exact inline-assembly constraints or compiler intrinsic/barrier mechanism used to establish the compiler boundary. The implementation must make both CPU ordering and compiler ordering explicit.

Report the timing unit as **`ticks`**, not `cycles`.

Capture `TSC_AUX` from both `RDTSCP` reads.

A run is valid only when:

1. both `TSC_AUX` values match; and
2. the value corresponds to the specified pinned logical CPU.

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

Use identical affinity for FP-A, FP-B, and FX.

The benchmark process is pinned to the specified logical CPU.

Record the SMT sibling state immediately before each run. Unexpected sibling activity during measurement is an environment observation; do not selectively discard runs because of it.

**Warm-up (v2: minimum specified).** Warm-up occurs entirely before timing. Run at least `1e5` full iterations of the identical timed-workload code path (update → detect → vector-accumulate) immediately before the timed region begins, in the same process invocation, to warm the instruction cache, data TLB, and branch predictor. This warm-up is unconditionally excluded from `measured_value`.

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

For each `eps`, interleave or randomize the 60 invocations across:

```text
FP-A
FP-B
FX
```

Use a fixed, recorded run order.

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
-fast-math
```

No other optimization-changing flags may differ between FP-A, FP-B, and FX, except the deliberate link-order/padding change used for the FP noise-floor comparison.

The hot loop must retain:

```text
16 independent logical states
4 YMM state vectors
```

Additional registers may be used for constants, masks, temporaries, and accumulation.

No scalarized replacement of the state representation is permitted.

---

# 7. Output

```text
spike/
├── src/
│   ├── fp_double.cpp
│   ├── fixed_u64.cpp
│   └── harness.cpp
├── CMakeLists.txt
└── results/
    ├── asm_check.md
    ├── raw.csv
    └── summary.md
```

## 7.1 `raw.csv`

Exactly one row per process invocation:

```text
implementation,build_variant,run_id,iterations,logical_states,eps,measured_value,unit,value_per_state,detections
```

No aggregates.

Required values:

```text
build_variant = FP-A | FP-B | FX
logical_states = 16
iterations = 10000000
unit = ticks
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

Exactly the fields required by Section 5, including warm-up iteration count.

### 2. Canonical Constants

For each `eps`, record:

```text
x0
x0_double_bits
Delta
Delta_double_bits
X0
D64
eps
eps_double_bits
one_minus_eps_double_bits
W_eps
```

### 3. Emitted Implementation

Relevant hot-loop assembly and dependency observations from Section 4.1, including the FP-A/FP-B instruction-equivalence check result (§4.2 precondition).

### 4. Measured Result

For each `eps`, report median / minimum / sample standard deviation of `value_per_state` for:

* FP-A, 20 runs;
* FP-B, 20 runs;
* pooled FP, 40 runs;
* FX, 20 runs.

### 5. Noise Floor

For each `eps`, report:

* FP-A median;
* FP-B median;
* `noise_floor_pct`.

### 6. Margin

Report:

* `FX_margin_pct(eps)` for each `eps`;
* final `Decision FX_margin_pct`.

### 7. Boundary Divergences

Report detection-count differences and where they occur.

### 8. Mechanism

Choose exactly one:

* vector execution-resource pressure
* dependency-chain depth
* implicit modulo-`2^64` wrap
* cheaper detection
* unclear

### 9. Verdict

Exactly one:

```text
no margin
```

or:

```text
margin present
```

### 10. Deviations

Any deviation from this brief and why.

## 7.3 Boundary Divergences

FP and FX detection counts are **not required to match**.

Possible causes include:

* different finite-precision state trajectories;
* `2^64` grid representation;
* detection-window boundary representation;
* half-open endpoint semantics.

Record where and by how much.

Expected divergence is not a defect and must not be corrected by altering either algorithm.

---

# 8. Stop Conditions

Stop only if:

* the FP baseline cannot be made to wrap branchlessly;
* the dependency chain fails for either variant and cannot be fixed without changing the algorithm; or
* the FP-A/FP-B hot-loop instruction streams do not match (v2 addition — see §4.2).

Nothing else is a stop condition.

Boundary divergence is recorded, not escalated.

Thermal, frequency, run-order, and ordinary system-state noise are handled by the prescribed measurement and noise-floor procedure.

Missing performance counters are irrelevant because hardware performance counters are not used.

---

# 9. Scope

This measures **ticks per state**.

It does not measure cost per *useful* search state, because the number of states required to produce a useful result depends on the detection predicate and the mathematics behind it.

The measured ticks/state is the answer to this spike.

It does not justify extrapolation to GPU or specialised hardware.