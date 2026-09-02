# asm_check.md — Emitted Implementation and Dependency Verification

Brief section 4.1 (loop-carried dependency, carry propagation, register
pressure) and section 4.2 (FP-A / FP-B instruction-equivalence precondition).

Disassembly: `objdump -d --no-show-raw-insn`, GNU binutils 2.46.
Symbol in all four binaries: `spike::(anonymous namespace)::run_loop`, declared
`__attribute__((noinline, noclone))` so that exactly one copy of the timed loop
exists per binary and is shared by the warm-up and the timed run.

The mathematical closed form is `x0 + t*Delta mod 1`. The checks below establish
that no such closed form was substituted in any variant.

## Hot-loop shape

Measured by `tools/loop_stats.py`, which locates the body as the span from the
backward branch target up to and including the branch:

```text
name    range         insns vector  stack  loads
FP-A    1660-1713    39     36      0      0
FP-B    16a0-1753    39     36      0      0
FX64    1660-16bf    23     20      0      0
FX128   16e0-1839    71     68     20     20
```

FP and FX64 bodies are entirely register-resident. FX128 is not; see section 4.

---

## 1. FX64 — `src/fixed_u64.cpp`

Register roles, from the SysV vector argument order
`run_loop(ctx, n, vd, vwb, vtb, vones)`:

```text
%rdi   Ctx*                         %rsi   n (trip count)
%ymm4  D64 broadcast                %ymm1  Wb = W_eps + 2^63
%ymm2  Tb  = 2*W_eps + 2^63         %ymm3  lane constant 1
%ymm8  %ymm7  %ymm6  %ymm5          the four YMM state vectors
%ymm12 %ymm11 %ymm10 %ymm9          the four running accumulators
```

Hot loop, 23 instructions, 20 of them vector:

```asm
    1660:	vpaddq %ymm4,%ymm8,%ymm8
    1664:	vpaddq %ymm4,%ymm7,%ymm7
    1668:	add    $0x1,%rax
    166c:	vpaddq %ymm1,%ymm8,%ymm0
    1670:	vpaddq %ymm4,%ymm6,%ymm6
    1674:	vpcmpgtq %ymm0,%ymm2,%ymm0
    1679:	vpaddq %ymm4,%ymm5,%ymm5
    167d:	vpand  %ymm3,%ymm0,%ymm0
    1681:	vpaddq %ymm12,%ymm0,%ymm12
    1686:	vpaddq %ymm1,%ymm7,%ymm0
    168a:	vpcmpgtq %ymm0,%ymm2,%ymm0
    168f:	vpand  %ymm3,%ymm0,%ymm0
    1693:	vpaddq %ymm11,%ymm0,%ymm11
    1698:	vpaddq %ymm1,%ymm6,%ymm0
    169c:	vpcmpgtq %ymm0,%ymm2,%ymm0
    16a1:	vpand  %ymm3,%ymm0,%ymm0
    16a5:	vpaddq %ymm10,%ymm0,%ymm10
    16aa:	vpaddq %ymm1,%ymm5,%ymm0
    16ae:	vpcmpgtq %ymm0,%ymm2,%ymm0
    16b3:	vpand  %ymm3,%ymm0,%ymm0
    16b7:	vpaddq %ymm9,%ymm0,%ymm9
    16bc:	cmp    %rax,%rsi
    16bf:	jne    1660
```

Observations:

1. **Loop-carried state.** Each of `%ymm8 %ymm7 %ymm6 %ymm5` is both a source and
   the destination of `vpaddq %ymm4,%ymmN,%ymmN`. The register written at
   iteration `t` is the register read at iteration `t+1`. Four distinct chains
   survive; they were not collapsed into one despite identical initial values,
   because `init_ctx` forces every vector through an opaque
   `asm volatile("" : "+x"(v))` barrier before storing it.
2. **Not recomputed from an induction variable.** `%rax` is incremented by one and
   compared against `%rsi`; it never scales, multiplies or otherwise feeds the
   state.
3. **Not a closed form.** Nothing computes `X0 + t*D64`.
4. **Not eliminated.** Accumulators stay live; the epilogue stores all eight
   vectors back into `Ctx`.
5. **No unrolling**, one copy of the four-state body per back-edge.
6. **Register-only body**, zero stack or memory references inside the loop.
7. **Wrap.** No wrap instruction exists; modulo-`2^64` is inherent in `vpaddq`.
8. **Detection and accumulation.** `vpaddq %ymm1,%ymmN,%ymm0` forms the biased
   `y`; `vpcmpgtq %ymm0,%ymm2,%ymm0` evaluates `Tb >s yb`, which is
   `y <u 2*W_eps`; `vpand %ymm3,%ymm0,%ymm0` turns the mask into a per-lane 0/1;
   `vpaddq` accumulates. All four vectors, every iteration.
9. **No scalarisation.** No `popcnt`, `vpmovmskb`, `vextract*`, `vmovq` or
   `vpextrq`. The only horizontal reduction is `reduce16`, after timing.
10. **64-bit lanes only**, so no 8-bit or 32-bit sublane can contribute.

---

## 2. FP-A — `src/fp_double.cpp`

Register roles, from `run_loop(ctx, n, vdelta, vone, veps, vome, vones)`:

```text
%rdi   Ctx*                         %rsi   n (trip count)
%ymm9  Delta broadcast              %ymm1  1.0
%ymm2  eps_d                        %ymm3  one_minus_eps_d
%ymm4  lane constant 1              %ymm8 %ymm14  temporaries
%ymm7  %ymm6  %ymm5  %ymm0          the four YMM state vectors
%ymm13 %ymm12 %ymm11 %ymm10         the four running accumulators
```

Hot loop, 39 instructions, 36 of them vector:

```asm
    1660:	vaddpd %ymm7,%ymm9,%ymm7
    1664:	vaddpd %ymm6,%ymm9,%ymm6
    1668:	add    $0x1,%rax
    166c:	vaddpd %ymm5,%ymm9,%ymm5
    1670:	vaddpd %ymm0,%ymm9,%ymm0
    1674:	vcmpge_oqpd %ymm1,%ymm7,%ymm8
    1679:	vandpd %ymm1,%ymm8,%ymm8
    167d:	vsubpd %ymm8,%ymm7,%ymm7
    1682:	vcmpge_oqpd %ymm1,%ymm6,%ymm8
    1687:	vandpd %ymm1,%ymm8,%ymm8
    168b:	vcmpge_oqpd %ymm3,%ymm7,%ymm14
    1690:	vsubpd %ymm8,%ymm6,%ymm6
    1695:	vcmpge_oqpd %ymm1,%ymm5,%ymm8
    169a:	vandpd %ymm1,%ymm8,%ymm8
    169e:	vsubpd %ymm8,%ymm5,%ymm5
    16a3:	vcmpge_oqpd %ymm1,%ymm0,%ymm8
    16a8:	vandpd %ymm1,%ymm8,%ymm8
    16ac:	vsubpd %ymm8,%ymm0,%ymm0
    16b1:	vcmplt_oqpd %ymm2,%ymm7,%ymm8
    16b6:	vorpd  %ymm14,%ymm8,%ymm8
    16bb:	vcmpge_oqpd %ymm3,%ymm6,%ymm14
    16c0:	vpand  %ymm8,%ymm4,%ymm8
    16c5:	vpaddq %ymm13,%ymm8,%ymm13
    16ca:	vcmplt_oqpd %ymm2,%ymm6,%ymm8
    16cf:	vorpd  %ymm14,%ymm8,%ymm8
    16d4:	vcmpge_oqpd %ymm3,%ymm5,%ymm14
    16d9:	vpand  %ymm8,%ymm4,%ymm8
    16de:	vpaddq %ymm12,%ymm8,%ymm12
    16e3:	vcmplt_oqpd %ymm2,%ymm5,%ymm8
    16e8:	vorpd  %ymm14,%ymm8,%ymm8
    16ed:	vcmpge_oqpd %ymm3,%ymm0,%ymm14
    16f2:	vpand  %ymm8,%ymm4,%ymm8
    16f7:	vpaddq %ymm11,%ymm8,%ymm11
    16fc:	vcmplt_oqpd %ymm2,%ymm0,%ymm8
    1701:	vorpd  %ymm14,%ymm8,%ymm8
    1706:	vpand  %ymm8,%ymm4,%ymm8
    170b:	vpaddq %ymm10,%ymm8,%ymm10
    1710:	cmp    %rax,%rsi
    1713:	jne    1660
```

Observations:

1. **Loop-carried state.** Each of `%ymm7 %ymm6 %ymm5 %ymm0` is the destination of
   `vaddpd %ymmN,%ymm9,%ymmN` and then of `vsubpd %ymm8,%ymmN,%ymmN`; the
   post-wrap register is what the next iteration adds `Delta` to.
2. **Not recomputed from an induction variable, and not a closed form.** `%rax`
   only counts iterations; no `vmulpd` and no FMA against an index appears in the
   body, although `-mfma` is enabled.
3. **Not eliminated, no unrolling, register-only body.**
4. **Branchless wrap present, as mandated by section 2.4.**
   `vcmpge_oqpd %ymm1,%ymmN,%ymm8` / `vandpd %ymm1,%ymm8,%ymm8` /
   `vsubpd %ymm8,%ymmN,%ymmN`. The only control transfer is the back-edge.
5. **Detection reads the post-wrap state.** `vcmpge_oqpd %ymm3,%ymm7,%ymm14` at
   `0x168b` is scheduled early, but its source `%ymm7` is already the wrapped
   value written at `0x167d`.
6. **Mask to 0/1 and accumulation** via `vorpd`, `vpand %ymm4`, `vpaddq`.
7. **No scalarisation.**

---

## 3. FP-B and the section 4.2 precondition

FP-B is the same `fp_double.cpp` object linked in a different order and behind a
padding translation unit (`src/padding.cpp`). The hot loop moves from `0x1660` to
`0x16a0`, so link layout demonstrably changed.

`tools/asm_norm.py` drops the symbol header and the absolute address column and
rewrites branch targets as offsets relative to the following instruction, so it
compares mnemonics and operands while ignoring absolute addresses and the offsets
introduced by link order or padding:

```text
$ python3 tools/asm_norm.py results/asm_fp_a.txt results/asm_fp_b.txt
instruction count: 60 vs 60
RESULT: MATCH instruction-for-instruction (mnemonics + operands; absolute
addresses and link-layout offsets normalised away)
```

**Precondition satisfied.** The FP-A/FP-B pair reflects link-layout difference
only and is a valid noise-floor reference.

---

## 4. FX128 — `src/fixed_u128.cpp`

`Ctx` layout: `lo[0..3]` at `0x00 0x20 0x40 0x60`, `hi[0..3]` at
`0x80 0xa0 0xc0 0xe0`, `acc[0..3]` at `0x100 0x120 0x140 0x160`.
Constants arrive through `%rdx` (`const Consts*`).

Register and slot roles, read off the prologue and body:

```text
%rdi   Ctx*            %rsi   n (trip count)      %rdx   Consts*
lo lanes   %ymm13 %ymm12 %ymm10 %ymm9    (registers)
hi lanes   %ymm8  %ymm7  %ymm6  %ymm5    (registers)
accumulators   -0x78(%rsp) -0x58(%rsp) -0x18(%rsp) -0x38(%rsp)   (stack)
constants in registers   %ymm14 D_lo   %ymm4 D_lo_b   %ymm3 W_lo_b
                         %ymm11 T2_lo_b   %ymm2 T2_hi_b
constants in stack slots 0x8(%rsp) D_hi   0x28(%rsp) W_hi_b   0x48(%rsp) ones
```

First of the four state pairs, from the 71-instruction body (the other three are
the same sequence on the other registers):

```asm
    16e0:	vpaddq %ymm13,%ymm4,%ymm0        ; slo_b = lo0 + D_lo_b
    16e5:	vpaddq 0x8(%rsp),%ymm8,%ymm8     ; hi0  += D_hi
    16eb:	add    $0x1,%rax
    16ef:	vpcmpgtq %ymm0,%ymm4,%ymm0       ; carry = (slo <u D_lo)
    16f4:	vpaddq %ymm14,%ymm13,%ymm13      ; lo0  += D_lo          (loop-carried)
    16ff:	vpaddq %ymm13,%ymm3,%ymm15       ; ylo_b = lo0 + W_lo_b
    1710:	vpsubq %ymm0,%ymm8,%ymm8         ; hi0  -= carry         (loop-carried)
    1714:	vpcmpgtq %ymm15,%ymm3,%ymm0      ; c2   = (ylo <u W_lo)
    1719:	vpcmpgtq %ymm15,%ymm11,%ymm15    ; llt  = (ylo <u T2_lo)
    171e:	vpsubq %ymm0,%ymm8,%ymm0         ; yhi  = hi0 - c2
    1722:	vpaddq 0x28(%rsp),%ymm0,%ymm0    ; yhi_b = yhi + W_hi_b
    1728:	vpcmpeqq %ymm0,%ymm2,%ymm1       ; heq  = (yhi == T2_hi)
    172d:	vpcmpgtq %ymm0,%ymm2,%ymm0       ; hlt  = (yhi <u T2_hi)
    1732:	vpand  %ymm1,%ymm15,%ymm1
    1736:	vpor   %ymm0,%ymm1,%ymm1         ; det
    173e:	vpand  0x48(%rsp),%ymm1,%ymm1    ; mask -> 0/1
    1749:	vpaddq -0x78(%rsp),%ymm1,%ymm15  ; acc0 += 0/1
    1754:	vmovdqa %ymm15,-0x78(%rsp)       ; acc0 written back
```

Observations:

1. **Carry propagation is present and consumed every iteration (section 4.1
   point 5).** `vpcmpgtq %ymm0,%ymm4,%ymm0` at `0x16ef` produces the carry out of
   the low lane as an all-ones/zero mask, and `vpsubq %ymm0,%ymm8,%ymm8` at
   `0x1710` consumes it into the high lane — subtracting `-1` adds one. This pair
   appears four times per iteration, once per state pair, with no deferral or
   batching. The carry compare is `slo <u D_lo`, which is equivalent to
   `slo <u lo` and lets the comparison run against a pre-biased constant.
2. **Loop-carried state, both lanes.** `%ymm13` is read and written by
   `vpaddq %ymm14,%ymm13,%ymm13`; `%ymm8` is read and written by the
   `vpaddq`/`vpsubq` pair. The high lane additionally depends on the low lane of
   the same iteration through the carry, which is precisely the mandated
   coupling: the two lanes do not evolve independently.
3. **Not recomputed from an induction variable, and not a closed form.** `%rax`
   only counts iterations. Nothing forms `X0_128 + t*D_128`.
4. **No unrolling**, one copy of the four-pair body per back-edge.
5. **No floating point.** The body contains no `vaddpd`, `vcmppd`, `vcvt*` or any
   other FP instruction; there is no reconstruction of the 128-bit value to
   `double`.
6. **Detection.** The unified window is used, mirroring FX64:
   `y = state + W_128` with its own carry (`vpcmpgtq` at `0x1714` consumed by
   `vpsubq` at `0x171e`), then `y <u 2*W_128` as
   `(y_hi <u T2_hi) || (y_hi == T2_hi && y_lo <u T2_lo)` — `vpcmpgtq`,
   `vpcmpeqq`, `vpcmpgtq`, `vpand`, `vpor`.
7. **Mask to 0/1 and accumulation** via `vpand` against the `ones` broadcast and
   `vpaddq`, every iteration, all four pairs, all 16 logical 128-bit states.
8. **No scalarisation**, no `popcnt`, no extract; the only horizontal reduction is
   `reduce16`, after timing.
9. **64-bit lane operations only** (`vpaddq`, `vpsubq`, `vpcmpgtq`, `vpcmpeqq`,
   `vpand`, `vpor`), so no 8-bit or 32-bit sublane can contribute a count.

### 4.1 Register pressure and spills — required explicit statement

**No live state is spilled.** All eight state vectors — `lo[0..3]` in
`%ymm13 %ymm12 %ymm10 %ymm9` and `hi[0..3]` in `%ymm8 %ymm7 %ymm6 %ymm5` — are
loaded once in the prologue, stay in registers across the whole loop, and are
stored once in the epilogue directly from those registers (`0x1869`–`0x187c`).
No `lo` or `hi` value is written to or reloaded from the stack inside the body.
The section 8 stop condition on state spills is therefore **not** triggered.

**What is spilled.** The 20 stack references per iteration are:

```text
4 x accumulator read-modify-write   vpaddq -0xNN(%rsp),... + vmovdqa ...,-0xNN(%rsp)
4 x D_hi     memory operand         vpaddq 0x8(%rsp),%ymmN,%ymmN
4 x W_hi_b   memory operand         vpaddq 0x28(%rsp),%ymm0,%ymm0
4 x ones     memory operand         vpand  0x48(%rsp),%ymm1,%ymm1
```

i.e. four accumulator spill slots (8 instructions: 4 loads folded into `vpaddq`
plus 4 explicit `vmovdqa` stores) and three constants held in stack slots and
used as folded memory operands. This is arithmetically unavoidable at this
working-set size: FX128 needs 8 state + 4 accumulator + 8 constant YMM values =
20 live vectors against 16 architectural YMM registers. FP and FX64 each need
4 + 4 + 4 or 5 = 12 or 13 and fit comfortably, which is why their bodies have
zero stack traffic.

**Effort already spent reducing it.** A first version used explicit `vpxor`
sign-biasing and carried `sign`, `W_lo` and `W_hi` as separate constants; that
version spilled `hi[3]` to the stack inside the loop, which *would* have been a
section 8 stop condition. Folding the bias into pre-biased constants — using
`x + 2^63 == x ^ 2^63` modulo `2^64`, the same fold FX64 already uses — removed
three `vpxor`s and two live constants per state and moved the last state vector
back into a register. The remaining accumulator and constant spills cannot be
removed without changing the accumulator structure away from the four per-lane
accumulators that FP and FX64 use, which would make the comparison less direct
rather than more.

**Consequence for the comparison.** Per section 1.2, spills of live *state* would
invalidate a clean comparison; none occurred. The accumulator and constant stack
traffic is a genuine, measured cost of FX128's register demand and is reported as
a secondary mechanism in `summary.md` section 8, not corrected for.

---

## 5. Data-dependency structure, side by side

Per state (or state pair), per iteration:

```text
FX64   loop-carried:  vpaddq                                      (1 dependent op)
       side chain:    vpaddq -> vpcmpgtq -> vpand -> vpaddq(acc)

FP     loop-carried:  vaddpd -> vcmpge_oqpd -> vandpd -> vsubpd    (4 dependent ops)
       side chain:    vcmplt_oqpd / vcmpge_oqpd -> vorpd -> vpand -> vpaddq(acc)

FX128  loop-carried lo:  vpaddq                                    (1 dependent op)
       loop-carried hi:  vpaddq -> vpsubq(carry)                   (2 dependent ops,
                         with the carry arriving from the lo chain of the same
                         iteration via vpaddq -> vpcmpgtq)
       side chain:    vpaddq -> vpcmpgtq(c2) -> vpsubq -> vpaddq -> vpcmpgtq/vpcmpeqq
                      -> vpand -> vpor -> vpand -> vpaddq(acc)
```

In every variant, detection and accumulation hang off the state and never feed
back into it. FX128's loop-carried depth is 2 on the high lane and 1 on the low
lane — shallower than FP's 4 — yet FX128 is the slowest variant, because its
per-iteration operation count is far larger.

Vector operations per iteration: **FP 36, FX64 20, FX128 68.** Ratios against
FX64: FP 1.80, FX128 3.40. `summary.md` section 8 weighs these against the
measured tick ratios.

Post-disassembly context only, cited to interpret the emitted instructions and
not to construct any numerical bound: on Zen 3 both AVX2 floating-point and AVX2
integer SIMD issue to the same set of 256-bit vector pipes, so all three loops
contend for the same execution resources. No latency, throughput or port figure
was used to form the hypothesis or to predict the result.

---

## 6. Full extracted listings

Complete `run_loop` disassembly for each binary, prologue and epilogue included:

```text
results/asm_fp_a.txt
results/asm_fp_b.txt
results/asm_fx64.txt
results/asm_fx128.txt
```
