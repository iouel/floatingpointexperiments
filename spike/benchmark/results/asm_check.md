# asm_check.md — Emitted Implementation and Dependency Verification

Disassembly: `objdump -d -C --no-show-raw-insn`, GNU binutils 2.46.
Symbol in all three binaries: `spike::(anonymous namespace)::run_loop`, declared
`__attribute__((noinline, noclone))` so exactly one copy of the timed loop exists
per binary and is shared by the warm-up and the timed run.

The mathematical closed form is `x0 + t*Delta mod 1`. The checks below establish
that no such closed form was substituted.

## Hot-loop shape

From `tools/loop_stats.py`, which delimits the body as the span from the backward
branch target up to and including the branch:

```text
name    range         insns vector  stack  loads
FP-A    15c0-1673    39     36      0      0
FP-B    1600-16b3    39     36      0      0
FX64    15c0-161f    23     20      0      0
```

**`stack` and `loads` are zero for every variant: no state vector — and in fact
no value at all — is spilled or reloaded inside either hot loop.** The only
memory references in either function are the prologue loads and epilogue stores
against `Ctx` at `0x00..0xe0(%rdi)`, outside the loop.

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
    15c0:	vpaddq %ymm4,%ymm8,%ymm8
    15c4:	vpaddq %ymm4,%ymm7,%ymm7
    15c8:	add    $0x1,%rax
    15cc:	vpaddq %ymm1,%ymm8,%ymm0
    15d0:	vpaddq %ymm4,%ymm6,%ymm6
    15d4:	vpcmpgtq %ymm0,%ymm2,%ymm0
    15d9:	vpaddq %ymm4,%ymm5,%ymm5
    15dd:	vpand  %ymm3,%ymm0,%ymm0
    15e1:	vpaddq %ymm12,%ymm0,%ymm12
    15e6:	vpaddq %ymm1,%ymm7,%ymm0
    15ea:	vpcmpgtq %ymm0,%ymm2,%ymm0
    15ef:	vpand  %ymm3,%ymm0,%ymm0
    15f3:	vpaddq %ymm11,%ymm0,%ymm11
    15f8:	vpaddq %ymm1,%ymm6,%ymm0
    15fc:	vpcmpgtq %ymm0,%ymm2,%ymm0
    1601:	vpand  %ymm3,%ymm0,%ymm0
    1605:	vpaddq %ymm10,%ymm0,%ymm10
    160a:	vpaddq %ymm1,%ymm5,%ymm0
    160e:	vpcmpgtq %ymm0,%ymm2,%ymm0
    1613:	vpand  %ymm3,%ymm0,%ymm0
    1617:	vpaddq %ymm9,%ymm0,%ymm9
    161c:	cmp    %rax,%rsi
    161f:	jne    15c0
```

Observations against the validation requirements:

1. **State vector is both source and destination of its update.** Each of
   `%ymm8 %ymm7 %ymm6 %ymm5` appears as source and destination of
   `vpaddq %ymm4,%ymmN,%ymmN`. The register written at iteration `t` is the
   register read at `t+1`, so the recurrence is genuinely loop-carried. Four
   distinct chains survive despite identical initial values, because `init_ctx`
   forces every vector through an opaque `asm volatile("" : "+x"(v))` barrier
   before storing it.
2. **No closed-form induction.** `%rax` is incremented by one and compared
   against `%rsi`; it never feeds the state. There is no `vmulpd`, no FMA and no
   shift or scale of an index anywhere in the body, so nothing computes
   `X0 + t*D64`.
3. **No loop unrolling.** One copy of the four-state body per back-edge.
4. **No wrap instruction.** Modulo-`2^64` reduction is inherent in `vpaddq`;
   there is nothing corresponding to FP's compare/and/subtract.
5. **Detection via folded bias.** `vpaddq %ymm1,%ymmN,%ymm0` forms the biased
   `y = state + Wb`, and `vpcmpgtq %ymm0,%ymm2,%ymm0` evaluates `Tb >s y`, which
   is exactly `(state + W_eps) <u 2*W_eps`. The bias lives in the constants,
   computed outside the timed region, so no `vpxor` appears in the loop.
6. **Mask to 0/1 and accumulation.** `vpand %ymm3,%ymm0,%ymm0` converts the
   all-ones mask to a per-64-bit-lane 0/1; `vpaddq` accumulates. Every iteration,
   all four state vectors.
7. **No state spills.** Zero stack references in the body.
8. **No scalarisation.** No `popcnt`, `vpmovmskb`, `vextract*`, `vmovq` or
   `vpextrq`. The only horizontal reduction is `reduce16`, after timing.
9. **64-bit lanes only.** `vpcmpgtq`, `vpand` against a `_mm256_set1_epi64x(1)`
   broadcast and `vpaddq` are all 64-bit-lane operations, so no 8-bit or 32-bit
   sublane can contribute a count.

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
    15c0:	vaddpd %ymm7,%ymm9,%ymm7
    15c4:	vaddpd %ymm6,%ymm9,%ymm6
    15c8:	add    $0x1,%rax
    15cc:	vaddpd %ymm5,%ymm9,%ymm5
    15d0:	vaddpd %ymm0,%ymm9,%ymm0
    15d4:	vcmpge_oqpd %ymm1,%ymm7,%ymm8
    15d9:	vandpd %ymm1,%ymm8,%ymm8
    15dd:	vsubpd %ymm8,%ymm7,%ymm7
    15e2:	vcmpge_oqpd %ymm1,%ymm6,%ymm8
    15e7:	vandpd %ymm1,%ymm8,%ymm8
    15eb:	vcmpge_oqpd %ymm3,%ymm7,%ymm14
    15f0:	vsubpd %ymm8,%ymm6,%ymm6
    15f5:	vcmpge_oqpd %ymm1,%ymm5,%ymm8
    15fa:	vandpd %ymm1,%ymm8,%ymm8
    15fe:	vsubpd %ymm8,%ymm5,%ymm5
    1603:	vcmpge_oqpd %ymm1,%ymm0,%ymm8
    1608:	vandpd %ymm1,%ymm8,%ymm8
    160c:	vsubpd %ymm8,%ymm0,%ymm0
    1611:	vcmplt_oqpd %ymm2,%ymm7,%ymm8
    1616:	vorpd  %ymm14,%ymm8,%ymm8
    161b:	vcmpge_oqpd %ymm3,%ymm6,%ymm14
    1620:	vpand  %ymm8,%ymm4,%ymm8
    1625:	vpaddq %ymm13,%ymm8,%ymm13
    162a:	vcmplt_oqpd %ymm2,%ymm6,%ymm8
    162f:	vorpd  %ymm14,%ymm8,%ymm8
    1634:	vcmpge_oqpd %ymm3,%ymm5,%ymm14
    1639:	vpand  %ymm8,%ymm4,%ymm8
    163e:	vpaddq %ymm12,%ymm8,%ymm12
    1643:	vcmplt_oqpd %ymm2,%ymm5,%ymm8
    1648:	vorpd  %ymm14,%ymm8,%ymm8
    164d:	vcmpge_oqpd %ymm3,%ymm0,%ymm14
    1652:	vpand  %ymm8,%ymm4,%ymm8
    1657:	vpaddq %ymm11,%ymm8,%ymm11
    165c:	vcmplt_oqpd %ymm2,%ymm0,%ymm8
    1661:	vorpd  %ymm14,%ymm8,%ymm8
    1666:	vpand  %ymm8,%ymm4,%ymm8
    166b:	vpaddq %ymm10,%ymm8,%ymm10
    1670:	cmp    %rax,%rsi
    1673:	jne    15c0
```

Observations against the validation requirements:

1. **State vector is both source and destination of its update.** Each of
   `%ymm7 %ymm6 %ymm5 %ymm0` is the destination of `vaddpd %ymmN,%ymm9,%ymmN` and
   then of `vsubpd %ymm8,%ymmN,%ymmN`; the post-wrap register is what the next
   iteration adds `Delta` to. Four distinct chains, preserved by the same
   opaque-barrier construction as FX64.
2. **No closed-form induction.** `%rax` only counts iterations. No `vmulpd` and no
   FMA against an index appears in the body, despite `-mfma` being enabled.
3. **No loop unrolling.** One copy of the four-state body per back-edge.
4. **FP wrap is branchless**, exactly as specified: per state vector
   `vcmpge_oqpd %ymm1,%ymmN,%ymm8` / `vandpd %ymm1,%ymm8,%ymm8` /
   `vsubpd %ymm8,%ymmN,%ymmN`. The only transfer of control in the body is the
   loop back-edge; there is no data-dependent branch.
5. **Detection reads the post-wrap state.** `vcmplt_oqpd %ymm2,%ymmN,...` and
   `vcmpge_oqpd %ymm3,%ymmN,...` both consume the register produced by the wrap
   `vsubpd`. Note that `vcmpge_oqpd %ymm3,%ymm7,%ymm14` at `0x15eb` is scheduled
   early, but its source `%ymm7` is already the wrapped value written at
   `0x15dd`.
6. **Mask to 0/1 and accumulation.** `vorpd` joins the two compares, `vpand`
   against `%ymm4` produces a per-64-bit-lane 0/1, `vpaddq` accumulates.
7. **No state spills.** Zero stack references in the body.
8. **No scalarisation.** No `popcnt`, `vpmovmskb` or extract inside the loop.

---

## 3. FP-A / FP-B instruction equivalence

`fp_b` is the same `fp_double.cpp` object linked in a different order and behind
a padding translation unit (`src/padding.cpp`). The hot loop moves from `0x15c0`
to `0x1600` and the function from `0x1580` to `0x15c0`, so link layout
demonstrably changed.

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

**They match exactly.** The FP-A/FP-B pair reflects link-layout difference only
and is a valid noise-floor reference. No stop condition was triggered.

---

## 4. Data-dependency structure, side by side

Per state vector, per iteration:

```text
FX64  loop-carried:  vpaddq                                     (1 dependent op)
      side chain:    vpaddq -> vpcmpgtq -> vpand -> vpaddq(acc)

FP    loop-carried:  vaddpd -> vcmpge_oqpd -> vandpd -> vsubpd   (4 dependent ops)
      side chain:    vcmplt_oqpd / vcmpge_oqpd -> vorpd -> vpand -> vpaddq(acc)
```

In both variants detection and accumulation hang off the state and never feed
back into it; the accumulators form their own independent `vpaddq` chains. The
difference in loop-carried depth is the whole structural difference between the
two hot loops, and the wrap creates it: FX64's wrap costs nothing because it is
inherent in `vpaddq`, whereas FP's wrap places three further dependent vector
operations on the recurrence itself.

Vector operations per iteration: **FP 36, FX64 20** — a ratio of 1.80.
`summary.md` section 8 weighs this against the measured tick ratio.

Post-disassembly context only, cited to interpret the emitted instructions and
not to construct a numerical bound: on Zen 3 both AVX2 floating-point and AVX2
integer SIMD issue to the same set of 256-bit vector pipes, so the two loops
contend for the same execution resources. No latency, throughput or port figure
was used to form the hypothesis or to predict the result.

---

## 5. Full extracted listings

Complete `run_loop` disassembly, prologue and epilogue included:

```text
results/asm_fp_a.txt
results/asm_fp_b.txt
results/asm_fx64.txt
```
