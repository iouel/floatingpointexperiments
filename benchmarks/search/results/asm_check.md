# Gate 4 and Gate 5 — Disassembly Check

Evidence: `asm_fp_a.txt`, `asm_fp_b.txt`, `asm_fx64.txt` (full disassembly of
the timed function `run_search<false>`) and `inner_fp_a.txt`, `inner_fp_b.txt`,
`inner_fx64.txt` (the recurrence loop alone), produced by
`tools/extract_asm.py` from `objdump -d -C --no-show-raw-insn`.

The timed function contains three loops — the `std::gcd` guard, the candidate
loop over `z`, and the `t`-loop carrying the recurrence. The recurrence loop is
selected as the innermost *counted* loop (its backedge is preceded by a trip
counter decrement); the `gcd` guard is innermost too but exits on a data
comparison, not a counter. Both leaf loops are listed in the tool's output so
the selection can be checked.

```text
name   inner range   insns  ymm  stack  memops  condjmp   all leaf loops
fp_a   1bc0-1bed      13     0     0       1       0      1b40-1b52,1b46-1b58,1bc0-1bed
fp_b   1b00-1b2d      13     0     0       1       0      1a80-1a92,1a86-1a98,1b00-1b2d
fx64   1c00-1c0d       5     0     0       0       0      1b60-1b75,1b67-1b7c,1c00-1c0d
```

`condjmp` excludes the loop backedge itself. Whole-function instruction counts:
FP-A 85, FP-B 85, FX64 99.

## FP recurrence loop — 13 instructions

```text
1bc0:  vaddsd      %xmm3,%xmm0,%xmm0          x += alpha
1bc4:  vcmpge_oqsd %xmm2,%xmm0,%xmm1          mask = (x >= 1.0)
1bc9:  vandpd      %xmm2,%xmm1,%xmm1          mask & 1.0
1bcd:  vsubsd      %xmm1,%xmm0,%xmm0          x -= (1.0 or +0.0)
1bd1:  vcomisd     %xmm0,%xmm4                eps_d ? x
1bd5:  seta        %al
1bd8:  vcomisd     0x1740(%rip),%xmm0         x ? one_minus_eps_d
1be0:  setae       %cl
1be3:  or          %ecx,%eax
1be5:  movzbl      %al,%eax
1be8:  add         %eax,%esi                  hits += 0/1
1bea:  sub         $0x1,%edx
1bed:  jne         1bc0
```

- **No conditional jump inside the loop.** The only branch is the backedge, so
  the wrap is genuinely branchless and carries no misprediction cost.
- **Loop-carried chain is 4 deep:** `vaddsd -> vcmpge_oqsd -> vandpd -> vsubsd`,
  all on `%xmm0`.
- **Detection is off the critical path.** Both `vcomisd` reads consume the
  post-wrap `%xmm0` but write only flags, feeding `%esi` (`hits`), which is not
  an input to the next iteration's state update.
- **State is loop-carried in a register** (`%xmm0`); `alpha` sits in `%xmm3`,
  `1.0` in `%xmm2`, `eps_d` in `%xmm4`.
- **No spills:** zero `(%rsp)` / `(%rbp)` references. The single memory operand
  is the rip-relative load of `one_minus_eps_d` from the read-only constant
  pool — a constant, L1-resident, and off the recurrence chain, not a spill.
- **No autovectorisation:** zero `%ymm` operands; the whole loop is scalar.
- **No closed-form substitution:** the `vaddsd` recurrence is still there.

### Gate 4 fallback was required

The brief's plain source form

```cpp
x -= (x >= 1.0) ? 1.0 : 0.0;
```

compiled under GCC 15.2 `-O3 -mavx2 -mfma` to a **conditional branch**:

```text
1bc4:  vcomisd %xmm1,%xmm0
1bc8:  jb      1bce              ; skip the subtract
1bca:  vsubsd  %xmm1,%xmm0,%xmm0
```

That is precisely the mispredicting baseline the design set out to avoid — the
wrap fires with probability about `alpha` and is unpredictable for most `z`, so
a branch there would inflate the measured FX64 speedup. The brief's prescribed
fallback was applied: the wrap is now written with scalar SSE/AVX intrinsics
(`_mm_cmp_sd(..., _CMP_GE_OQ)` / `_mm_and_pd` / `_mm_sub_sd`), the form already
validated in the accumulator spike. It is semantically identical — it subtracts
exactly `1.0` or `+0.0`, and `x - (+0.0) == x` for every `x >= 0` — and gates 2
and 3 were re-run after the change, still with zero mismatches.

## FX64 recurrence loop — 5 instructions

```text
1c00:  cmp   %rdx,%rsi                 (2W-1) ? (x + W)
1c03:  sbb   $0xffffffff,%r15d         hits += ((x + W) < 2W)
1c07:  add   %rax,%rdx                 x += alpha   (implicit mod 2^64)
1c0a:  sub   $0x1,%ecx
1c0d:  jne   1c00
```

- **Single `add` update, no wrap instruction.** The modular reduction is the
  integer add's own overflow; nothing is emitted for it.
- **Loop-carried chain is 1 deep:** `add %rax,%rdx`.
- GCC folded the `+ kWEps` of the window test into the accumulator itself:
  `%rdx` holds `x + W` directly and `%rsi` holds the constant `2W - 1`
  (`0x5604189374bc6a7`). The test then falls out of the carry flag, and
  `sbb $-1, %r15d` increments `hits` exactly when `(x + W) < 2W`. Detection is
  off the accumulator's critical path: it reads `%rdx` and writes `%r15d`.
- **No conditional jump** other than the backedge; **no spills** (zero stack
  references); **no memory operands at all**; **no `%ymm`** — fully scalar.

### Gate 4 also caught an autovectorisation, which was suppressed

Left to itself, GCC recognised `x += alpha` as an affine induction variable
modulo `2^64`, substituted the closed form `x_t = t*alpha`, and vectorised the
loop **8 lanes wide** — 125 iterations of two YMM `vpaddq` chains with a
`vpcmpgtq` window test and a shuffle-based horizontal reduction:

```text
1d40:  vpaddq      %ymm2,%ymm8,%ymm1
1d44:  vpaddq      %ymm7,%ymm2,%ymm0
1d5d:  vpaddq      %ymm9,%ymm2,%ymm2        ; x += 8*alpha
1d58:  vpcmpgtq    %ymm0,%ymm5,%ymm0
1d8e:  cmp         $0x7d,%r14d              ; 125 iterations x 8 lanes = 1000
```

This is a real and legitimate further advantage of the fixed-point
representation — the FP loop cannot get it, because the conditional wrap makes
its recurrence non-affine and there is no closed form to substitute — but it
measures SIMD width rather than accumulator dependency depth, and the brief
scopes vectorisation out ("Both are scalar... Vectorising across candidates is
a real further optimisation for both sides and is out of scope here"). Gate 4
requires "no autovectorisation and no closed-form substitution of the
recurrence".

`#pragma GCC novector` was therefore applied to the recurrence loop in **both**
implementations, so the two hot loops are held to identical codegen
constraints. The pragma constrains only the compiler's model and emits no
instruction. In FP it is a no-op — that loop was never vectorisable. The
consequence for the reported result is recorded in `summary.md` section 8: the
measured speedup is a *lower bound* on what the fixed-point representation can
deliver on this workload.

## `alpha_fixed` — one `__udivmodti4` per candidate

The 128-bit division in `alpha_fixed` was not strength-reduced; GCC emits an
out-of-line call:

```text
1bc5:  call  1170 <__udivmodti4@plt>
```

There is exactly **one** such call site, in the candidate loop — one call per
candidate, i.e. one per 1000 inner iterations. It is charged entirely to FX64
and has no FP counterpart (FP's per-candidate `alpha` is a single `vdivsd`), so
it makes the reported speedup conservative.

The round-to-nearest-even adjustment compiled branchlessly:

```text
1be4:  lea   (%rdi,%rdi,1),%rdx      twice = 2*r
1be8:  cmp   $0x186a4,%rdx           twice ? N+1
1bf9:  sbb   $0xffffffffffffffff,%rax   q += (twice > N)
```

GCC dropped the tie case (`twice == kN && (q & 1)`) entirely, and correctly so:
`kN = 100003` is odd, so `2*r == kN` is impossible for any integer `r`. An
exact tie can never occur here, and round-half-even degenerates to
round-half-up with the tie branch provably dead.

## Gate 5 — FP-A / FP-B instruction equivalence

`tools/asm_norm.py` run **verbatim** (branch-target normalisation only) reports
MISMATCH on 8 of 85 instructions in the function, 1 of 13 in the inner loop.
Every one of the differences is a rip-relative displacement into the read-only
constant pool and the address objdump resolves it to:

```text
  [57] vcomisd 0x1740(%rip),%xmm0 # 3320   |  vcomisd 0x1808(%rip),%xmm0 # 3328
  [68] vxorpd  0x1747(%rip),%xmm0,%xmm3 # 3350 | vxorpd 0x1817(%rip),%xmm0,%xmm3 # 3360
  ... (6 more, all vmovsd constant loads)
```

These are exactly the "absolute addresses/offsets introduced by link order or
padding" that the normalisation exists to remove; the copied tool simply did
not cover this class, because the accumulator spike's hot loops referenced no
constant pool. An opt-in `--norm-rip` flag was added — it rewrites each
displacement to `RIP` and each resolved pool address to `POOL[k]` by *rank*
among the addresses that listing references, so a genuine difference (different
slots, or the same slots in a different order) would still surface. The default
behaviour is unchanged and the original verdict remains reproducible.

```text
inner loop, rip-normalised:  13 vs 13   RESULT: MATCH instruction-for-instruction
full function, rip-normalised: 85 vs 85 RESULT: MATCH instruction-for-instruction
FP-A vs FX64 (control):      13 vs  5   RESULT: MISMATCH
```

The control comparison confirms the normalisation is not vacuous.

Two independent checks confirm nothing was normalised away:

1. **The referenced constant is the same value.** FP-A's inner loop loads pool
   address `0x3320`, FP-B's loads `0x3328`. Dumping `.rodata`:

   ```text
   FP-A  3320:  448b6ce7 fba9ef3f   -> 0x3fefa9fbe76c8b44
   FP-B  3328:  448b6ce7 fba9ef3f   -> 0x3fefa9fbe76c8b44
   ```

   Both are `one_minus_eps_double_bits` from `constants_expected.txt`.

2. **The object file is byte-identical.** `search_fp.cpp.o` has the same MD5
   (`0e141bf0b5ab3d04a21194a81773be2a`) in both targets, as does
   `reference.cpp.o`. FP-A and FP-B are literally the same compiled code; they
   differ only in link order and in the presence of the padding TU.

**Gate 5: PASS.** The FP-A/FP-B pair is a valid build-layout noise-floor
reference.
