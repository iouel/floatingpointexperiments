#!/usr/bin/env python3
"""Canonical constant generation for the Kronecker-sequence search benchmark.

Every scaling and rounding step uses exact rational / arbitrary-precision
integer arithmetic (fractions.Fraction + int).  `eps` is parsed as an exact
decimal, scaled by 2**64 exactly, and rounded round-to-nearest-even.  No
`double` intermediate is constructed anywhere in the derivation; the only
`float` values produced are the final correctly-rounded doubles whose bit
patterns are emitted for the FP implementation, and each of those comes from
the exact rational by a single correctly-rounded conversion.

The script is also where the two load-bearing preconditions of the parameter
set are proved rather than assumed:

  P1  T is not a multiple of N  ->  hits() genuinely discriminates candidates.
  P2  eps*N is not an integer   ->  no reachable point k/N lands on a window
                                    edge, so FP, FX64 and the exact integer
                                    reference are forced to identical hits.

Outputs:
    src/search_constants.hpp        - consumed by all three implementations
    results/constants.md            - the record reproduced in summary.md
    results/constants_expected.txt  - key=value lines that each binary's
                                      --constants output is checked against
"""

import struct
from fractions import Fraction
from pathlib import Path

TWO64 = 1 << 64

# ---------------------------------------------------------------------------
# The parameter set.
# ---------------------------------------------------------------------------
N = 100003               # prime modulus
T = 1000                 # steps per candidate
EPS_P, EPS_Q = 21, 2000  # eps = 21/2000, the exact decimal 0.0105

EPS = Fraction(EPS_P, EPS_Q)

SAMPLE_Z = [1, 2, 3, 17, 1000, 50000, N - 1]


def round_half_even(fr):
    """round_nearest_even of a non-negative exact rational."""
    assert fr >= 0
    q, r = divmod(fr.numerator, fr.denominator)
    twice = 2 * r
    if twice > fr.denominator:
        q += 1
    elif twice == fr.denominator and (q & 1):
        q += 1
    return q


def dbits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def to_double(fr):
    """Correctly-rounded double nearest an exact rational."""
    return fr.numerator / fr.denominator


def is_prime(n):
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def alpha_fixed(z):
    """Exact round-to-nearest-even of z * 2**64 / N: the FX64 increment."""
    return round_half_even(Fraction(z * TWO64, N))


def min_gap(numer_edge):
    """min over integer k of |k*EPS_Q - numer_edge|."""
    r = numer_edge % EPS_Q
    return min(r, EPS_Q - r)


def main():
    root = Path(__file__).resolve().parent.parent

    # -- P0: N prime, so every z in 1..N-1 is coprime to N -------------------
    assert is_prime(N), "N must be prime"

    # -- P1: T is not a multiple of N ---------------------------------------
    # x_t = ((t*z) mod N)/N.  gcd(z,N)=1 makes multiplication by z a
    # permutation of the residues, so over a full period t=1..N every
    # candidate visits every residue exactly once and hits(z) is the same
    # constant for all z.  T must therefore not be a multiple of N.
    assert T % N != 0, "T is a multiple of N: the search would be degenerate"
    assert T < N, "T >= N weakens the discrimination argument"

    # -- P2: eps*N is not an integer ----------------------------------------
    # Reachable points are exactly k/N.  If eps*N were integral one of them
    # would sit precisely on a window edge and rounding could push it either
    # way, differently per implementation.
    eps_N = EPS * N
    assert eps_N.denominator != 1, "eps*N is an integer: edge collision"

    gap_lo = min_gap(EPS_P * N)                 # lower edge, eps
    gap_hi = min_gap((EPS_Q - EPS_P) * N)       # upper edge, 1-eps
    assert gap_lo > 0 and gap_hi > 0
    clearance = Fraction(min(gap_lo, gap_hi), N * EPS_Q)

    # FP: alpha = fl(z/N) carries <= 2^-53 absolute error and each of the T
    # additions rounds with <= 2^-53 absolute error (the intermediate is < 2,
    # and the conditional `- 1.0` is exact because it cancels within one
    # binade).  Accumulated deviation after T steps is bounded by 2*T*2^-53.
    fp_drift = Fraction(2 * T, 1 << 53)
    # FX64: alpha_fixed is exact round-to-nearest, error <= 1/2 ulp = 2^-65 of
    # the unit interval; the mod-2^64 accumulation adds nothing further.
    fx_drift = Fraction(T, 1 << 65)

    fp_ratio = clearance / fp_drift
    fx_ratio = clearance / fx_drift
    assert fp_ratio > 100000, "edge clearance does not dominate FP drift"
    assert fx_ratio > 100000, "edge clearance does not dominate FX64 drift"

    # -- fixed-point window --------------------------------------------------
    W = round_half_even(EPS * TWO64)
    assert 0 < 2 * W < TWO64, "2*W_eps out of range"

    # -- alpha_fixed cannot overflow 2^64 ------------------------------------
    # q = round(z*2^64/N) <= round((N-1)*2^64/N) = 2^64 - round(2^64/N).
    for z in (1, 2, N - 2, N - 1):
        assert 0 < alpha_fixed(z) < TWO64, z

    # -- doubles for the FP implementation and the shared metric -------------
    eps_d = to_double(EPS)
    ome_d = 1.0 - eps_d
    two_eps_d = to_double(2 * EPS)

    # reference-model intermediates stay well inside uint64
    max_kQ = (N - 1) * EPS_Q
    assert max_kQ < (1 << 63)

    # ---------------------------------------------------------------- header
    hpp = root / "src" / "search_constants.hpp"
    with hpp.open("w", newline="\n") as f:
        f.write("// GENERATED by tools/gen_constants.py - do not edit by hand.\n")
        f.write("// Exact Fraction/integer derivation: no double intermediate, no\n")
        f.write("// long double, and no double representation of 2^64.\n")
        f.write("#pragma once\n#include <cstdint>\n\nnamespace search {\n\n")
        f.write("// --- search parameters -------------------------------------------------\n")
        f.write("inline constexpr uint32_t kN = %du;            // prime modulus\n" % N)
        f.write("inline constexpr uint32_t kT = %du;              // steps per candidate\n" % T)
        f.write("inline constexpr uint32_t kEpsP = %du;              // eps = kEpsP / kEpsQ\n" % EPS_P)
        f.write("inline constexpr uint32_t kEpsQ = %du;\n" % EPS_Q)
        f.write("inline constexpr uint32_t kNumCandidates = %du;\n\n" % (N - 1))
        f.write("// --- FP constants (correctly rounded doubles, as bit patterns) ---------\n")
        f.write("// eps       = %s = %s\n" % (EPS, eps_d.hex()))
        f.write("inline constexpr uint64_t kEpsBits    = 0x%016xull;\n" % dbits(eps_d))
        f.write("// 1 - eps_d = %s\n" % ome_d.hex())
        f.write("inline constexpr uint64_t kOmeBits    = 0x%016xull;\n" % dbits(ome_d))
        f.write("// 2*eps     = %s = %s   (target occupancy in metric_D)\n"
                % (2 * EPS, two_eps_d.hex()))
        f.write("inline constexpr uint64_t kTwoEpsBits = 0x%016xull;\n\n" % dbits(two_eps_d))
        f.write("// --- FX64 constants (fractions of 2^64, round-to-nearest-even) ---------\n")
        f.write("inline constexpr uint64_t kWEps    = 0x%016xull; // %d\n" % (W, W))
        f.write("inline constexpr uint64_t kTwoWEps = 0x%016xull; // %d\n\n" % (2 * W, 2 * W))
        f.write("// --- separation proof (exact rationals, see results/constants.md) ------\n")
        f.write("// min_k |k/N - edge|       = %d/%d = %.6e\n"
                % (clearance.numerator, clearance.denominator, float(clearance)))
        f.write("// FP   drift bound 2T*2^-53 = %.6e  (safety ratio %.4e)\n"
                % (float(fp_drift), float(fp_ratio)))
        f.write("// FX64 drift bound   T*2^-65 = %.6e  (safety ratio %.4e)\n"
                % (float(fx_drift), float(fx_ratio)))
        f.write("\n} // namespace search\n")

    # ------------------------------------------------------- expected record
    lines = [
        ("N", "%d" % N),
        ("T", "%d" % T),
        ("eps_P", "%d" % EPS_P),
        ("eps_Q", "%d" % EPS_Q),
        ("num_candidates", "%d" % (N - 1)),
        ("eps_double_bits", "0x%016x" % dbits(eps_d)),
        ("one_minus_eps_double_bits", "0x%016x" % dbits(ome_d)),
        ("two_eps_double_bits", "0x%016x" % dbits(two_eps_d)),
        ("W_eps", "0x%016x" % W),
        ("two_W_eps", "0x%016x" % (2 * W)),
    ]
    for z in SAMPLE_Z:
        lines.append(("alpha_fixed[%d]" % z, "0x%016x" % alpha_fixed(z)))

    exp = root / "results" / "constants_expected.txt"
    with exp.open("w", newline="\n") as f:
        for k, v in lines:
            f.write("%s = %s\n" % (k, v))

    # -------------------------------------------------------------- markdown
    md = root / "results" / "constants.md"
    with md.open("w", newline="\n") as f:
        f.write("# Canonical Constants and Separation Proof\n\n")
        f.write("Generated by `tools/gen_constants.py`. `eps` is parsed as an exact\n")
        f.write("decimal fraction, scaled by `2^64` with `fractions.Fraction` and\n")
        f.write("arbitrary-precision integers, and rounded with an explicit\n")
        f.write("round-to-nearest-even. No `double` intermediate appears anywhere in\n")
        f.write("the derivation and no `double` representation of `2^64` is built.\n\n")

        f.write("## Parameters\n\n```text\n")
        f.write("N   = %-10d prime -> %d candidates, all coprime to N\n" % (N, N - 1))
        f.write("T   = %-10d steps per candidate (%.5e inner iterations total)\n"
                % (T, float(T * (N - 1))))
        f.write("eps = %d/%d      exact decimal %s\n" % (EPS_P, EPS_Q, "0.0105"))
        f.write("```\n\n")

        f.write("## Precondition 1 - `T` is not a multiple of `N`\n\n")
        f.write("`x_t = ((t*z) mod N)/N`. Because `gcd(z,N)=1`, multiplication by `z`\n")
        f.write("permutes the residues mod `N`, so over a full period `t = 1..N` every\n")
        f.write("candidate visits every residue exactly once and `hits` would take the\n")
        f.write("same value for every `z`. With `T = %d << N = %d` each candidate\n" % (T, N))
        f.write("visits a `z`-dependent subset and `hits` genuinely discriminates.\n\n")
        f.write("```text\nT %% N = %d   (non-zero, asserted)      T < N: yes\n```\n\n" % (T % N))

        f.write("## Precondition 2 - `eps*N` is not an integer\n\n")
        f.write("Reachable points are exactly `k/N`. If `eps*N` were integral a\n")
        f.write("reachable point would land precisely on a window edge and FP drift,\n")
        f.write("FX64 rounding and exact integer arithmetic could disagree about which\n")
        f.write("side it falls on. The exact closest approach of any reachable point to\n")
        f.write("either edge is `min_k |k*Q - E*N| / (N*Q)`, with `E = P` for the lower\n")
        f.write("edge and `E = Q-P` for the upper.\n\n```text\n")
        f.write("eps*N                   = %d/%d = %s   (not an integer)\n"
                % (eps_N.numerator, eps_N.denominator, float(eps_N)))
        f.write("min_k |k*Q - P*N|       = %d           (lower edge, eps)\n" % gap_lo)
        f.write("min_k |k*Q - (Q-P)*N|   = %d           (upper edge, 1-eps)\n" % gap_hi)
        f.write("edge clearance          = %d/%d = %.6e\n\n"
                % (clearance.numerator, clearance.denominator, float(clearance)))
        f.write("FP   drift bound        = 2*T*2^-53 = %.6e\n" % float(fp_drift))
        f.write("FX64 drift bound        =   T*2^-65 = %.6e\n" % float(fx_drift))
        f.write("safety ratio (FP)       = %.6e    (asserted > 1e5)\n" % float(fp_ratio))
        f.write("safety ratio (FX64)     = %.6e    (asserted > 1e5)\n" % float(fx_ratio))
        f.write("```\n\n")
        f.write("The FP bound charges `2^-53` to the rounding of `alpha = fl(z/N)` and\n")
        f.write("`2^-53` to each of the `T` additions; the conditional `- 1.0` is exact\n")
        f.write("because it cancels within a single binade. No reachable point comes\n")
        f.write("within six orders of magnitude of an edge, so FP, FX64 and the exact\n")
        f.write("integer reference are *forced* to identical `hits` for every candidate.\n\n")
        f.write("Also `k = (t*z) mod N` is never 0 for `1 <= t <= %d < N` with `N`\n" % T)
        f.write("prime and `gcd(z,N)=1`, so there is no `x = 0` edge case.\n\n")

        f.write("## Emitted constants\n\n```text\n")
        f.write("eps_double_bits           = 0x%016x  (%s)\n" % (dbits(eps_d), eps_d.hex()))
        f.write("one_minus_eps_double_bits = 0x%016x  (%s)\n" % (dbits(ome_d), ome_d.hex()))
        f.write("two_eps_double_bits       = 0x%016x  (%s)\n"
                % (dbits(two_eps_d), two_eps_d.hex()))
        f.write("W_eps                     = 0x%016x  (%d)\n" % (W, W))
        f.write("2*W_eps                   = 0x%016x  (%d)\n" % (2 * W, 2 * W))
        f.write("```\n\nConstraint `0 < 2*W_eps < 2^64`: ok.\n\n")

        f.write("## alpha_fixed samples\n\n")
        f.write("`alpha_fixed(z) = round_nearest_even(z * 2^64 / N)`, computed here in\n")
        f.write("exact rationals and in the binary with `unsigned __int128` against the\n")
        f.write("`constexpr` modulus. Every binary's `--constants` output is compared\n")
        f.write("against these values.\n\n```text\n")
        for z in SAMPLE_Z:
            f.write("alpha_fixed(%-6d) = 0x%016x\n" % (z, alpha_fixed(z)))
        f.write("```\n\n")
        f.write("Maximum value `alpha_fixed(%d) = 0x%016x`, below `2^64`, so the\n"
                % (N - 1, alpha_fixed(N - 1)))
        f.write("round-up step cannot overflow.\n\n")

        f.write("## Reference-model range\n\n```text\n")
        f.write("max k*Q   = %d * %d = %d   (far inside 2^63)\n" % (N - 1, EPS_Q, max_kQ))
        f.write("P*N       = %d\n" % (EPS_P * N))
        f.write("(Q-P)*N   = %d\n" % ((EPS_Q - EPS_P) * N))
        f.write("```\n")

    print("wrote", hpp)
    print("wrote", md)
    print("wrote", exp)
    print("edge clearance    = %.6e" % float(clearance))
    print("FP   safety ratio = %.6e" % float(fp_ratio))
    print("FX64 safety ratio = %.6e" % float(fx_ratio))
    print("W_eps             = 0x%016x" % W)


if __name__ == "__main__":
    main()
