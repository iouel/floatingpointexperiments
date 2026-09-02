// FX64 implementation: scalar uint64 accumulator holding a fraction of 2^64.
//
//     x += alpha                          implicit modulo 2^64, no wrap
//     hits += (x + kWEps) < kTwoWEps      unified two-sided window test
//
// The wrap is the integer add's own overflow, so the loop-carried chain is a
// single `add`.  The window test folds [0,W) u [2^64-W, 2^64) into one
// unsigned comparison by rotating the interval to [0, 2W).

#include <numeric> // std::gcd

#include <cstdio>

#include "search_common.hpp"

namespace search {
namespace {

using u128 = unsigned __int128;

volatile uint32_t g_sink_z;
volatile double   g_sink_d;
volatile uint64_t g_sink_c;

// Exact round-to-nearest-even of z * 2^64 / N, with no double intermediate.
// The division is against the constexpr modulus so the compiler is free to
// strength-reduce it.  q cannot overflow: q <= round((N-1)*2^64/N)
// = 2^64 - round(2^64/N), asserted for the extremes in gen_constants.py.
inline uint64_t alpha_fixed(uint32_t z) {
    const u128 num = static_cast<u128>(z) << 64;
    uint64_t q = static_cast<uint64_t>(num / kN);
    const uint64_t r = static_cast<uint64_t>(num % kN);
    const uint64_t twice = 2u * r; // r < N ~ 1e5, cannot overflow
    if (twice > kN || (twice == kN && (q & 1u))) ++q;
    return q;
}

// noipa additionally disables GCC's interprocedural pure/const discovery, so
// the two identical calls in run_search_timed() (warm-up, then timed) cannot
// be common-subexpression-eliminated into one.
template <bool Record>
__attribute__((noinline, noclone, noipa))
SearchResult run_search(uint32_t* hits_out) {
    uint32_t best_z = 0;
    double   best_D = 2.0; // strictly above any attainable D (which is <= 1)
    uint64_t checksum = 0;
    uint32_t candidates = 0;

    for (uint32_t z = 1; z < kN; ++z) {
        if (std::gcd(z, kN) != 1u) continue; // trivially true for prime N
        ++candidates;

        const uint64_t alpha = alpha_fixed(z);
        uint64_t x = 0;
        uint32_t hits = 0;
        // `#pragma GCC novector` keeps the loop scalar, which is the premise of
        // the whole comparison.  Left to itself, GCC 15.2 at -O3 recognises
        // `x += alpha` as an affine induction variable modulo 2^64, substitutes
        // the closed form x_t = t*alpha, and vectorises 8 lanes wide (gate 4
        // caught this: 125 iterations of two YMM vpaddq chains).  That is a
        // real and legitimate further win for the fixed-point representation --
        // the FP loop cannot get it, because the conditional wrap makes its
        // recurrence non-affine -- but it measures SIMD width rather than
        // accumulator dependency depth, and the brief scopes vectorisation out.
        // The pragma constrains only the compiler's model; it emits nothing.
#pragma GCC novector
        for (uint32_t t = 0; t < kT; ++t) {
            x += alpha;
            hits += static_cast<unsigned>((x + kWEps) < kTwoWEps);
        }

        if (Record) hits_out[z] = hits;
        checksum += hits;

        const double D = metric_D(hits);
        if (D < best_D) { best_D = D; best_z = z; } // first strict improvement
    }

    SearchResult r;
    r.best_z = best_z;
    r.best_D = best_D;
    r.hits_checksum = checksum;
    r.candidates = candidates;
    r.time_ms = 0.0;
    return r;
}

} // namespace

const char* impl_name() { return "FX64"; }

void impl_report_constants() {
    std::printf("impl = FX64\n");
    std::printf("N = %u\n", kN);
    std::printf("T = %u\n", kT);
    std::printf("eps_P = %u\n", kEpsP);
    std::printf("eps_Q = %u\n", kEpsQ);
    std::printf("num_candidates = %u\n", kNumCandidates);
    std::printf("eps_double_bits = 0x%016llx\n", (unsigned long long)kEpsBits);
    std::printf("one_minus_eps_double_bits = 0x%016llx\n", (unsigned long long)kOmeBits);
    std::printf("two_eps_double_bits = 0x%016llx\n", (unsigned long long)kTwoEpsBits);
    std::printf("W_eps = 0x%016llx\n", (unsigned long long)kWEps);
    std::printf("two_W_eps = 0x%016llx\n", (unsigned long long)kTwoWEps);
    const uint32_t samples[] = {1u, 2u, 3u, 17u, 1000u, 50000u, kN - 1u};
    for (uint32_t z : samples)
        std::printf("alpha_fixed[%u] = 0x%016llx\n", z,
                    (unsigned long long)alpha_fixed(z));
}

SearchResult run_search_recorded(uint32_t* hits_out) { return run_search<true>(hits_out); }

SearchResult run_search_timed() {
    // One full untimed warm-up search through the identical code path.  Its
    // result is consumed by volatile sinks so it cannot be eliminated.
    const SearchResult w = run_search<false>(nullptr);
    g_sink_z = w.best_z;
    g_sink_d = w.best_D;
    g_sink_c = w.hits_checksum;

    barrier();
    const auto t0 = std::chrono::steady_clock::now();
    barrier();
    SearchResult r = run_search<false>(nullptr);
    barrier();
    const auto t1 = std::chrono::steady_clock::now();
    barrier();

    // best_z / best_D consumed through volatile sinks *after* the stop
    // timestamp; all printing happens later still, in the harness.
    g_sink_z = r.best_z;
    g_sink_d = r.best_D;
    g_sink_c = r.hits_checksum;

    r.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

} // namespace search
