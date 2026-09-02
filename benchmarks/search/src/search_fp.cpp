// FP implementation: scalar `double` accumulator in [0,1) with an explicit
// branchless wrap.
//
//     x += alpha
//     x -= (x >= 1.0) ? 1.0 : 0.0        compare + mask + subtract
//     hits += (x < eps_d) | (x >= ome_d) bitwise |, no short-circuit branch
//
// The wrap must be branchless.  It fires with probability about alpha and is
// unpredictable for most z, so a data-dependent branch would load the baseline
// with misprediction cost and inflate the measured FX64 speedup.  The
// loop-carried chain is therefore addsd -> cmp -> and -> subsd; the detection
// reads the post-wrap state but feeds only `hits`, so it is off the critical
// path.
//
// The wrap is written with SSE/AVX scalar intrinsics rather than the plain
// ternary.  GCC 15.2 at -O3 compiles
//     x -= (x >= 1.0) ? 1.0 : 0.0;
// into vcomisd + a conditional jump over the vsubsd (gate 4 caught this), which
// is precisely the mispredicting baseline the design set out to avoid.  The
// intrinsic form below is the one already validated in the accumulator spike
// and is semantically identical: it subtracts exactly 1.0 or +0.0, and
// x - (+0.0) == x for every x >= 0.

#include <immintrin.h>

#include <numeric> // std::gcd

#include <cstdio>

#include "search_common.hpp"

namespace search {
namespace {

volatile uint32_t g_sink_z;
volatile double   g_sink_d;
volatile uint64_t g_sink_c;

// noipa additionally disables GCC's interprocedural pure/const discovery, so
// the two identical calls in run_search_timed() (warm-up, then timed) cannot
// be common-subexpression-eliminated into one.
template <bool Record>
__attribute__((noinline, noclone, noipa))
SearchResult run_search(uint32_t* hits_out) {
    const double  eps_d = from_bits(kEpsBits);
    const double  ome_d = from_bits(kOmeBits);
    const __m128d vone  = _mm_set_sd(1.0);

    uint32_t best_z = 0;
    double   best_D = 2.0; // strictly above any attainable D (which is <= 1)
    uint64_t checksum = 0;
    uint32_t candidates = 0;

    for (uint32_t z = 1; z < kN; ++z) {
        if (std::gcd(z, kN) != 1u) continue; // trivially true for prime N
        ++candidates;

        const double alpha = static_cast<double>(z) / static_cast<double>(kN);
        const __m128d valpha = _mm_set_sd(alpha);
        __m128d  vx = _mm_setzero_pd();
        uint32_t hits = 0;
        // Scalar by construction: see the note on `#pragma GCC novector` in
        // search_fx64.cpp.  This loop is not vectorisable in any case, because
        // the conditional wrap makes the recurrence non-affine; the pragma is
        // applied to both implementations so the two hot loops are held to
        // identical codegen constraints.
#pragma GCC novector
        for (uint32_t t = 0; t < kT; ++t) {
            vx = _mm_add_sd(vx, valpha);
            vx = _mm_sub_sd(vx, _mm_and_pd(_mm_cmp_sd(vx, vone, _CMP_GE_OQ), vone));
            const double x = _mm_cvtsd_f64(vx);
            hits += static_cast<unsigned>((x < eps_d) | (x >= ome_d));
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

const char* impl_name() { return "FP"; }

void impl_report_constants() {
    const double eps_d = from_bits(kEpsBits);
    const double ome_d = from_bits(kOmeBits);
    const double two_eps_d = from_bits(kTwoEpsBits);
    std::printf("impl = FP\n");
    std::printf("N = %u\n", kN);
    std::printf("T = %u\n", kT);
    std::printf("eps_P = %u\n", kEpsP);
    std::printf("eps_Q = %u\n", kEpsQ);
    std::printf("num_candidates = %u\n", kNumCandidates);
    std::printf("eps_double_bits = 0x%016llx\n", (unsigned long long)to_bits(eps_d));
    std::printf("one_minus_eps_double_bits = 0x%016llx\n",
                (unsigned long long)to_bits(ome_d));
    std::printf("two_eps_double_bits = 0x%016llx\n",
                (unsigned long long)to_bits(two_eps_d));
    std::printf("W_eps = 0x%016llx\n", (unsigned long long)kWEps);
    std::printf("two_W_eps = 0x%016llx\n", (unsigned long long)kTwoWEps);
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
