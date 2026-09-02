// FP variant: AVX2 `double` state in [0,1) with an explicit branchless wrap.
//
//   x[t+1] = wrap(x[t] + Delta)          wrap: subtract 1.0 where x >= 1.0
//   detect(x) := (x < eps_d) || (x >= one_minus_eps_d)
//
// 16 logical states = 4 YMM state vectors x 4 double lanes.
// Detection stays in the vector domain: mask AND 1 -> 0/1 per 64-bit lane,
// accumulated with VPADDQ every iteration.  No scalarisation, no POPCNT and no
// horizontal reduction inside the timed region.

#include <immintrin.h>

#include <cstdio>
#include <cstring>

#include "constants.hpp"
#include "spike.hpp"

namespace spike {
namespace {

struct alignas(32) Ctx {
    __m256d s[4];   // loop-carried state
    __m256i acc[4]; // running per-lane 0/1 accumulators
};

inline double from_bits(uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// The complete timed workload: update -> wrap -> detect -> mask-to-0/1 ->
// vector accumulate, repeated `n` times.  noinline+noclone so that exactly one
// copy of this loop exists, is shared by warm-up and the timed run, and cannot
// be specialised on the constants.
__attribute__((noinline, noclone))
void run_loop(Ctx* __restrict ctx, uint64_t n,
              const __m256d vdelta, const __m256d vone,
              const __m256d veps, const __m256d vome, const __m256i vones) {
    __m256d s0 = ctx->s[0], s1 = ctx->s[1], s2 = ctx->s[2], s3 = ctx->s[3];
    __m256i a0 = ctx->acc[0], a1 = ctx->acc[1], a2 = ctx->acc[2], a3 = ctx->acc[3];

    for (uint64_t i = 0; i < n; ++i) {
        // update + mandatory branchless wrap
        s0 = _mm256_add_pd(s0, vdelta);
        s0 = _mm256_sub_pd(s0, _mm256_and_pd(_mm256_cmp_pd(s0, vone, _CMP_GE_OQ), vone));
        s1 = _mm256_add_pd(s1, vdelta);
        s1 = _mm256_sub_pd(s1, _mm256_and_pd(_mm256_cmp_pd(s1, vone, _CMP_GE_OQ), vone));
        s2 = _mm256_add_pd(s2, vdelta);
        s2 = _mm256_sub_pd(s2, _mm256_and_pd(_mm256_cmp_pd(s2, vone, _CMP_GE_OQ), vone));
        s3 = _mm256_add_pd(s3, vdelta);
        s3 = _mm256_sub_pd(s3, _mm256_and_pd(_mm256_cmp_pd(s3, vone, _CMP_GE_OQ), vone));

        // detect on the post-wrap state, then mask -> 0/1 -> VPADDQ
        a0 = _mm256_add_epi64(a0, _mm256_and_si256(vones, _mm256_castpd_si256(
                 _mm256_or_pd(_mm256_cmp_pd(s0, veps, _CMP_LT_OQ),
                              _mm256_cmp_pd(s0, vome, _CMP_GE_OQ)))));
        a1 = _mm256_add_epi64(a1, _mm256_and_si256(vones, _mm256_castpd_si256(
                 _mm256_or_pd(_mm256_cmp_pd(s1, veps, _CMP_LT_OQ),
                              _mm256_cmp_pd(s1, vome, _CMP_GE_OQ)))));
        a2 = _mm256_add_epi64(a2, _mm256_and_si256(vones, _mm256_castpd_si256(
                 _mm256_or_pd(_mm256_cmp_pd(s2, veps, _CMP_LT_OQ),
                              _mm256_cmp_pd(s2, vome, _CMP_GE_OQ)))));
        a3 = _mm256_add_epi64(a3, _mm256_and_si256(vones, _mm256_castpd_si256(
                 _mm256_or_pd(_mm256_cmp_pd(s3, veps, _CMP_LT_OQ),
                              _mm256_cmp_pd(s3, vome, _CMP_GE_OQ)))));
    }

    ctx->s[0] = s0; ctx->s[1] = s1; ctx->s[2] = s2; ctx->s[3] = s3;
    ctx->acc[0] = a0; ctx->acc[1] = a1; ctx->acc[2] = a2; ctx->acc[3] = a3;
}

// Initialise all 16 logical states from the same mathematical x0.  Each vector
// is forced through an opaque asm barrier so the compiler cannot prove the four
// state vectors equal and collapse the four dependency chains into one.
void init_ctx(Ctx* ctx, double x0) {
    for (int k = 0; k < 4; ++k) {
        __m256d v = _mm256_set1_pd(x0);
        __asm__ __volatile__("" : "+x"(v));
        ctx->s[k] = v;
        __m256i z = _mm256_setzero_si256();
        __asm__ __volatile__("" : "+x"(z));
        ctx->acc[k] = z;
    }
}

// Exactly 16 logical 64-bit lanes, once, after timing.
uint64_t reduce16(const Ctx* ctx) {
    uint64_t total = 0;
    for (int k = 0; k < 4; ++k) {
        uint64_t lane[4];
        std::memcpy(lane, &ctx->acc[k], sizeof lane);
        for (int j = 0; j < 4; ++j) total += lane[j];
    }
    return total;
}

volatile uint64_t g_sink;

} // namespace

const char* impl_name() { return "FP"; }

void impl_report_constants(int e) {
    double eps = from_bits(kEpsBits[e]);
    double ome = from_bits(kOneMinusEpsBits[e]);
    double x0 = from_bits(kX0Bits);
    double dl = from_bits(kDeltaBits);
    uint64_t b;
    std::printf("impl=FP eps_label=%s\n", kEpsLabel[e]);
    std::memcpy(&b, &x0, 8);  std::printf("  x0_double_bits            = 0x%016llx (%a)\n", (unsigned long long)b, x0);
    std::memcpy(&b, &dl, 8);  std::printf("  Delta_double_bits         = 0x%016llx (%a)\n", (unsigned long long)b, dl);
    std::memcpy(&b, &eps, 8); std::printf("  eps_double_bits           = 0x%016llx (%a)\n", (unsigned long long)b, eps);
    std::memcpy(&b, &ome, 8); std::printf("  one_minus_eps_double_bits = 0x%016llx (%a)\n", (unsigned long long)b, ome);
}

RunResult run_measured(int e) {
    const __m256d vdelta = _mm256_set1_pd(from_bits(kDeltaBits));
    const __m256d vone   = _mm256_set1_pd(1.0);
    const __m256d veps   = _mm256_set1_pd(from_bits(kEpsBits[e]));
    const __m256d vome   = _mm256_set1_pd(from_bits(kOneMinusEpsBits[e]));
    const __m256i vones  = _mm256_set1_epi64x(1);
    const double  x0     = from_bits(kX0Bits);

    Ctx ctx;

    // Warm-up: the identical code path, entirely before timing, excluded from
    // measured_value.  State and accumulators are then reset so the reported
    // detection count covers exactly the timed iterations.
    init_ctx(&ctx, x0);
    run_loop(&ctx, kWarmupIters, vdelta, vone, veps, vome, vones);
    g_sink = reduce16(&ctx);

    init_ctx(&ctx, x0);

    RunResult r;
    const uint64_t t0 = rdtscp_fenced(&r.aux_start);
    run_loop(&ctx, kIterations, vdelta, vone, veps, vome, vones);
    const uint64_t t1 = rdtscp_fenced(&r.aux_end);

    r.ticks = t1 - t0;
    r.detections = reduce16(&ctx); // horizontal reduction, once, after timing
    g_sink = r.detections;         // volatile sink, after timing
    return r;
}

} // namespace spike
