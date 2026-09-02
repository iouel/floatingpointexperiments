// FX64 variant: AVX2 uint64 state as a fraction of 2^64.  The wrap is implicit in
// modulo-2^64 unsigned addition.
//
//   x[t+1] = x[t] + D64                  (mod 2^64, inherent)
//   y      = x + W_eps                   (mod 2^64)
//   detect(x) := y <u 2*W_eps            window [0,W) u [2^64-W, 2^64)
//
// AVX2 has no unsigned 64-bit compare, so sign-bit biasing is used:
//   a <u b  <=>  (a + 2^63) <s (b + 2^63)   (mod 2^64)
// and since adding 2^63 modulo 2^64 is exactly XOR with 2^63, the bias is
// folded into the addend outside the timed region:
//   Wb = W_eps + 2^63,  Tb = 2*W_eps + 2^63
//   yb = x + Wb  (== (x + W_eps) ^ 2^63)
//   detect <=> Tb >s yb   ->  VPCMPGTQ(Tb, yb)
//
// 16 logical states = 4 YMM state vectors x 4 uint64 lanes.  Detection stays in
// the vector domain: mask AND 1 -> 0/1 per 64-bit lane, accumulated with VPADDQ
// every iteration.  No 8-bit or 32-bit sublane counting, no POPCNT and no
// horizontal reduction inside the timed region.

#include <immintrin.h>

#include <cstdio>
#include <cstring>

#include "constants.hpp"
#include "spike.hpp"

namespace spike {
namespace {

struct alignas(32) Ctx {
    __m256i s[4];   // loop-carried state
    __m256i acc[4]; // running per-lane 0/1 accumulators
};

__attribute__((noinline, noclone))
void run_loop(Ctx* __restrict ctx, uint64_t n,
              const __m256i vd, const __m256i vwb, const __m256i vtb,
              const __m256i vones) {
    __m256i s0 = ctx->s[0], s1 = ctx->s[1], s2 = ctx->s[2], s3 = ctx->s[3];
    __m256i a0 = ctx->acc[0], a1 = ctx->acc[1], a2 = ctx->acc[2], a3 = ctx->acc[3];

    for (uint64_t i = 0; i < n; ++i) {
        // update: modulo-2^64 unsigned addition, wrap inherent
        s0 = _mm256_add_epi64(s0, vd);
        s1 = _mm256_add_epi64(s1, vd);
        s2 = _mm256_add_epi64(s2, vd);
        s3 = _mm256_add_epi64(s3, vd);

        // detect on the updated state, then mask -> 0/1 -> VPADDQ
        a0 = _mm256_add_epi64(a0, _mm256_and_si256(vones,
                 _mm256_cmpgt_epi64(vtb, _mm256_add_epi64(s0, vwb))));
        a1 = _mm256_add_epi64(a1, _mm256_and_si256(vones,
                 _mm256_cmpgt_epi64(vtb, _mm256_add_epi64(s1, vwb))));
        a2 = _mm256_add_epi64(a2, _mm256_and_si256(vones,
                 _mm256_cmpgt_epi64(vtb, _mm256_add_epi64(s2, vwb))));
        a3 = _mm256_add_epi64(a3, _mm256_and_si256(vones,
                 _mm256_cmpgt_epi64(vtb, _mm256_add_epi64(s3, vwb))));
    }

    ctx->s[0] = s0; ctx->s[1] = s1; ctx->s[2] = s2; ctx->s[3] = s3;
    ctx->acc[0] = a0; ctx->acc[1] = a1; ctx->acc[2] = a2; ctx->acc[3] = a3;
}

void init_ctx(Ctx* ctx, uint64_t x0) {
    for (int k = 0; k < 4; ++k) {
        __m256i v = _mm256_set1_epi64x(static_cast<long long>(x0));
        __asm__ __volatile__("" : "+x"(v));
        ctx->s[k] = v;
        __m256i z = _mm256_setzero_si256();
        __asm__ __volatile__("" : "+x"(z));
        ctx->acc[k] = z;
    }
}

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

const char* impl_name() { return "FX64"; }

void impl_report_constants(int e) {
    std::printf("impl=FX eps_label=%s\n", kEpsLabel[e]);
    std::printf("  X0    = 0x%016llx\n", (unsigned long long)kX0Fx);
    std::printf("  D64   = 0x%016llx\n", (unsigned long long)kD64Fx);
    std::printf("  W_eps = 0x%016llx\n", (unsigned long long)kWEps[e]);
    std::printf("  2W    = 0x%016llx\n", (unsigned long long)(kWEps[e] * 2ull));
}

RunResult run_measured(int e) {
    const uint64_t W    = kWEps[e];
    const uint64_t sign = 1ull << 63;
    const __m256i vd    = _mm256_set1_epi64x(static_cast<long long>(kD64Fx));
    const __m256i vwb   = _mm256_set1_epi64x(static_cast<long long>(W + sign));
    const __m256i vtb   = _mm256_set1_epi64x(static_cast<long long>(2ull * W + sign));
    const __m256i vones = _mm256_set1_epi64x(1);

    Ctx ctx;

    init_ctx(&ctx, kX0Fx);
    run_loop(&ctx, kWarmupIters, vd, vwb, vtb, vones);
    g_sink = reduce16(&ctx);

    init_ctx(&ctx, kX0Fx);

    RunResult r;
    const uint64_t t0 = rdtscp_fenced(&r.aux_start);
    run_loop(&ctx, kIterations, vd, vwb, vtb, vones);
    const uint64_t t1 = rdtscp_fenced(&r.aux_end);

    r.ticks = t1 - t0;
    r.detections = reduce16(&ctx);
    g_sink = r.detections;
    return r;
}

} // namespace spike
