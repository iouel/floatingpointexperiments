// FX128 variant: true modulo-2^128 state held as two 64-bit lanes per logical
// state, with mandatory carry propagation from the low lane into the high lane
// on every iteration.
//
//   sum_lo = lo + D_lo                    (mod 2^64)
//   carry  = (sum_lo <u D_lo)             carry out of the low lane
//   lo'    = sum_lo
//   hi'    = hi + D_hi + carry            (mod 2^64)
//
// Detection uses the unified window, mirroring the FX64 formulation exactly:
//
//   y      = state + W_eps_128            (mod 2^128, carry-propagated)
//   detect := y <u 2*W_eps_128
//          := (y_hi <u T2_hi) || (y_hi == T2_hi && y_lo <u T2_lo)
//
// The window is [0, W_128) u [2^128 - W_128, 2^128), the same half-open pair of
// intervals FX64 detects, evaluated on the 2^128 grid.
//
// AVX2 has no unsigned 64-bit compare, so sign-bit biasing is used throughout:
//   a <u b  <=>  (a ^ 2^63) <s (b ^ 2^63)
// Constants are pre-biased once, outside the timed region.  Equality is
// unaffected by the bias, so the `==` test also runs on biased values.
//
// Because x + 2^63 == x ^ 2^63 modulo 2^64, every biased intermediate is reached
// by adding a pre-biased constant instead of by an explicit XOR: this is the
// same fold FX64 already uses.  It removes three VPXORs and two live constants
// per state, which is what keeps all eight state vectors in registers.
//
// The carry out of a + b is `sum <u b`, which lets the carry compare run against
// a pre-biased *constant* rather than against the running state.
//
// 16 logical 128-bit states = 8 YMM state vectors (4 lo + 4 hi) x 4 lanes.
// Detection stays in the vector domain: mask AND 1 -> 0/1 per 64-bit lane,
// accumulated with VPADDQ every iteration.  Exactly 16 logical detection
// results, no 8-bit or 32-bit sublane counting, no POPCNT and no horizontal
// reduction inside the timed region.
//
// NOTE on degeneracy: X0_128_lo and D_128_lo are both exactly zero, because x0
// and Delta are doubles and their exact 2^128 scalings have no bits below
// 2^-64.  The low lane therefore holds zero for every iteration and the update
// carry is always zero.  The sequence is branchless, so this does not change its
// cost, but it does mean the constants must be made runtime-opaque or the
// compiler would fold D_lo == 0 and delete the carry chain outright.  See
// `init_consts` below and summary.md.

#include <immintrin.h>

#include <cstdio>
#include <cstring>

#include "constants.hpp"
#include "spike.hpp"

namespace spike {
namespace {

struct alignas(32) Ctx {
    __m256i lo[4];  // loop-carried low lanes
    __m256i hi[4];  // loop-carried high lanes
    __m256i acc[4]; // running per-lane 0/1 accumulators
};

// Loop constants, held in one aligned block.  Passing them by pointer rather
// than by value lets the register allocator choose memory operands for them and
// keep all eight live state vectors in registers; see results/asm_check.md.
struct alignas(32) Consts {
    __m256i d_lo, d_hi, d_lo_b;
    __m256i w_lo_b, w_hi_b;
    __m256i t2_hi_b, t2_lo_b, ones;
};

__attribute__((noinline, noclone))
void run_loop(Ctx* __restrict ctx, uint64_t n, const Consts* __restrict k) {
    __m256i l0 = ctx->lo[0], l1 = ctx->lo[1], l2 = ctx->lo[2], l3 = ctx->lo[3];
    __m256i h0 = ctx->hi[0], h1 = ctx->hi[1], h2 = ctx->hi[2], h3 = ctx->hi[3];
    __m256i a0 = ctx->acc[0], a1 = ctx->acc[1], a2 = ctx->acc[2], a3 = ctx->acc[3];

#define SPIKE_FX128_STEP(LO, HI, ACC)                                              \
    do {                                                                           \
        /* update: 128-bit add with mandatory carry propagation.                */ \
        /* slo_b is the sign-biased form of slo, obtained directly as           */ \
        /* lo + (D_lo + 2^63) because adding 2^63 mod 2^64 is XOR with 2^63.    */ \
        __m256i slo = _mm256_add_epi64(LO, k->d_lo);                               \
        __m256i slo_b = _mm256_add_epi64(LO, k->d_lo_b);                           \
        __m256i carry = _mm256_cmpgt_epi64(k->d_lo_b, slo_b); /* slo <u D_lo */    \
        HI = _mm256_sub_epi64(_mm256_add_epi64(HI, k->d_hi), carry);               \
        LO = slo;                                                                  \
        /* detect: y = state + W_128, then y <u 2*W_128, all on biased values */   \
        __m256i ylo_b = _mm256_add_epi64(slo, k->w_lo_b);                          \
        __m256i c2 = _mm256_cmpgt_epi64(k->w_lo_b, ylo_b);    /* ylo <u W_lo */    \
        __m256i yhi_b = _mm256_sub_epi64(_mm256_add_epi64(HI, k->w_hi_b), c2);     \
        __m256i hlt = _mm256_cmpgt_epi64(k->t2_hi_b, yhi_b);  /* yhi <u T2_hi */   \
        __m256i heq = _mm256_cmpeq_epi64(yhi_b, k->t2_hi_b);                       \
        __m256i llt = _mm256_cmpgt_epi64(k->t2_lo_b, ylo_b);  /* ylo <u T2_lo */   \
        __m256i det = _mm256_or_si256(hlt, _mm256_and_si256(heq, llt));            \
        ACC = _mm256_add_epi64(ACC, _mm256_and_si256(det, k->ones));               \
    } while (0)

    for (uint64_t i = 0; i < n; ++i) {
        SPIKE_FX128_STEP(l0, h0, a0);
        SPIKE_FX128_STEP(l1, h1, a1);
        SPIKE_FX128_STEP(l2, h2, a2);
        SPIKE_FX128_STEP(l3, h3, a3);
    }

#undef SPIKE_FX128_STEP

    ctx->lo[0] = l0; ctx->lo[1] = l1; ctx->lo[2] = l2; ctx->lo[3] = l3;
    ctx->hi[0] = h0; ctx->hi[1] = h1; ctx->hi[2] = h2; ctx->hi[3] = h3;
    ctx->acc[0] = a0; ctx->acc[1] = a1; ctx->acc[2] = a2; ctx->acc[3] = a3;
}

inline __m256i opaque(uint64_t v) {
    __m256i x = _mm256_set1_epi64x(static_cast<long long>(v));
    __asm__ __volatile__("" : "+x"(x));
    return x;
}

// Every constant is forced through an opaque barrier. This is required, not
// cosmetic: D_128_lo is zero, and without the barrier the compiler folds the
// carry away entirely and the mandated carry propagation disappears from the
// emitted loop.
void init_consts(Consts* k, int e) {
    const uint64_t sign = 1ull << 63;
    k->d_lo    = opaque(kD_128_lo);
    k->d_hi    = opaque(kD_128_hi);
    k->d_lo_b  = opaque(kD_128_lo ^ sign);
    k->w_lo_b  = opaque(kW128Lo[e] ^ sign);
    k->w_hi_b  = opaque(kW128Hi[e] ^ sign);
    k->t2_hi_b = opaque(kTwoW128Hi[e] ^ sign);
    k->t2_lo_b = opaque(kTwoW128Lo[e] ^ sign);
    k->ones    = opaque(1);
}

void init_ctx(Ctx* ctx) {
    for (int j = 0; j < 4; ++j) {
        ctx->lo[j] = opaque(kX0_128_lo);
        ctx->hi[j] = opaque(kX0_128_hi);
        ctx->acc[j] = opaque(0);
    }
}

// Exactly 16 logical 128-bit lanes, once, after timing.
uint64_t reduce16(const Ctx* ctx) {
    uint64_t total = 0;
    for (int j = 0; j < 4; ++j) {
        uint64_t lane[4];
        std::memcpy(lane, &ctx->acc[j], sizeof lane);
        for (int i = 0; i < 4; ++i) total += lane[i];
    }
    return total;
}

volatile uint64_t g_sink;

} // namespace

const char* impl_name() { return "FX128"; }

void impl_report_constants(int e) {
    std::printf("impl=FX128 eps_label=%s\n", kEpsLabel[e]);
    std::printf("  X0_128_lo   = 0x%016llx\n", (unsigned long long)kX0_128_lo);
    std::printf("  X0_128_hi   = 0x%016llx\n", (unsigned long long)kX0_128_hi);
    std::printf("  D_128_lo    = 0x%016llx\n", (unsigned long long)kD_128_lo);
    std::printf("  D_128_hi    = 0x%016llx\n", (unsigned long long)kD_128_hi);
    std::printf("  W_128_lo    = 0x%016llx\n", (unsigned long long)kW128Lo[e]);
    std::printf("  W_128_hi    = 0x%016llx\n", (unsigned long long)kW128Hi[e]);
    std::printf("  2W_128_lo   = 0x%016llx\n", (unsigned long long)kTwoW128Lo[e]);
    std::printf("  2W_128_hi   = 0x%016llx\n", (unsigned long long)kTwoW128Hi[e]);
}

RunResult run_measured(int e) {
    Consts k;
    init_consts(&k, e);

    Ctx ctx;
    init_ctx(&ctx);
    run_loop(&ctx, kWarmupIters, &k);
    g_sink = reduce16(&ctx);

    init_ctx(&ctx);

    RunResult r;
    const uint64_t t0 = rdtscp_fenced(&r.aux_start);
    run_loop(&ctx, kIterations, &k);
    const uint64_t t1 = rdtscp_fenced(&r.aux_end);

    r.ticks = t1 - t0;
    r.detections = reduce16(&ctx); // horizontal reduction, once, after timing
    g_sink = r.detections;         // volatile sink, after timing
    return r;
}

} // namespace spike
