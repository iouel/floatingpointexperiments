// Harness: argument handling, TSC_AUX / affinity validation, CSV emission and
// an out-of-band scalar cross-check.  Contains no part of the timed workload.
//
// usage:
//   bench <eps_index> <build_variant> <run_id>   -> one raw.csv row on stdout
//   bench --constants <eps_index>                -> constants used by this binary
//   bench --verify    <eps_index>                -> vector vs scalar detection check

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "constants.hpp"
#include "spike.hpp"

namespace {

double from_bits(uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// Independent scalar model of one logical lane, used only by --verify.
// FX128 uses __uint128_t, which appears nowhere in any timed region.
uint64_t scalar_reference(int e, uint64_t n) {
    uint64_t count = 0;
    const char* impl = spike::impl_name();
    if (std::strcmp(impl, "FP") == 0) {
        const double delta = from_bits(spike::kDeltaBits);
        const double eps = from_bits(spike::kEpsBits[e]);
        const double ome = from_bits(spike::kOneMinusEpsBits[e]);
        double x = from_bits(spike::kX0Bits);
        for (uint64_t i = 0; i < n; ++i) {
            x = x + delta;
            if (x >= 1.0) x = x - 1.0;
            if (x < eps || x >= ome) ++count;
        }
    } else if (std::strcmp(impl, "FX64") == 0) {
        const uint64_t twoW = spike::kWEps[e] * 2ull;
        uint64_t x = spike::kX0Fx;
        for (uint64_t i = 0; i < n; ++i) {
            x = x + spike::kD64Fx;          // modulo-2^64
            uint64_t y = x + spike::kWEps[e];
            if (y < twoW) ++count;
        }
    } else {
        typedef unsigned __int128 u128;
        const u128 D = (static_cast<u128>(spike::kD_128_hi) << 64) | spike::kD_128_lo;
        const u128 W = (static_cast<u128>(spike::kW128Hi[e]) << 64) | spike::kW128Lo[e];
        const u128 T2 = (static_cast<u128>(spike::kTwoW128Hi[e]) << 64) | spike::kTwoW128Lo[e];
        u128 x = (static_cast<u128>(spike::kX0_128_hi) << 64) | spike::kX0_128_lo;
        for (uint64_t i = 0; i < n; ++i) {
            x = x + D;                      // modulo-2^128
            if (static_cast<u128>(x + W) < T2) ++count;
        }
    }
    return count;
}

int usage() {
    std::fprintf(stderr, "usage: bench <eps_index 0..2> <build_variant> <run_id>\n"
                         "       bench --constants <eps_index>\n"
                         "       bench --verify <eps_index>\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();

    if (std::strcmp(argv[1], "--constants") == 0) {
        int e = std::atoi(argv[2]);
        if (e < 0 || e >= spike::kNumEps) return usage();
        spike::impl_report_constants(e);
        return 0;
    }

    if (std::strcmp(argv[1], "--verify") == 0) {
        int e = std::atoi(argv[2]);
        if (e < 0 || e >= spike::kNumEps) return usage();
        spike::RunResult r = spike::run_measured(e);
        uint64_t expect = scalar_reference(e, spike::kIterations) * spike::kLogicalStates;
        std::printf("impl=%s eps=%s vector_detections=%" PRIu64
                    " scalar_reference_x16=%" PRIu64 " %s\n",
                    spike::impl_name(), spike::kEpsLabel[e], r.detections, expect,
                    r.detections == expect ? "MATCH" : "MISMATCH");
        return r.detections == expect ? 0 : 1;
    }

    if (argc < 4) return usage();
    int e = std::atoi(argv[1]);
    if (e < 0 || e >= spike::kNumEps) return usage();
    const char* variant = argv[2];
    const char* run_id = argv[3];

    const int cpu_before = sched_getcpu();
    spike::RunResult r = spike::run_measured(e);
    const int cpu_after = sched_getcpu();

    // A run is valid only when both TSC_AUX values match and correspond to the
    // pinned logical CPU.  Linux programs IA32_TSC_AUX as (node << 12) | cpu.
    const uint32_t aux_cpu_start = r.aux_start & 0xfffu;
    const uint32_t aux_cpu_end = r.aux_end & 0xfffu;
    if (r.aux_start != r.aux_end || static_cast<int>(aux_cpu_start) != cpu_before ||
        cpu_before != cpu_after) {
        std::fprintf(stderr,
                     "INVALID run: aux_start=0x%x aux_end=0x%x cpu_before=%d cpu_after=%d\n",
                     r.aux_start, r.aux_end, cpu_before, cpu_after);
        return 3;
    }

    const double denom = static_cast<double>(spike::kIterations) *
                         static_cast<double>(spike::kLogicalStates);
    // implementation,build_variant,run_id,iterations,logical_states,eps,
    // measured_value,unit,value_per_state,detections
    std::printf("%s,%s,%s,%" PRIu64 ",%d,%s,%" PRIu64 ",ticks,%.9f,%" PRIu64 "\n",
                spike::impl_name(), variant, run_id, spike::kIterations,
                spike::kLogicalStates, spike::kEpsLabel[e], r.ticks,
                static_cast<double>(r.ticks) / denom, r.detections);
    std::fprintf(stderr, "aux=0x%x cpu=%d\n", r.aux_start, aux_cpu_end);
    return 0;
}
