// Common declarations and the timestamp primitive.
#pragma once

#include <cstdint>

namespace spike {

inline constexpr uint64_t kIterations    = 10000000ull; // exactly 1e7 timed iterations
inline constexpr uint64_t kWarmupIters   =   500000ull; // >= 5e5, brief section 5.1
inline constexpr int      kLogicalStates = 16;          // 4 YMM state vectors x 4 lanes

struct RunResult {
    uint64_t ticks;      // end_timestamp - start_timestamp
    uint64_t detections; // post-timing horizontal reduction over all 16 lanes
    uint32_t aux_start;  // TSC_AUX from the opening RDTSCP
    uint32_t aux_end;    // TSC_AUX from the closing RDTSCP
};

// ---------------------------------------------------------------------------
// Timestamp primitive.
//
// CPU ordering:      LFENCE ; RDTSCP ; LFENCE   (explicit, per brief section 5)
// Compiler ordering: the "memory" clobber on a __volatile__ asm statement.
//
// The "memory" clobber forbids GCC from moving any memory access across the
// statement, and __volatile__ forbids deleting or reordering it with respect to
// other volatile/asm statements.  The timed workload is a call to a
// noinline+noclone function that reads and writes a context object in memory,
// so it is itself a memory access and cannot cross either barrier.
//
// Outputs: EAX/EDX carry the 64-bit TSC value, ECX carries TSC_AUX.
// ---------------------------------------------------------------------------
static inline uint64_t rdtscp_fenced(uint32_t* aux) {
    uint32_t lo, hi, a;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtscp\n\t"
        "lfence"
        : "=a"(lo), "=d"(hi), "=c"(a)
        :
        : "memory");
    *aux = a;
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// Implemented by exactly one of fp_double.cpp / fixed_u64.cpp.
const char* impl_name();
// Bit patterns actually used by this binary for the given eps index.
void        impl_report_constants(int eps_index);
RunResult   run_measured(int eps_index);

} // namespace spike
