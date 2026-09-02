// Shared declarations for the Kronecker-sequence parameter search benchmark.
//
// The search: for alpha = z/N,
//     x_0 = 0,  x_{t+1} = (x_t + alpha) mod 1
//     hits(z) = |{ t in 1..T : x_t in [0,eps) u [1-eps,1) }|
//     D(z)    = | hits(z)/T - 2*eps |
//     best_z  = argmin D(z) over z in 1..N-1 with gcd(z,N) = 1
// Ties are broken by keeping the first strict improvement, which makes best_z
// deterministic and identical across implementations.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>

#include "search_constants.hpp"

namespace search {

// Compiler-ordering barrier used to pin the timed region.
inline void barrier() { __asm__ __volatile__("" ::: "memory"); }

inline double from_bits(uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

inline uint64_t to_bits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

// ---------------------------------------------------------------------------
// The uniformity metric.
//
// One definition, used verbatim by search_fp.cpp, search_fx64.cpp and
// reference.cpp.  Because all three feed it the same integer `hits` and it is
// the same sequence of IEEE-754 operations (no -ffast-math, no reassociation,
// no FMA contraction opportunity), all three produce bit-identical doubles and
// therefore make bit-identical `<` decisions.
// ---------------------------------------------------------------------------
inline double metric_D(uint32_t hits) {
    const double occupancy = static_cast<double>(hits) / static_cast<double>(kT);
    const double d = occupancy - from_bits(kTwoEpsBits);
    return d < 0.0 ? -d : d;
}

struct SearchResult {
    uint32_t best_z;        // argmin D, first strict improvement wins
    double   best_D;        // the winning metric value
    uint64_t hits_checksum; // sum of hits over all candidates
    uint32_t candidates;    // number of z that passed the gcd guard
    double   time_ms;       // wall clock of the timed region (timed mode only)
};

// Implemented by exactly one of search_fp.cpp / search_fx64.cpp.
const char*  impl_name();
void         impl_report_constants();
SearchResult run_search_timed();                       // Record = false
SearchResult run_search_recorded(uint32_t* hits_out);  // Record = true

// Exact integer ground truth, linked into every binary (reference.cpp).
SearchResult reference_search(uint32_t* hits_out);

} // namespace search
