// Exact integer ground truth.  No rounding anywhere in the trajectory or in
// the window test.
//
// With eps = P/Q the reachable point is x = k/N, k = (t*z) mod N, and
//     x < eps        <=>  k*Q < P*N
//     x >= 1 - eps   <=>  k*Q >= (Q-P)*N
// The largest intermediate is (N-1)*Q = 200004000, far inside uint64_t.
//
// This TU is linked into all three binaries.  Checking both production
// implementations against it is strictly stronger than checking them against
// each other: it proves both correct rather than merely mutually consistent.

#include <numeric> // std::gcd

#include "search_common.hpp"

namespace search {

SearchResult reference_search(uint32_t* hits_out) {
    const uint64_t lo_bound = static_cast<uint64_t>(kEpsP) * kN;          // P*N
    const uint64_t hi_bound = static_cast<uint64_t>(kEpsQ - kEpsP) * kN;  // (Q-P)*N

    uint32_t best_z = 0;
    double   best_D = 2.0;
    uint64_t checksum = 0;
    uint32_t candidates = 0;

    for (uint32_t z = 1; z < kN; ++z) {
        if (std::gcd(z, kN) != 1u) continue;
        ++candidates;

        uint32_t k = 0;
        uint32_t hits = 0;
        for (uint32_t t = 0; t < kT; ++t) {
            k += z;
            if (k >= kN) k -= kN;                       // exact mod N
            const uint64_t kQ = static_cast<uint64_t>(k) * kEpsQ;
            hits += static_cast<unsigned>((kQ < lo_bound) | (kQ >= hi_bound));
        }

        if (hits_out) hits_out[z] = hits;
        checksum += hits;

        const double D = metric_D(hits);
        if (D < best_D) { best_D = D; best_z = z; }
    }

    SearchResult r;
    r.best_z = best_z;
    r.best_D = best_D;
    r.hits_checksum = checksum;
    r.candidates = candidates;
    r.time_ms = 0.0;
    return r;
}

} // namespace search
