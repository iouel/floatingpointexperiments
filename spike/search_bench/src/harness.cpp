// Harness: argument handling, mode dispatch and output.  Contains no part of
// the timed workload -- warm-up, timing and the search itself all live inside
// run_search_timed() in the implementation TU.
//
// usage:
//   search_xx <run_id>            warm-up + one timed search -> CSV row stdout
//   search_xx --verify            own per-z hits vs the exact reference
//   search_xx --dump-hits <path>  z,hits for all candidates (this impl)
//   search_xx --dump-ref  <path>  z,hits for all candidates (exact reference)
//   search_xx --constants         the constants actually used by this binary

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "search_common.hpp"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: search_xx <run_id>\n"
                 "       search_xx --verify\n"
                 "       search_xx --dump-hits <path>\n"
                 "       search_xx --dump-ref <path>\n"
                 "       search_xx --constants\n");
    return 2;
}

int dump(const char* path, const std::vector<uint32_t>& hits,
         const search::SearchResult& r) {
    std::FILE* f = std::fopen(path, "w");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    std::fprintf(f, "z,hits\n");
    for (uint32_t z = 1; z < search::kN; ++z)
        std::fprintf(f, "%u,%u\n", z, hits[z]);
    std::fclose(f);
    std::fprintf(stderr, "best_z=%u best_D=%.17g best_D_bits=0x%016llx "
                         "checksum=%" PRIu64 " candidates=%u\n",
                 r.best_z, r.best_D,
                 (unsigned long long)search::to_bits(r.best_D),
                 r.hits_checksum, r.candidates);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    using namespace search;

    if (argc < 2) return usage();

    if (std::strcmp(argv[1], "--constants") == 0) {
        impl_report_constants();
        return 0;
    }

    if (std::strcmp(argv[1], "--verify") == 0) {
        std::vector<uint32_t> mine(kN, 0u), ref(kN, 0u);
        const SearchResult a = run_search_recorded(mine.data());
        const SearchResult b = reference_search(ref.data());

        uint32_t mismatches = 0, first_z = 0;
        for (uint32_t z = 1; z < kN; ++z) {
            if (mine[z] != ref[z]) {
                if (mismatches == 0) first_z = z;
                if (mismatches < 10)
                    std::fprintf(stderr, "  hits mismatch at z=%u: %s=%u reference=%u\n",
                                 z, impl_name(), mine[z], ref[z]);
                ++mismatches;
            }
        }

        const bool ok = mismatches == 0 && a.best_z == b.best_z &&
                        to_bits(a.best_D) == to_bits(b.best_D) &&
                        a.hits_checksum == b.hits_checksum &&
                        a.candidates == b.candidates;

        std::printf("impl=%s candidates=%u/%u hits_mismatches=%u "
                    "best_z=%u/%u best_D_bits=0x%016llx/0x%016llx "
                    "checksum=%" PRIu64 "/%" PRIu64 " %s\n",
                    impl_name(), a.candidates, b.candidates, mismatches,
                    a.best_z, b.best_z,
                    (unsigned long long)to_bits(a.best_D),
                    (unsigned long long)to_bits(b.best_D),
                    a.hits_checksum, b.hits_checksum,
                    ok ? "MATCH" : "MISMATCH");
        if (!ok && mismatches) std::printf("first mismatching z = %u\n", first_z);
        std::printf("best_D=%.17g (impl)  %.17g (reference)\n", a.best_D, b.best_D);
        return ok ? 0 : 1;
    }

    if (std::strcmp(argv[1], "--dump-hits") == 0) {
        if (argc < 3) return usage();
        std::vector<uint32_t> hits(kN, 0u);
        const SearchResult r = run_search_recorded(hits.data());
        return dump(argv[2], hits, r);
    }

    if (std::strcmp(argv[1], "--dump-ref") == 0) {
        if (argc < 3) return usage();
        std::vector<uint32_t> hits(kN, 0u);
        const SearchResult r = reference_search(hits.data());
        return dump(argv[2], hits, r);
    }

    // Timed mode.
    const char* run_id = argv[1];
    const SearchResult r = run_search_timed();

    // stdout carries exactly the raw_times.csv row shape; the run provenance
    // goes to stderr and is captured in results/run_log.json.
    std::printf("%s,%s,%.6f\n", impl_name(), run_id, r.time_ms);
    std::fprintf(stderr, "best_z=%u best_D=%.17g best_D_bits=0x%016llx "
                         "checksum=%" PRIu64 " candidates=%u\n",
                 r.best_z, r.best_D,
                 (unsigned long long)to_bits(r.best_D),
                 r.hits_checksum, r.candidates);
    return 0;
}
