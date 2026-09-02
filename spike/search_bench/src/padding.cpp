// Link-layout padding translation unit, used only by the FP-B build.
//
// It is linked ahead of fp_double.o and changes the address at which the hot
// loop lands, without contributing any code to the hot loop itself.  The
// FP-A/FP-B hot-loop instruction streams must remain identical mnemonic-for-
// mnemonic and operand-for-operand (brief section 4.2 precondition).

#include <cstdint>

namespace {
volatile uint64_t g_pad_sink;
}

extern "C" __attribute__((noinline, used)) uint64_t spike_padding_never_called(uint64_t v) {
    uint64_t acc = v;
    for (int i = 0; i < 97; ++i) {
        acc = acc * 6364136223846793005ull + 1442695040888963407ull;
        acc ^= acc >> 31;
        g_pad_sink = acc;
    }
    return acc;
}
