#!/usr/bin/env python3
"""Normalise an objdump listing and compare two hot loops.

Normalisation drops the symbol header line and the absolute address column, and
rewrites branch targets as offsets relative to the following instruction, so
that only genuine mnemonic/operand differences survive.  This is the comparison
required by brief section 4.2 ("ignoring absolute addresses/offsets introduced
by link order or padding").

Copied verbatim from spike/benchmark, with one addition: the optional
`--norm-rip` flag.  The original tool normalises branch targets only, which was
sufficient there because those hot loops referenced no constant pool.  The FP
loop here does -- it loads `1 - eps_d` with a rip-relative operand -- and the
displacement of that load, together with the address objdump resolves it to,
shifts with link layout exactly as branch targets do.  `--norm-rip` therefore
rewrites each `0xNNN(%rip)` displacement to `RIP` and each resolved pool
address to `POOL[k]`, where `k` is that address's rank among all pool addresses
referenced by the listing.  Ranking rather than erasing means a genuine
difference -- the two builds referencing *different* pool slots, or the same
slots in a different order -- still shows up as a mismatch.

The flag is opt-in so the default behaviour, and therefore the verdict the
original tool would give, is preserved and can still be reproduced.
"""

import re
import sys


def load(path, norm_rip=False):
    rows = []
    for line in open(path):
        m = re.match(r"^\s*([0-9a-f]+):\t(.*)$", line.rstrip("\n"))
        if m:
            rows.append((int(m.group(1), 16), m.group(2).strip()))

    pool_rank = {}
    if norm_rip:
        seen = []
        for _, text in rows:
            m = re.search(r"#\s*([0-9a-f]+)\s*<", text)
            if m:
                seen.append(int(m.group(1), 16))
        for k, addr in enumerate(sorted(set(seen))):
            pool_rank[addr] = k

    out = []
    for i, (addr, text) in enumerate(rows):
        text = re.sub(r"\s+", " ", text)
        if norm_rip:
            text = re.sub(r"#\s*([0-9a-f]+)\s*<[^>]*>",
                          lambda m: "# POOL[%d]" % pool_rank[int(m.group(1), 16)], text)
        text = re.sub(r"\s*<[^>]*>", "", text)
        if norm_rip:
            text = re.sub(r"0x[0-9a-f]+\(%rip\)", "RIP", text)
        nxt = rows[i + 1][0] if i + 1 < len(rows) else addr
        if re.match(r"^(j\w+|call|loop)\b", text):
            text = re.sub(r"\b[0-9a-f]{4,}\b",
                          lambda m: "REL%+d" % (int(m.group(0), 16) - nxt), text)
        out.append(text)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    norm_rip = "--norm-rip" in sys.argv[1:]
    a, b = load(args[0], norm_rip), load(args[1], norm_rip)
    print("instruction count: %d vs %d%s"
          % (len(a), len(b), "   (rip displacements normalised)" if norm_rip else ""))
    if a == b:
        print("RESULT: MATCH instruction-for-instruction "
              "(mnemonics + operands; absolute addresses and link-layout "
              "offsets normalised away)")
        return 0
    print("RESULT: MISMATCH")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print("  [%d] %-40s | %s" % (i, x, y))
    if len(a) != len(b):
        print("  length differs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
