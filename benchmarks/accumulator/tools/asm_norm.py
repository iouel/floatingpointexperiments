#!/usr/bin/env python3
"""Normalise an objdump listing and compare two hot loops.

Normalisation drops the symbol header line and the absolute address column, and
rewrites branch targets as offsets relative to the following instruction, so
that only genuine mnemonic/operand differences survive.  This is the comparison
required by brief section 4.2 ("ignoring absolute addresses/offsets introduced
by link order or padding").
"""

import re
import sys


def load(path):
    rows = []
    for line in open(path):
        m = re.match(r"^\s*([0-9a-f]+):\t(.*)$", line.rstrip("\n"))
        if m:
            rows.append((int(m.group(1), 16), m.group(2).strip()))
    out = []
    for i, (addr, text) in enumerate(rows):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*<[^>]*>", "", text)
        nxt = rows[i + 1][0] if i + 1 < len(rows) else addr
        if re.match(r"^(j\w+|call|loop)\b", text):
            text = re.sub(r"\b[0-9a-f]{4,}\b",
                          lambda m: "REL%+d" % (int(m.group(0), 16) - nxt), text)
        out.append(text)
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print("instruction count: %d vs %d" % (len(a), len(b)))
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
