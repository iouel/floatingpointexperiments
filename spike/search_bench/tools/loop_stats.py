#!/usr/bin/env python3
"""Locate the hot loop in an objdump listing and report its shape.

The loop body is delimited by the backward conditional branch at the end of the
function: everything from its target up to and including the branch itself.
"""

import re
import sys


def rows(path):
    out = []
    for line in open(path):
        m = re.match(r"^\s*([0-9a-f]+):\t(.*)$", line.rstrip("\n"))
        if m:
            out.append((int(m.group(1), 16), re.sub(r"\s+", " ", m.group(2).strip())))
    return out


def body(path):
    r = rows(path)
    back = None
    for addr, text in r:
        m = re.match(r"^j\w+\s+([0-9a-f]+)", text)
        if m and int(m.group(1), 16) < addr:
            back = (int(m.group(1), 16), addr)
    if back is None:
        raise SystemExit("no backward branch found in %s" % path)
    lo, hi = back
    return lo, hi, [(a, t) for a, t in r if lo <= a <= hi]


def main():
    print("%-7s %-13s %5s %6s %6s %6s" %
          ("name", "range", "insns", "vector", "stack", "loads"))
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        lo, hi, b = body(path)
        vec = sum(1 for _, t in b if t.startswith("v"))
        stack = sum(1 for _, t in b if "(%rsp)" in t or "(%rbp)" in t)
        loads = sum(1 for _, t in b if re.search(r"\(%r[a-z0-9]+\)", t))
        print("%-7s %04x-%04x %5d %6d %6d %6d" % (name, lo, hi, len(b), vec, stack, loads))


if __name__ == "__main__":
    main()
