#!/usr/bin/env python3
"""Gate 4 support: dump the timed search function and isolate its inner loop.

Run inside WSL:  python3 tools/extract_asm.py <build_dir>

`loop_stats.py` (copied verbatim from spike/benchmark) delimits a hot loop by
the *last* backward branch in a function.  Here the timed function contains
three nested loops -- the std::gcd guard, the candidate loop and the t-loop --
so the last backward branch is the outer candidate loop and that heuristic
picks the wrong body.  This script instead reports every backward branch and
selects the one whose span is smallest and which contains no other backward
branch: the innermost loop, i.e. the recurrence under test.

Writes, per binary:
    results/asm_<name>.txt     full disassembly of run_search<false>
    results/inner_<name>.txt   just the inner-loop body
"""

import re
import subprocess
import sys
from pathlib import Path

BINARIES = [("fp_a", "search_fp"), ("fp_b", "search_fp_b"), ("fx64", "search_fx64")]
FUNC = "run_search<false>"

ROW = re.compile(r"^\s*([0-9a-f]+):\t(.*)$")
BACKWARD = re.compile(r"^(j\w+)\s+([0-9a-f]+)")


def disassemble(exe):
    out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", str(exe)],
                         capture_output=True, text=True, check=True).stdout
    lines, capturing = [], False
    for line in out.splitlines():
        if line.endswith(">:"):
            capturing = FUNC in line
            if capturing:
                lines.append(line)
            continue
        if capturing:
            if not line.strip():
                break
            lines.append(line)
    if not lines:
        raise SystemExit("function %s not found in %s" % (FUNC, exe))
    return lines


def rows(lines):
    out = []
    for line in lines:
        m = ROW.match(line)
        if m:
            out.append((int(m.group(1), 16), re.sub(r"\s+", " ", m.group(2).strip()), line))
    return out


def backward_branches(r):
    spans = []
    for addr, text, _ in r:
        m = BACKWARD.match(text)
        if m:
            tgt = int(m.group(2), 16)
            if tgt < addr:
                spans.append((tgt, addr))
    return spans


def innermost_all(spans):
    """Every span containing no other backward branch: the leaf loops."""
    out = []
    for lo, hi in spans:
        nested = any(lo < a and b < hi for a, b in spans if (a, b) != (lo, hi))
        if not nested:
            out.append((lo, hi))
    return sorted(set(out))


def is_counted(body):
    """A counted loop decrements a trip counter immediately before the backedge.

    The timed function has two leaf loops: the data-dependent std::gcd guard
    (which exits on a value comparison) and the t-loop over the recurrence
    (which GCC compiles as a down-counter from kT).  Only the latter is
    counted, so this distinguishes them without reference to any particular
    mnemonic of either implementation.
    """
    return len(body) >= 2 and re.match(r"^sub\s+\$0x1,", body[-2][1]) is not None


def recurrence_loop(spans, r):
    """The leaf loop that carries the recurrence: the innermost counted one."""
    leaves = innermost_all(spans)
    counted = []
    for lo, hi in leaves:
        body = [x for x in r if lo <= x[0] <= hi]
        if is_counted(body):
            counted.append((lo, hi))
    if len(counted) != 1:
        raise SystemExit("expected exactly one counted leaf loop, found %r "
                         "(leaves %r)" % (counted, leaves))
    return counted[0], leaves


def main():
    build = Path(sys.argv[1]).expanduser()
    res = Path(__file__).resolve().parent.parent / "results"
    res.mkdir(exist_ok=True)

    print("%-6s %-13s %5s %6s %6s %6s %6s" %
          ("name", "inner range", "insns", "vector", "stack", "memops", "condjmp"))
    for name, exe in BINARIES:
        lines = disassemble(build / exe)
        r = rows(lines)
        (res / ("asm_%s.txt" % name)).open("w", newline="\n").write("\n".join(lines) + "\n")

        spans = backward_branches(r)
        (lo, hi), leaves = recurrence_loop(spans, r)
        body = [(a, t, raw) for a, t, raw in r if lo <= a <= hi]
        (res / ("inner_%s.txt" % name)).open("w", newline="\n").write(
            "\n".join(raw for _, _, raw in body) + "\n")

        ymm = sum(1 for _, t, _ in body if "%ymm" in t)
        stack = sum(1 for _, t, _ in body if "(%rsp)" in t or "(%rbp)" in t)
        memops = sum(1 for _, t, _ in body if re.search(r"\(%r[a-z0-9]+\)|rip\)", t))
        # conditional jumps other than the loop backedge itself
        cond = sum(1 for a, t, _ in body
                   if re.match(r"^j(?!mp\b)\w+", t) and a != hi)
        print("%-6s %04x-%04x %5d %6d %6d %6d %6d   leaves=%s"
              % (name, lo, hi, len(body), ymm, stack, memops, cond,
                 ",".join("%04x-%04x" % s for s in leaves)))
    print()
    print("vector column counts %ymm operands; condjmp excludes the loop backedge")
    print("leaves lists every innermost loop; the one selected is the counted t-loop")


if __name__ == "__main__":
    main()
