"""Audit the Cortex-M3 allocator object for loops and recursion.

Usage: python tools/check_branches.py OBJDUMP_TEXT
The input is `arm-none-eabi-objdump -d build/tlsf_m3.o`.

For every function an instruction-level control-flow graph is built
(fall-through, conditional and unconditional branches, CBZ/CBNZ; calls and
branches to other functions are treated as calls that return / tail calls;
returns end a path).  A cycle in that graph is a loop.  Backward branches
alone are not loops: at -Os GCC jumps back to shared exit blocks.

The audit fails unless every function reachable from tlsf_malloc and
tlsf_free has an acyclic CFG, the call graph from them is acyclic (no
recursion), and no unexpected external symbol is called.  tlsf_init is
reported but exempt: its table-clearing loops have a compile-time bound and
run once per heap, not per allocation.
"""
import re
import sys

func_re = re.compile(r"^([0-9a-f]+) <([^>]+)>:")
ins_re = re.compile(r"^\s+([0-9a-f]+):\s+((?:[0-9a-f]{4} ?){1,2})\s+(\S+)\s*(.*)$")
tgt_re = re.compile(r"([0-9a-f]+) <([^>+]+)(?:\+0x[0-9a-f]+)?>")
cond = r"(eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?"

funcs = {}
order = []
cur = None
for line in open(sys.argv[1]):
    m = func_re.match(line)
    if m:
        cur = m.group(2)
        funcs[cur] = []
        order.append(cur)
        continue
    m = ins_re.match(line)
    if m and cur:
        funcs[cur].append((int(m.group(1), 16), m.group(3), m.group(4).split(";")[0].strip()))


def analyse(name):
    ins = funcs[name]
    addrs = [a for a, _, _ in ins]
    nxt = {addrs[i]: (addrs[i + 1] if i + 1 < len(addrs) else None) for i in range(len(addrs))}
    succ, calls, indirect = {}, set(), 0
    it_left = 0
    for a, mnem, ops in ins:
        base = mnem.split(".")[0]
        s = []
        predicated = it_left > 0
        if it_left:
            it_left -= 1
        if re.match(r"^it[te]*$", base):
            it_left = len(base) - 1
            s.append(nxt[a])
        elif re.match(r"^(b|cbz|cbnz)" + cond + "$", base) and base != "bl":
            t = tgt_re.search(ops)
            uncond = base == "b" and not predicated
            if t and t.group(2) == name:
                s.append(int(t.group(1), 16))
            elif t:
                calls.add(t.group(2))  # branch to another function: tail call
            if not uncond:
                s.append(nxt[a])
        elif base == "bl":
            t = tgt_re.search(ops)
            if t:
                calls.add(t.group(2))
            s.append(nxt[a])
        elif base in ("blx",):
            indirect += 1
            s.append(nxt[a])
        elif (base.startswith("bx") and "lr" in ops) or (base.startswith("pop") and "pc" in ops) or \
                (base.startswith("ldr") and ops.startswith("pc")):
            if predicated:
                s.append(nxt[a])
        elif base.startswith("bx"):  # bx rN: indirect tail call
            indirect += 1
            if predicated:
                s.append(nxt[a])
        else:
            s.append(nxt[a])
        succ[a] = [x for x in s if x is not None]
    # cycle detection (iterative DFS, colours)
    colour, cycles = {}, []
    for root in addrs[:1]:
        stack = [(root, iter(succ[root]))]
        colour[root] = 1
        while stack:
            node, it = stack[-1]
            for x in it:
                if colour.get(x, 0) == 1:
                    cycles.append(f"{node:x}->{x:x}")
                elif colour.get(x, 0) == 0:
                    colour[x] = 1
                    stack.append((x, iter(succ.get(x, []))))
                    break
            else:
                colour[node] = 2
                stack.pop()
    return {"n": len(ins), "cycles": cycles, "calls": calls, "indirect": indirect}


info = {f: analyse(f) for f in order}
print(f"{'function':<14} {'insns':>5} {'CFG cycles (loops)':<22} {'indirect':>8}  direct calls")
for f in order:
    i = info[f]
    print(f"{f:<14} {i['n']:>5} {','.join(i['cycles']) or 'none':<22} {i['indirect']:>8}  "
          f"{','.join(sorted(i['calls'])) or '-'}")

ok = True
for root in ("tlsf_malloc", "tlsf_free"):
    seen, stack = set(), [(root, [root])]
    while stack:
        fn, path = stack.pop()
        if fn not in info:
            print(f"FAIL: {root} reaches external symbol {fn}")
            ok = False
            continue
        seen.add(fn)
        if info[fn]["cycles"]:
            print(f"FAIL: {fn} (reachable from {root}) contains a loop")
            ok = False
        for c in info[fn]["calls"]:
            if c in path:
                print(f"FAIL: recursion {' -> '.join(path + [c])}")
                ok = False
            else:
                stack.append((c, path + [c]))
    print(f"{root}: reachable functions {sorted(seen)} (indirect calls = lock hooks via call_hook)")
print("LOOP/RECURSION AUDIT PASS" if ok else "LOOP/RECURSION AUDIT FAIL")
sys.exit(0 if ok else 1)
