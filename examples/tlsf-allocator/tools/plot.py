"""Plot fragmentation snapshots and operation-count maxima.

Usage: python tools/plot.py LOGDIR OUTDIR
Reads LOGDIR/frag_*.csv and LOGDIR/stress.log (written by `make stress`).
"""
import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logdir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

names = ["small_uniform", "mixed", "pow2", "pressure"]
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i, n in enumerate(names):
    path = os.path.join(logdir, f"frag_{n}.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ops = [int(r["op"]) for r in rows]
    c = f"C{i}"
    axes[0].plot(ops, [float(r["ext_frag"]) for r in rows], c=c, lw=1,
                 label=f"{n}: 1 - L/F")
    axes[0].plot(ops, [float(r["phys_frag"]) for r in rows], c=c, lw=1,
                 ls=":", label=f"{n}: 1 - largest_free/F")
    axes[1].plot(ops, [float(r["ext_frag"]) - float(r["phys_frag"]) for r in rows],
                 c=c, lw=1, label=n)
    axes[2].plot(ops, [int(r["free_payload_F"]) / 1024 for r in rows], c=c,
                 label=f"{n} F", lw=1)
    axes[2].plot(ops, [int(r["largest_free"]) / 1024 for r in rows], c=c,
                 ls=":", lw=1, label=f"{n} largest free")
    axes[2].plot(ops, [int(r["largest_alloc_L"]) / 1024 for r in rows], c=c,
                 ls="--", lw=1, label=f"{n} L")
axes[0].set_ylabel("fragmentation")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
axes[0].set_title("TLSF fragmentation workloads (tests/test_stress.c, snapshots every 500 ops)\n"
                  "solid: 1 - L/F (good-fit request); dotted: 1 - largest_free/F (physical)")
axes[1].set_ylabel("lookup-rounding gap\n(L/F metric - physical)")
axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
axes[2].set_ylabel("KiB")
axes[2].set_xlabel("operation")
axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
p1 = os.path.join(outdir, "fragmentation.png")
fig.savefig(p1, dpi=120)
print("wrote", p1)

# Operation-count maxima vs analytical bounds.
text = open(os.path.join(logdir, "stress.log")).read()
section = text.split("Operation-count maxima over the whole stress run:")[1]
cols = ["bitmap", "insert", "remove", "neighbor", "split", "coalesce"]
meas = {}
for line in section.splitlines():
    m = re.match(r"(malloc_ok|malloc_fail|free)\s+(\d+)\s+" + r"\s+".join([r"(\d+)"] * 6), line)
    if m:
        meas[m.group(1)] = [int(x) for x in m.groups()[2:]]
bounds = {
    "malloc_ok": [3, 1, 1, 1, 1, 0],
    "malloc_fail": [2, 0, 0, 0, 0, 0],
    "free": [0, 1, 2, 2, 0, 2],
}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, op in zip(axes, ["malloc_ok", "malloc_fail", "free"]):
    x = range(len(cols))
    ax.bar([i - 0.2 for i in x], bounds[op], width=0.4, label="analytical bound")
    ax.bar([i + 0.2 for i in x], meas[op], width=0.4, label="observed max")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cols, rotation=40, fontsize=8)
    ax.set_title(op)
axes[0].set_ylabel("count per operation")
axes[0].legend(fontsize=8)
fig.suptitle("Primitive operations per call: stress run maxima vs source-derived bounds")
fig.tight_layout()
p2 = os.path.join(outdir, "opcounts.png")
fig.savefig(p2, dpi=120)
print("wrote", p2)
print("measured maxima:", meas)
