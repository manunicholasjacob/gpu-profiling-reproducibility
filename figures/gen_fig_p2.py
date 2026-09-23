#!/usr/bin/env python3
"""Paper 2 figure: three profiling protocols on a consumer laptop GPU.
(a) run-order drift correlation; (b) seed-to-seed CV. Thermal gating removes the
drift bias but not the variance floor, 62% of which is SM clock state. Measured on RTX 3050 Laptop."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import matplotlib
# Type-3 fonts are matplotlib's default and an explicit desk-reject trigger at DATE.
# 42 selects TrueType, which is what the venues want.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({"font.size":8,"font.family":"serif","figure.dpi":300})
GOOD="#2b6cb0"; BAD="#c05621"; INK="#1a1a1a"

protocols=["v1\nblocked","v2\nrandomized","v3\nthermal-gated"]
drift=[None,0.57,0.14]        # corr(run_order, rel_latency); v1 confounded (n/a)
cv=[0.304,0.68,0.32]          # median seed CV
x=np.arange(3)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,2.6))

# (a) drift correlation
vals=[0,0.57,0.14]
cols=[INK,BAD,GOOD]
ax1.bar(x,vals,color=cols,width=0.6,edgecolor=INK,lw=.5)
ax1.text(0,0.03,"confounded\n(model×time)",ha="center",fontsize=6,color=INK)
ax1.text(1,0.59,"drift",ha="center",fontsize=6.5,color=BAD)
ax1.text(2,0.16,"n.s.\n(p=0.24)",ha="center",fontsize=6.5,color=GOOD)
ax1.axhline(0,color=INK,lw=.6)
ax1.set_xticks(x); ax1.set_xticklabels(protocols,fontsize=7)
ax1.set_ylabel("corr(run order, latency)")
ax1.set_title("(a) Run-order drift",fontsize=8)
ax1.set_ylim(0,0.7); ax1.spines[["top","right"]].set_visible(False)

# (b) seed CV
ax2.bar(x,cv,color=[INK,BAD,GOOD],width=0.6,edgecolor=INK,lw=.5)
for xi,v in zip(x,cv): ax2.text(xi,v+0.015,f"{v:.2f}",ha="center",fontsize=6.5)
ax2.axhline(0.32,color=GOOD,ls="--",lw=1,label="floor ~0.32 (62% is SM clock)")
ax2.set_xticks(x); ax2.set_xticklabels(protocols,fontsize=7)
ax2.set_ylabel("median seed-to-seed CV")
ax2.set_title("(b) Run-to-run variance",fontsize=8)
ax2.set_ylim(0,0.75); ax2.legend(fontsize=6.5,loc="upper right")
ax2.spines[["top","right"]].set_visible(False)

fig.tight_layout(pad=0.6)
fig.savefig("fig_p2_protocols.pdf",bbox_inches="tight"); fig.savefig("fig_p2_protocols.png",bbox_inches="tight",dpi=150)
print("wrote fig_p2_protocols.pdf/.png")
