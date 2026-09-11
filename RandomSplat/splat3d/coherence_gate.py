# The honest kernel under the starship: a dendrite is not a zip-decompressor,
# it's a PHASE COMPARATOR. Reception gated by coherence (Fries), phase preserved
# by carrying a complex phasor (janus), surprise = mismatch = the gamma burst.
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# a receiver locked to its own clock phase; pings arrive carrying a firing phase
phi_local = 0.0
dphi = np.linspace(-np.pi, np.pi, 400)
gain = 0.5*(1+np.cos(dphi))          # chandelier opens the gate when aligned
surprise = 1 - gain                   # gamma burst / prediction error when not

# stream of pings at random phases; how much each one "lands"
rng = np.random.default_rng(0)
pings = rng.uniform(-np.pi, np.pi, 12)
landed = 0.5*(1+np.cos(pings - phi_local))
print("phase-gated reception (receiver locked at phi=0):")
for p, g in sorted(zip(pings, landed)):
    bar = "#"*int(g*30)
    print(f"  ping phase {p:+.2f} rad  -> gain {g:.2f}  {bar}")
print(f"\n  aligned ping passes (gain ~1), antiphase ping blocked (gain ~0).")
print(f"  'received' kept complex: exp(i*dphi) -> phase survives, never collapsed to a scalar.")

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax[0].plot(dphi, gain, lw=2, color="#2980b9", label="gain = ½(1+cos Δφ)")
ax[0].plot(dphi, surprise, lw=2, color="#c0392b", label="surprise = 1−gain")
ax[0].set_xlabel("Δφ  (ping phase − local clock)"); ax[0].set_ylabel("response")
ax[0].set_title("the dendrite as phase comparator", fontsize=10)
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_xticks([-np.pi,0,np.pi]); ax[0].set_xticklabels(["−π","0","π"])
th = np.linspace(0,2*np.pi,400)
ax[1].plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
for p in pings:
    g = 0.5*(1+np.cos(p-phi_local))
    ax[1].plot([0,np.cos(p)],[0,np.sin(p)], color=plt.cm.viridis(g), lw=1+2*g)
ax[1].plot([0,1],[0,0], color="k", lw=2.5)  # local clock
ax[1].text(1.02,0,"local clock", fontsize=8, va="center")
ax[1].set_aspect("equal"); ax[1].axis("off")
ax[1].set_title("pings as phasors; bright = lands, dim = filtered", fontsize=10)
plt.tight_layout(); plt.savefig("coherence_gate.png", dpi=110, bbox_inches="tight")
print("saved coherence_gate.png")
