import numpy as np, itertools
import fieldworld as F

def cyc(P, margin):
    rel={}
    for i,j in itertools.combinations(range(3),2):
        m=(P[i][1]*P[j][1])>0.10
        if m.sum()<30: return None
        def keep(t,f):
            a=f[m]-f[m].mean(); b=t[m]-t[m].mean()
            n=np.linalg.norm(a)*np.linalg.norm(b)
            return float(a@b/n) if n>1e-12 else 0.0
        both=F.sigmoid(P[i][0]+P[j][0])
        si=keep(P[i][0],both)/max(abs(keep(P[i][0],F.sigmoid(P[i][0]))),1e-9)
        sj=keep(P[j][0],both)/max(abs(keep(P[j][0],F.sigmoid(P[j][0]))),1e-9)
        if abs(si-sj)<margin: return None
        rel[(i,j)] = i if si>sj else j
    w=[0,0,0]
    for v in rel.values(): w[v]+=1
    return sorted(w)==[1,1,1]

def run(n, margin, equal_amp, seed):
    rng=np.random.default_rng(seed); c=d=0
    for _ in range(n):
        R=0.34; P=[]
        for k in range(3):
            a=2*np.pi*k/3+rng.uniform(-0.2,0.2)
            amp = 5.0 if equal_amp else rng.uniform(2.0,9.0)
            P.append(F.packet(R*np.cos(a),R*np.sin(a),
                              0.30*(1+rng.uniform(-0.45,0.45)),
                              4.0*(1+rng.uniform(-0.5,0.9)),
                              rng.uniform(0,np.pi), rng.uniform(0,2*np.pi), amp))
        r=cyc(P,margin)
        if r is None: continue
        d+=1; c+=int(r)
    return c,d

print("ROBUSTNESS: do the cycles survive a wider decision margin?")
print(f"  {'margin':>7} {'cycles/decidable':>18} {'rate':>8}")
for m in (0.02,0.05,0.10,0.20):
    c,d=run(400,m,False,0); print(f"  {m:>7.2f} {f'{c}/{d}':>18} {c/max(d,1):>8.3f}")

print("\nREMOVE THE SCALAR: amplitudes EQUAL, only phase/orientation/scale differ")
print("  (if ownership is pairwise-local rather than a scalar sort, cycles live here)")
print(f"  {'margin':>7} {'cycles/decidable':>18} {'rate':>8}")
for m in (0.02,0.05,0.10):
    tot_c=tot_d=0
    for s in (0,1,2):
        c,d=run(300,m,True,s); tot_c+=c; tot_d+=d
    print(f"  {m:>7.2f} {f'{tot_c}/{tot_d}':>18} {tot_c/max(tot_d,1):>8.3f}")
