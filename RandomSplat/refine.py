import numpy as np, time
from scipy.ndimage import label, center_of_mass
from universe_v2 import Universe
N=40; K0=2*np.pi/8.0
def packet(centres,amp,sig):
    g=np.arange(N); X,Y,Z=np.meshgrid(g,g,g,indexing='ij'); b=np.zeros((N,N,N))
    for c in centres:
        r=np.sqrt((X-c[0])**2+(Y-c[1])**2+(Z-c[2])**2)
        b+=amp*np.exp(-r**2/(2*sig**2))*np.cos(K0*r)
    return b
def ov(p,t):
    a=p.ravel()-p.mean(); b=t.ravel()-t.mean()
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return 0.0 if na<1e-12 else float(a@b/(na*nb))
def objs(phi,thr=0.30,minv=8):
    lab,k=label(phi>thr,structure=np.ones((3,3,3)))
    if not k: return []
    v=np.bincount(lab.ravel())
    return [(np.asarray(center_of_mass(lab==i)),int(v[i])) for i in range(1,k+1) if v[i]>=minv]

print("A-WINDOW: single packet, does it stay and for how long?  (gamma=0.05, amp=1.2, sig=6)")
print("   a      bg_max    t=1000        t=3000        t=6000")
for a in [-0.01,-0.02,-0.03,-0.05,-0.08]:
    u=Universe(N=N,lam0=8.0,a=a,b=1.0,c=1.0,quintic=True,gamma=0.05,amp=0.005,seed=1)
    u.step(300); bg=float(np.abs(u.phi).max())
    t=packet([(10,10,10)],1.2,6.0); u.phi=u.phi+t; u.phi_o=u.phi_o+t
    row=[]
    for tot in (1000,2000,3000):
        u.step(tot if not row else tot)
        cs=objs(u.phi); d=min([np.linalg.norm(c-np.array([10,10,10])) for c,_ in cs],default=np.inf)
        row.append((ov(u.phi,t),d,len(cs)))
    print(f" {a:+.2f}  {bg:.1e}  " + "  ".join(f"ov{o:+.2f} d{d:>4.1f} n{n}" for o,d,n in row))
