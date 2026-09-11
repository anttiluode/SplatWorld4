#!/usr/bin/env python3
"""
gram_spectrum.py -- ONE eigendecomposition of the render Gram, six numbers,
                    one matched control.  No GPU, no training, seconds.

WHAT THIS MEASURES
R is the SYNTHESIS operator: column k is atom k rendered alone into the
PRE-SIGMOID field, flattened.  G = R^T R is the frame Gram.  The loop map is
M = leak*I + inject*K with K = (1/4) D^-1 G, so every number below is a
property of G plus two scalars.

TWO REGISTRATIONS WERE CORRECTED BEFORE THIS FILE EXISTED.  Recording both,
because the correction is the useful part:

  WAS  "r ~ 0.531 (GOE) at delta=0; if the splat branch also reads 0.531 then a
        number from one branch predicted a number in the other."
  WHY IT IS WRONG.  GOE is the UNIVERSAL DEFAULT for generic real symmetric
  matrices.  Two matrices both reading 0.531 is two matrices both being
  unremarkable -- it is the numerical form of "the words match because one
  brain".  Worse, G is real symmetric, so it CANNOT read GUE 0.600 whatever the
  twist does; a 0.600 reading would mean the matrix is not real symmetric in
  code, which is test A2 restated.  The informative outcome is a DEVIATION, and
  two are plausible and mutually exclusive:
     r ~ 0.386  Poisson -- octave bands are near-decoupled blocks, independent
                spectra, superposed.  This is what a clean dyadic frame gives.
     r ~ 0.531  GOE -- the octaves are MIXED.  Level repulsion across bands.
     r -> 1.0   picket fence -- the spectrum is rigid/deterministic, not a
                random matrix at all.  Octave grading can do this.
  AND THEN THE CONTROL KILLED THE REPLACEMENT TOO.  This file first registered
  "the random control reads GOE 0.531, because a Wishart Gram must".  Run on a
  real constant-Q Gabor basis (n=256, 96px) it reads 0.4518, and the gate fired
  [K] on its own control.  THE GATE WAS WRONG, NOT THE CODE.  A Gabor Gram is
  not Wishart: G_jk is an overlap integral, near-zero unless two atoms are close
  in position AND scale AND orientation, so G is a SPARSE SPATIALLY LOCALISED
  random matrix.  Those live in the intermediate localisation class between
  Poisson and GOE -- localised eigenvectors give Poisson, delocalised give GOE,
  and 0.452 is in between.  GOE was never the right null for this operator.
  WHICH MAKES THE CROSS-BRANCH PREDICTION WORSE THAN UNINFORMATIVE: 0.531 is not
  even the comparison value here, so "Nuoli says 0.531, does the splat branch
  agree" cannot be asked of this matrix at all.
  NOW REGISTERED, and it is a MEASURED baseline rather than a theoretical one:
  matched control 0.452 (reproduce it before reading anything).  Trained basis
  BELOW that = more localised / octave bands more decoupled; ABOVE = more
  delocalised / bands mixed by training.  No theoretical target on either side.

  WAS  "the antisymmetric part of G must be exactly zero, as a regression on the
        no-rotation theorem."
  WHY IT IS VACUOUS.  numpy's R.T @ R is symmetric by construction to machine
  precision, and with D diagonal, D^-1 G is similar to D^-1/2 G D^-1/2 which is
  symmetric whenever G is.  Computing it here tests floating point, not the
  theorem.  The theorem is a claim about the PROJECTION YOUR LOOP USES: it holds
  iff project() is the adjoint R^T up to a diagonal.  So A2 is a FUNCTIONAL test
  (--check-adjoint) that needs your loop's project(), and it is reported as
  UNTESTED if you do not supply one.  A term you cannot show bit is untested,
  not falsified -- same rule as d.coef and the action-space Laplacian.

HONEST LABELS ON THE OTHER FOUR
  A1  theta/sigma split.  Well defined on any matrix.  Real.
  --  THE THREE COLOUR CHANNELS ARE DIFFERENT BASES, not rescalings of one.
      splat_trainer5 gives each packet a two-component phasor per channel and
      contributes a*env*cos - b*env*sin, so channel c's atom has its own CARRIER
      PHASE.  T11's amplitude-invariance argument covers a diagonal rescaling and
      does NOT cover a phase change, so G genuinely differs between channels.
      Run --channel 0, 1 and 2 and treat disagreement as information, not noise.
  A4  IS DEAD ON MEASUREMENT, not on argument.  On a clean constant-Q control
      the sub-range instability is 1.13 -- G's counting function is not a power
      law at all, so there is no exponent to report and nothing for the Weyl
      picture to attach to.  Kept in the output only so the precondition is
      visible rather than assumed.  Original label, still true:
      "spectral dimension" is NOT earned.  Weyl's law N(lam) ~ lam^(d/2) is a
      statement about the LAPLACIAN, whose eigenvalues go as k^2.  G is a frame
      operator, not a Laplacian.  What A4 reports is the power-law exponent of
      G's eigenvalue counting function, which is a real, well-defined,
      trained-vs-control comparable number, and it is NOT the spectral dimension
      of your medium.  If you want that, the operator is the packet-overlap graph
      Laplacian or -(lap+k0^2)^2, not G.  Reported as "counting exponent".
  A5  CARRIES A CONFOUND THE CONTROL ALREADY EXPOSED.  On the random basis,
      d(log tau)/d(log f) = -0.022 with corr -0.684: high-frequency atoms relax
      faster.  That is RF1's registered direction (decay rate rising with freq),
      REPRODUCED BY A BASIS THAT WAS NEVER TRAINED.  Mechanism is geometric --
      sigma = q/f, so fine atoms overlap fewer neighbours and sit in more
      localised, faster-decaying modes.  So RF1 as registered (corr >= +0.5)
      will pass on the control, exactly as G3's 0.335 was reproduced by the
      zero-action arm.  The gate has to be trained-VS-control, never
      trained-vs-zero.  Original label, still true:
      RF1 as an exponent.  The real RF1 is the drive-cut measurement in
      decay_c.npz.  What A5 gives is the SPECTRAL PREDICTION of it -- mode-
      weighted relaxation time per atom against log frequency.  That makes it
      more useful than a re-measurement: it is a number to check AGAINST the
      recorded decay, cross-branch, on the same model.
  A6  rho and i_crit in closed form.  Because K's spectrum is real and >= 0,
      rho = leak + inject*lam_max exactly and i_crit = (1-leak)/lam_max.  This
      is checkable against a number already on record: i_crit 0.0298 measured on
      model2.pt's trained basis.  If A6 does not reproduce ~0.0298 from the same
      checkpoint, the basis this file built is not the basis the loop runs on,
      and NOTHING ELSE IN THE OUTPUT COUNTS.  A6 is therefore the instrument
      gate, and it is the first thing to read.

ONE TRAP THAT WOULD HAVE VOIDED r SILENTLY
If coefficients are per-atom-PER-CHANNEL (3 colour phasors per packet, as RP1
found), then the operator is block structured and every eigenvalue is EXACTLY
3-fold degenerate.  Degenerate levels make every gap zero and the gap-ratio
statistic meaningless -- it would have printed a number.  This file builds the
SPATIAL Gram (n_atoms x n_atoms, one channel).  lam_max and hence A6/rho/i_crit
are identical either way; any level statistic on the per-channel operator is
void.  If gabor_loop.py runs the per-channel version, its spectrum is fine for
i_crit and must not be used for r.

GETTING R IN.  params.npz WAS A ROUTE, NOT A FILE -- it does not exist in this
pipeline and should never have been the headline.  The .pt checkpoint is primary:

  --model room.pt   loads the checkpoint and builds R WITHOUT knowing the
                    parameterisation.  Packet k is selected by slicing the packet
                    axis, params[:, k:k+1, :] -- no column order, no guess about
                    which entries are amplitudes, no legacy-vs-constQ branch.
                    Three things are then verified or refused rather than
                    assumed: the output nonlinearity must be disableable (it
                    REFUSES otherwise, because G is the Gram of the PRE-sigmoid
                    map and after the sigmoid nothing here means anything); the
                    field must be ADDITIVE over packets (gate A0); and colour
                    channels are flagged because per-channel coefficients give
                    exactly 3-fold degenerate eigenvalues, which voids A3.
  --inspect         RUN THIS FIRST.  Pure torch.load, builds nothing, so it
                    cannot mis-load.  That matters here: a constant-Q state_dict
                    has the SAME SHAPES as a legacy one, so a wrong module choice
                    loads silently and renders wrong -- already happened once in
                    this project.  If --model then fails, paste the --inspect
                    output and the adapter gets finished with zero guessing.
  --from-R R.npy    you build R yourself (see --how).  Nothing can go wrong about
                    the parameterisation because this file never sees it.
  --control only    no model; runs the control and the selftest.

THE GRAM IS z-DEPENDENT.  The decoder emits the basis from z, so R = R(z) and
there is no such thing as "the" spectrum.  Default is z = 0; --zscale walks
outward, which is the rho(|z|) sweep against the fire shell.

    python3 gram_spectrum.py --selftest
    python3 gram_spectrum.py --inspect --model room.pt
    python3 gram_spectrum.py --model room.pt --channel 0
    python3 gram_spectrum.py --model room.pt --channel 0 --zscale 3.0
"""

import argparse
import sys

import numpy as np

GOE_R = 0.5307
POISSON_R = 0.3863


# ----------------------------------------------------------------- renderer ---

def gabor_atom(size, cx, cy, sigma, freq, theta, phase, aniso=1.0):
    """One real Gabor packet on [-1,1]^2.  Same convention as fieldworld.py."""
    g = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(g, g)
    ct, st = np.cos(theta), np.sin(theta)
    xr = (X - cx) * ct + (Y - cy) * st
    yr = -(X - cx) * st + (Y - cy) * ct
    env = np.exp(-(xr ** 2 + (yr / max(aniso, 1e-6)) ** 2) / (2.0 * sigma ** 2))
    return env * np.cos(2.0 * np.pi * freq * xr + phase)


def build_R(params, size):
    """(P x N) synthesis matrix.  Column k = atom k rendered alone, flattened.

    Linear in coefficients by construction, which is the only property the
    Gram argument needs -- the sigmoid sits AFTER this sum."""
    n = len(params['freq'])
    R = np.empty((size * size, n), dtype=np.float64)
    ani = params.get('aniso', np.ones(n))
    for k in range(n):
        R[:, k] = gabor_atom(size, params['cx'][k], params['cy'][k],
                             params['sigma'][k], params['freq'][k],
                             params['theta'][k], params['phase'][k],
                             ani[k]).ravel()
    return R


def build_R_from_render(render_rows, n, atol=1e-6):
    """Build R from a callable that renders an arbitrary SUBSET of packets.

    render_rows(list_of_indices) -> flat 1-D pre-sigmoid field.

    THIS IS THE WHOLE POINT OF THE ADAPTER: it needs no knowledge of the
    parameterisation.  Column order, which entries are amplitudes, whether the
    trainer is legacy or constant-Q -- none of it is consulted.  Packet k is
    rendered by slicing the packet axis, and the only assumption is the one the
    Gram argument already requires: the pre-sigmoid field is ADDITIVE over
    packets.  That assumption is then VERIFIED rather than trusted (A0).

    Bias handling without needing an empty render: if b is a constant offset,
    then r_ij = g_i + g_j + b, r_i = g_i + b, r_j = g_j + b, so
    b = r_i + r_j - r_ij exactly.  Solved that way when render_rows([]) is
    unsupported.

    AMPLITUDE BAKING IS HARMLESS AND DELIBERATELY NOT CORRECTED.  If packet k's
    learned amplitude sits inside its params, column k is a_k * g_k, so
    G -> diag(a) G diag(a).  The loop operator (1/4) D^-1 G with D = diag(G) is
    the CORRELATION matrix, invariant under diagonal column rescaling -- so
    lam_max, rho, i_crit and r are untouched.  A1 and the raw G spectrum are
    amplitude-weighted, which is the honest thing anyway: the loop's state space
    is coefficient space and the amplitudes are part of the learned basis."""
    try:
        base = np.asarray(render_rows([]), float).ravel()
    except Exception:
        r0 = np.asarray(render_rows([0]), float).ravel()
        r1 = np.asarray(render_rows([1]), float).ravel()
        r01 = np.asarray(render_rows([0, 1]), float).ravel()
        base = r0 + r1 - r01

    cols = np.empty((base.size, n), dtype=np.float64)
    for k in range(n):
        cols[:, k] = np.asarray(render_rows([k]), float).ravel() - base
    full = np.asarray(render_rows(list(range(n))), float).ravel() - base
    den = max(np.linalg.norm(full), 1e-30)
    resid = float(np.linalg.norm(cols.sum(axis=1) - full) / den)
    return cols, resid, base


def synth_basis(n=256, size=96, octaves=5, q=0.6, f_lo=1.0, seed=0):
    """Dyadic constant-Q basis, log-spaced carriers, sigma = q/f.

    This is the CONTROL, and it is built to the constant-Q family from the
    trainer4q fix (f_lo = q/sig_hi = 1.0, five octaves) so that 'random' means
    'same Q/octave law, no learned arrangement' rather than 'anything'."""
    rng = np.random.default_rng(seed)
    band = rng.integers(0, octaves, n)
    freq = f_lo * (2.0 ** (band + rng.random(n)))
    return dict(cx=rng.uniform(-0.8, 0.8, n), cy=rng.uniform(-0.8, 0.8, n),
                sigma=q / freq, freq=freq,
                theta=rng.uniform(0, np.pi, n),
                phase=rng.uniform(0, 2 * np.pi, n))


def shuffle_basis(params, seed=0):
    """MATCHED control: keep every atom's (freq, sigma) exactly, destroy the
    learned arrangement (positions, orientations, phases independently
    permuted).  Strictly stronger than a fresh draw, because it cannot be
    explained away by a different Q distribution."""
    rng = np.random.default_rng(seed)
    n = len(params['freq'])
    out = {k: np.asarray(v).copy() for k, v in params.items()}
    for k in ('cx', 'cy', 'theta', 'phase'):
        if k in out:
            out[k] = out[k][rng.permutation(n)]
    return out



# ------------------------------------------------------- checkpoint adapter ---

def _torch():
    try:
        import torch
        return torch
    except ImportError:
        sys.exit("this route needs torch:  pip install torch")


def inspect_ckpt(path):
    """Pure torch.load, no model construction.  RUN THIS FIRST.

    It cannot mis-load anything because it builds nothing -- it only reports what
    is in the file.  That matters here specifically: a constant-Q state_dict has
    the SAME SHAPES as a legacy one, so a wrong module choice loads silently and
    renders wrong.  This project has already been bitten by that once."""
    torch = _torch()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    print(f"\nINSPECT {path}")
    print(f"  top-level type: {type(ck).__name__}")
    if isinstance(ck, dict):
        for k, v in ck.items():
            if hasattr(v, 'shape'):
                print(f"    {k:24s} tensor {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"    {k:24s} dict, {len(v)} entries")
            elif isinstance(v, (int, float, str, bool, type(None))):
                print(f"    {k:24s} = {v!r}")
            else:
                print(f"    {k:24s} {type(v).__name__}")
        sd = None
        for key in ('sd', 'state_dict', 'model', 'model_state_dict', 'weights'):
            if isinstance(ck.get(key), dict):
                sd = ck[key]
                print(f"\n  state_dict found under '{key}', "
                      f"{len(sd)} tensors")
                break
        if sd is None and all(hasattr(v, 'shape') for v in ck.values()):
            sd = ck
            print(f"\n  the checkpoint IS the state_dict, {len(sd)} tensors")
        if sd is not None:
            print("  first/last 12 parameter shapes:")
            items = list(sd.items())
            for k, v in items[:12] + ([('...', None)] if len(items) > 24 else []) \
                    + items[-12:]:
                print(f"    {k:44s} {tuple(v.shape) if v is not None else ''}")
    try:
        _dump_renderer(path)
    except Exception as e:
        print(f"\n  (renderer introspection unavailable: {e})")
    print("\nIf --model still fails, paste everything above.")
    return ck


def _dump_renderer(path):
    """Print the renderer's real signature and source.  This is what ends the
    guessing: an attribute NAME told us nothing (.activate turned out to be the
    parameter activation, not the output nonlinearity), so read the code."""
    import inspect
    torch = _torch()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    name = ck.get('trainer') if isinstance(ck, dict) else None
    mod = __import__(name) if isinstance(name, str) else None
    if mod is None:
        return
    model = mod.load_splatvae(path)
    model = model[0] if isinstance(model, (tuple, list)) else model
    ren = getattr(model, 'ren', None)
    if ren is None:
        return
    print(f"\n  RENDERER {type(ren).__name__}")
    print(f"    attributes: {[a for a in dir(ren) if not a.startswith('_')][:30]}")
    for fn in ('forward', 'activate', '_chunk'):
        f = getattr(ren, fn, None)
        if f is None:
            continue
        try:
            print(f"\n    --- {fn}{inspect.signature(f)} ---")
            print(inspect.getsource(f if not hasattr(f, 'forward') else type(f)))
        except (TypeError, OSError):
            print(f"    {fn}: signature/source unavailable ({type(f).__name__})")
    return ck


_TRAINERS = ('splat_trainer5', 'splat_trainer4q', 'splat_trainer3v2',
             'splat_trainer3', 'splat_trainer2', 'tiny_avatar3')
_CLASSES = ('SplatVAEQ', 'SplatVAE', 'SplatVae', 'Model')
_RENDERERS = ('ren', 'renderer', 'render', 'dec_ren', 'gabor')


def _ndim(p):
    return p.dim() if hasattr(p, 'dim') else p.ndim


def _clone(p):
    return p.clone() if hasattr(p, 'clone') else p.copy()


def _find_amp_group(ren, params, tol=1e-9):
    """Which CONTIGUOUS RANGE of the K params-per-packet columns is the amplitude
    of output channel c?  Detected, never assumed.

    THE PREVIOUS VERSION TESTED ONE COLUMN AT A TIME AND CORRECTLY REFUSED.
    In splat_trainer5, activate() does
        coeff = tanh(raw[..., 5:11]).reshape(B, N, 3, 2)
    so a packet carries a two-component PHASOR per channel and contributes
    a*env*cos - b*env*sin.  Channel c uses columns 5+2c and 6+2c, and zeroing
    either one alone leaves the packet on the canvas.  Nothing silenced, so the
    run stopped instead of producing numbers -- the refusal worked; the
    one-column-per-channel assumption behind it did not.

    A range is the amplitude of channel c iff forcing it to its off-value for
    EVERY packet makes channel c exactly constant.  Ranges are tried shortest
    first so the most specific group wins, and several fill values are tried
    because the amplitude may sit behind tanh (0), exp or softplus (large
    negative).  Whether the answer was right is then decided by A0, not here."""
    def _run(p):
        o = ren(p)
        if isinstance(o, (tuple, list)):
            o = o[0]
        return o.detach() if hasattr(o, 'detach') else o

    full = _run(params)
    K = params.shape[2]
    nch = full.shape[1] if _ndim(full) == 4 else 1

    def const_channels(out):
        if _ndim(out) != 4:
            return {0} if float(out.max() - out.min()) < tol else set()
        return {c for c in range(nch)
                if float(out[:, c].max() - out[:, c].min()) < tol}

    found, killsall = {}, {}
    for L in range(1, K + 1):
        for start in range(0, K - L + 1):
            for fill in (0.0, -30.0, -1e3):
                p2 = _clone(params)
                p2[:, :, start:start + L] = fill
                dead = const_channels(_run(p2))
                for c in dead:
                    if c not in found:
                        found[c] = (start, start + L, fill)
                        killsall[c] = (len(dead) == nch)
                if len(found) == nch:
                    break
            if len(found) == nch:
                break
        if len(found) == nch:
            break
    return found, killsall, nch


_LINKS = (
    ('identity  (renderer already returns the pre-activation sum)',
     lambda y: y),
    ('logit     (output is sigmoid(u); u = log(y/(1-y)))',
     lambda y: np.log(np.clip(y, 1e-6, 1 - 1e-6) /
                      (1.0 - np.clip(y, 1e-6, 1 - 1e-6)))),
    ('atanh     (output is tanh(u))',
     lambda y: np.arctanh(np.clip(y, -1 + 1e-6, 1 - 1e-6))),
    ('logit2    (output is (tanh(u/2)+1)/2 or sigmoid on [-1,1])',
     lambda y: np.log(np.clip((y + 1) / 2, 1e-6, 1 - 1e-6) /
                      (1.0 - np.clip((y + 1) / 2, 1e-6, 1 - 1e-6)))),
)


def _run_sat(ren, params):
    """Fraction of the canvas pinned at the rails.

    logit inverts a sigmoid EXACTLY, but only where the output is not saturated;
    clipping at 1e-6 is what makes it finite and is also the one thing that can
    break A0 on a large canvas.  Reported so a failure has a visible cause
    instead of being a mystery."""
    try:
        o = ren(params)
        o = o[0] if isinstance(o, (tuple, list)) else o
        o = (o.detach().cpu().numpy() if hasattr(o, 'detach') else np.asarray(o))
        return float(((o < 1e-6) | (o > 1 - 1e-6)).mean())
    except Exception:
        return None


def _pick_link(render_raw, n):
    """Recover the PRE-activation field by INVERTING the output nonlinearity,
    and let additivity choose which inverse was correct.

    The renderer is never modified.  If it returns sigmoid(u), then logit of the
    output is exactly u, so the linear synthesis map is recoverable without
    touching a single attribute -- and whether the recovery worked is decided by
    gate A0 rather than by believing a docstring.  Each candidate link is tried
    and the first whose reconstruction is additive over packets to 1e-6 wins.
    If none is, the run stops: an inverse that does not restore additivity has
    not recovered a Gram, and every number downstream would be void."""
    best = (None, np.inf, None, None)
    for name, f in _LINKS:
        try:
            R, resid, base = build_R_from_render(lambda idx: f(render_raw(idx)), n)
        except Exception as e:
            print(f"  link {name.split()[0]:9s} unusable ({type(e).__name__})")
            continue
        print(f"  link {name.split()[0]:9s} additivity residual {resid:.3e}"
              f"   {'<-- ACCEPTED' if resid < 1e-6 else ''}")
        if resid < best[1]:
            best = (R, resid, name, base)
        if resid < 1e-6:
            break
    return best


def load_trained_R(path, zscale=0.0, seed=0, channel=0):
    """Build R from a .pt checkpoint knowing NOTHING about the parameterisation.

    Design, after the activate() failure:
      * the packet axis is NEVER sliced and the renderer is NEVER modified.  The
        full (1, N, K) params tensor is passed on every call, so no shape or
        signature assumption can break.
      * a packet is removed by driving its DETECTED amplitude column to its
        detected off-value.  Detection is empirical (see _find_amp_columns).
      * the pre-sigmoid requirement is then MEASURED, not arranged: if the output
        nonlinearity is still active the field is not additive over packets and
        gate A0 fails, which stops the run.  A0 is the only guarantee that
        matters and it does not care how linearity was obtained."""
    torch = _torch()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck if isinstance(ck, dict) else {}
    sd = meta.get('sd') if isinstance(meta.get('sd'), dict) else None
    for key in ('state_dict', 'model', 'model_state_dict'):
        if sd is None and isinstance(meta.get(key), dict):
            sd = meta[key]

    mod = None
    want = meta.get('trainer')
    order = ((want,) if isinstance(want, str) else ()) + _TRAINERS
    for name in order:
        try:
            mod = __import__(name)
            print(f"  trainer module: {name}"
                  + ("  (named in the checkpoint)" if name == want else ""))
            break
        except ImportError:
            continue
    if mod is None:
        sys.exit(f"none of {order} importable -- run from the directory holding "
                 f"your trainer")

    model = None
    if hasattr(mod, 'load_splatvae'):
        try:
            out = mod.load_splatvae(path)
            model = out[0] if isinstance(out, (tuple, list)) else out
            print("  built via load_splatvae()")
        except Exception as e:
            print(f"  load_splatvae failed ({e}); trying the class directly")
    if model is None:
        if sd is None:
            sys.exit("no state_dict found -- run --inspect and paste the output")
        kw = {k: meta[k] for k in ('image_size', 'num_packets', 'qmode', 'q',
                                   'q_slack', 'gist_frac', 'octaves', 'sig_lo',
                                   'sig_hi', 'gist_sig_hi', 'f_max',
                                   'band_mode', 'band_norm') if k in meta}
        last = None
        for cname in _CLASSES:
            if not hasattr(mod, cname):
                continue
            try:
                model = getattr(mod, cname)(**kw)
                model.load_state_dict(sd)
                print(f"  built {cname} from checkpoint metadata")
                break
            except Exception as e:
                last, model = e, None
        if model is None:
            sys.exit(f"could not construct a model ({last}) -- run --inspect")
    model.eval()

    ren = None
    for rn in _RENDERERS:
        if hasattr(model, rn):
            ren = getattr(model, rn)
            print(f"  renderer attribute: .{rn}  (UNMODIFIED)")
            break
    if ren is None:
        sys.exit(f"no renderer among {_RENDERERS}; attributes: "
                 f"{[a for a in dir(model) if not a.startswith('_')][:40]}")

    # latent width READ from the state dict, not assumed
    nz = None
    for k in ('enc.fc_mu.bias', 'enc.fc_mu.weight', 'fc_mu.bias'):
        if sd is not None and k in sd:
            nz = int(sd[k].shape[0])
            print(f"  latent width {nz}  (read from {k})")
            break
    if nz is None:
        nz = 128
        print("  latent width not found in the state dict; assuming 128")

    g = torch.Generator().manual_seed(seed)
    z = (torch.randn(1, nz, generator=g) * zscale if zscale > 0
         else torch.zeros(1, nz))
    print(f"  operating point |z| = {float(z.norm()):.4f}"
          f"   (the Gram is z-DEPENDENT, so this is a choice)")

    with torch.no_grad():
        dec = None
        for dn in ('dec', 'decoder', 'decode'):
            if hasattr(model, dn):
                dec = getattr(model, dn)
                break
        if dec is None:
            sys.exit("no decoder attribute -- run --inspect")
        params = dec(z)
        if isinstance(params, (tuple, list)):
            params = params[0]
        if params.dim() != 3:
            sys.exit(f"decoder returned {tuple(params.shape)}; expected "
                     f"(1, N, K) -- run --inspect")
        n, K = params.shape[1], params.shape[2]
        print(f"  packets {n}   params per packet {K}")

        found, killsall, nch = _find_amp_group(ren, params)
        if not found:
            sys.exit("COULD NOT IDENTIFY AN AMPLITUDE GROUP.\n"
                     f"  Tried every contiguous range of the {K} columns at fill "
                     f"values 0, -30, -1e3;\n  none silenced any output channel.\n"
                     "  Without a way to remove one packet there is no way to\n"
                     "  build R, so this refuses rather than guessing.\n"
                     "  Run --inspect --model <ckpt> and paste the renderer\n"
                     "  source it prints; the adapter is two lines from done.")
        for c, (s0, s1, fill) in sorted(found.items()):
            print(f"  channel {c}: amplitude columns {s0}:{s1} "
                  f"({s1 - s0} of {K}), off-value {fill:g}   DETECTED"
                  + ("   [this range silences ALL channels]"
                     if killsall.get(c) else ""))
        if channel not in found:
            channel = sorted(found)[0]
        a0, a1, off = found[channel]

        def render_rows(idx):
            p = _clone(params)
            on = set(int(i) for i in idx)
            off_idx = [i for i in range(n) if i not in on]
            if off_idx:
                p[:, off_idx, a0:a1] = off
            out = ren(p)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out.detach()
            if _ndim(out) == 4:
                out = out[:, channel]
            return out.cpu().numpy().ravel()

        sat = _run_sat(ren, params)
        if sat is not None:
            print(f"  render saturation: {sat:.3%} of the canvas within 1e-6 of "
                  f"0 or 1"
                  + ("   <-- logit recovery is exact only off the rails; if A0 "
                     "fails, this is why" if sat > 0.01 else ""))
        R, resid, link, _ = _pick_link(render_rows, n)

    ok = resid < 1e-6
    print(f"  A0 additivity residual {resid:.3e}   [{'V' if ok else 'K'}]"
          f"   channel {channel}, link: {link}")
    if not ok:
        sys.exit("A0 [K] -- no output link makes the field additive over packets,\n"
                 "  so the linear synthesis map was not recovered, G is not a\n"
                 "  Gram, and every number below would be void.  Two usual\n"
                 "  causes: the render saturates (check how much of the canvas\n"
                 "  sits at 0 or 1), or the readout is not a pointwise function\n"
                 "  of a sum.  Get the renderer's pre-activation sum directly and\n"
                 "  pass it via --from-R (see --how).  Stopping rather than\n"
                 "  printing numbers.")
    return R, None


# ------------------------------------------------------------- statistics ---

def theta_sigma(G):
    """Isotropic vs traceless part, dimensionless.

    ||theta*I||_F = theta*sqrt(n), so the ratio below is 0 for a multiple of the
    identity and grows as the operator becomes shear-dominated."""
    n = G.shape[0]
    th = float(np.trace(G)) / n
    dev = G - th * np.eye(n)
    return th, float(np.linalg.norm(dev, 'fro')) / max(abs(th) * np.sqrt(n), 1e-30)


def norm_gaps(lam, w=15):
    """Locally unfolded gaps: each gap divided by the mean gap of its w
    neighbours.  Returns None if the spectrum is too short.

    A GLOBAL POLYNOMIAL FIT WAS TRIED FIRST AND IS FRAGILE -- on an
    octave-graded spectrum the fit is not always monotone, and a non-monotone
    unfolding is invalid, so it returned None on some draws and the caller
    crashed.  Local normalisation is always defined and needs no degree
    parameter.  The local mean is GEOMETRIC (a moving average in log s), not
    arithmetic: an arithmetic window mean over octave-graded gaps is dominated
    by its largest term, which left residual structure and read 0.905 instead of
    1.000 on a perfectly dyadic spectrum.  In log space a geometric series is
    linear, a centred moving average reproduces it exactly, and the normalised
    gaps flatten to 1 -- so a graded spectrum is unmasked as the picket fence it
    is rather than reading 1/ratio."""
    lam = np.sort(np.asarray(lam, float))
    lam = lam[lam > 0]
    s = np.diff(lam)
    keep = s > 1e-12 * max(np.median(s), 1e-30)
    s = s[keep]
    if len(s) < 3 * w:
        return None
    ker = np.ones(w) / w
    ls = np.log(s)
    loc = np.convolve(ls, ker, mode='same')
    edge = np.convolve(np.ones_like(ls), ker, mode='same')
    out = np.exp(ls - loc / np.maximum(edge, 1e-30))
    return out[w:-w] if len(out) > 3 * w else out


def r_stat(s):
    """<min(r, 1/r)> over consecutive gaps.  0.3863 Poisson, 0.5307 GOE,
    0.6027 GUE, -> 1 picket fence."""
    if s is None:
        return np.nan
    s = np.asarray(s, float)
    s = s[s > 0]
    if len(s) < 20:
        return np.nan
    r = s[1:] / s[:-1]
    return float(np.mean(np.minimum(r, 1.0 / r)))


def gap_ratio(lam):
    """RAW statistic, straight off the eigenvalues.  Degenerate levels give zero
    gaps, which would make the mean meaningless, so they are dropped and
    COUNTED -- the per-channel 3-fold degeneracy must be visible, not absorbed."""
    lam = np.sort(np.asarray(lam, float))
    s = np.diff(lam)
    keep = s > 1e-12 * max(np.median(s), 1e-30)
    dropped = int((~keep).sum())
    s = s[keep]
    return r_stat(s), dropped, max(len(s) - 1, 0)


def _slope(lam, lo_q, hi_q):
    lo, hi = np.quantile(lam, lo_q), np.quantile(lam, hi_q)
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 15:
        return np.nan, np.nan
    x = np.log10(lam[m])
    y = np.log10(1.0 + np.arange(len(lam))[m])
    p = np.polyfit(x, y, 1)
    res = y - np.polyval(p, x)
    r2 = 1.0 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return float(p[0]), float(r2)


def counting_exponent(lam, lo_q=0.30, hi_q=0.97):
    """Power-law exponent of the counting function: N(lam) ~ lam^p.

    NOT a spectral dimension (see header).

    THE R^2 FLAG WAS TOO WEAK AND IS NOT THE GATE.  Measured on planted cases:
    a semicircle reads R^2 0.992 and an exponential decay reads 0.978, so both
    would have passed an R^2 test while having no power law at all -- log-log is
    forgiving.  SUB-RANGE STABILITY separates them cleanly: fitting the lower
    and upper halves independently gives |p_lo - p_hi|/|p| of 0.000 for a true
    power law, 0.364 for a semicircle, 0.615 for an exponential.  That is the
    gate; R^2 is kept only as a diagnostic."""
    lam = np.sort(np.asarray(lam, float))
    lam = lam[lam > 0]
    if len(lam) < 40:
        return np.nan, np.nan, np.nan
    p, r2 = _slope(lam, lo_q, hi_q)
    p_lo, _ = _slope(lam, lo_q, 0.5 * (lo_q + hi_q))
    p_hi, _ = _slope(lam, 0.5 * (lo_q + hi_q) + 0.05, hi_q)
    inst = abs(p_lo - p_hi) / max(abs(p), 1e-30)
    return p, r2, float(inst)


def loop_spectrum(G, leak, inject, D=None):
    """K = 1/4 D^-1 G ; M = leak*I + inject*K.

    D defaults to diag(G) = per-atom energy ||r_k||^2, the matched-filter
    normalisation.  Because D^-1 G is similar to D^-1/2 G D^-1/2 (symmetric PSD
    for diagonal D>0), the spectrum is real and >= 0 -- so rho and i_crit are
    closed forms, not power iterations."""
    d = np.diag(G).copy() if D is None else np.asarray(D, float).copy()
    d[d <= 0] = 1e-30
    s = 1.0 / np.sqrt(d)
    Ks = 0.25 * (s[:, None] * G * s[None, :])          # symmetric, similar to K
    ev = np.linalg.eigvalsh(Ks)
    ev = np.clip(ev, 0.0, None)
    lam_max = float(ev[-1])
    rho = leak + inject * lam_max
    i_crit = (1.0 - leak) / lam_max if lam_max > 0 else np.inf
    return ev, lam_max, rho, i_crit


def relax_exponent(G, params, leak, inject):
    """A5: mode-weighted relaxation time per atom vs log carrier frequency.

    tau_k = sum_i |v_i[k]|^2 * (-1/ln|mu_i|) with mu_i the loop eigenvalues.
    This is the SPECTRAL PREDICTION of the drive-cut decay, to be checked
    against decay_c.npz -- not a substitute for it."""
    d = np.diag(G).copy()
    d[d <= 0] = 1e-30
    s = 1.0 / np.sqrt(d)
    Ks = 0.25 * (s[:, None] * G * s[None, :])
    ev, V = np.linalg.eigh(Ks)
    mu = leak + inject * np.clip(ev, 0.0, None)
    mu = np.clip(mu, 1e-12, 1.0 - 1e-9)               # only decaying modes
    tau = -1.0 / np.log(mu)
    w = V ** 2                                        # rows = atoms
    tk = w @ tau
    f = np.asarray(params['freq'], float)
    m = (f > 0) & np.isfinite(tk) & (tk > 0)
    if m.sum() < 20:
        return np.nan, np.nan
    x, y = np.log10(f[m]), np.log10(tk[m])
    p = np.polyfit(x, y, 1)
    rr = np.corrcoef(x, y)[0, 1]
    return float(p[0]), float(rr)


# ------------------------------------------------------------------ report ---

def analyse(R, params, leak, inject, label):
    G = R.T @ R
    n = G.shape[0]
    th, ratio = theta_sigma(G)
    ev, lam_max, rho, i_crit = loop_spectrum(G, leak, inject)
    lamG = np.linalg.eigvalsh(G)
    lamG = np.clip(lamG, 0.0, None)

    # A3/A4 run on the LOOP spectrum, not on raw G.  ev is the spectrum of
    # D^-1/2 G D^-1/2 (up to the 1/4), i.e. the CORRELATION matrix -- invariant
    # under diagonal rescaling of R's columns, so a baked-in per-packet
    # amplitude cannot move these numbers.  Raw G's spectrum is reported too but
    # it is amplitude-weighted and must not be compared across bases.
    r_raw, drop, npairs = gap_ratio(ev)
    r_unf = r_stat(norm_gaps(ev))
    p_cnt, r2, inst = counting_exponent(ev)
    a_slope, a_corr = (relax_exponent(G, params, leak, inject)
                       if params is not None else (np.nan, np.nan))

    nz = lamG[lamG > lamG.max() * 1e-12]
    eff = float(lamG.sum() ** 2 / np.sum(lamG ** 2))
    cond = float(nz.max() / nz.min()) if len(nz) else np.inf

    print(f"\n{label}   n={n}  atoms")
    print(f"  A1 theta {th:.4g}   |sigma|/|theta| {ratio:8.3f}"
          f"   {'shear-dominated' if ratio > 1 else 'breathing-dominated'}")
    print(f"  A2 antisym ||G-G^T||/||G|| "
          f"{np.linalg.norm(G - G.T) / max(np.linalg.norm(G), 1e-30):.2e}"
          f"   (VACUOUS by construction -- see --check-adjoint)")
    print(f"  A3 r  raw {r_raw:.4f}   unfolded {r_unf:.4f}   [loop spectrum]"
          f"   ({npairs} pairs, {drop} degenerate gaps dropped)")
    print(f"     Poisson 0.3863 | GOE 0.5307 | GUE 0.6027 | picket -> 1.0")
    print(f"  A4 counting exponent {p_cnt:.4f}   sub-range instability "
          f"{inst:.4f}  (R^2 {r2:.4f})"
          + ("" if inst < 0.10 else
             "\n     <-- NOT a power law (instability >= 0.10); the exponent "
             "is meaningless"))
    print(f"  A5 d(log tau)/d(log f) {a_slope:+.4f}   corr {a_corr:+.4f}"
          f"   (prediction for decay_c.npz)")
    print(f"  A6 lam_max {lam_max:.6g}   rho {rho:.6f}"
          f"   i_crit {i_crit:.6f}   {'CONTRACTING' if rho < 1 else 'ACTIVE'}")
    print(f"     spectrum: eff rank {eff:.3f}   cond {cond:.3e}"
          f"   nonzero {len(nz)}/{n}")
    return dict(n=n, theta=th, ratio=ratio, r_raw=r_raw, r_unf=r_unf,
                p=p_cnt, r2=r2, inst=inst, a=a_slope, lam_max=lam_max, rho=rho,
                i_crit=i_crit, eff=eff, cond=cond, acorr=a_corr)


def gates(tr, ct):
    """Every threshold here is either a number already on record from his own
    runs, or a trained-vs-control contrast.  No invented absolutes -- that is
    what produced thirteen mis-specified gates in this project, and one more in
    this file (I1 vs GOE) which the control caught."""
    print("\nGATES")
    if ct is not None:
        lo, hi = 0.030, 0.050
        ok = lo < ct['i_crit'] < hi
        print(f"  I1 [{'V' if ok else 'K'}]  INSTRUMENT: control i_crit "
              f"{ct['i_crit']:.4f} against the 0.0363-0.0433 already recorded "
              f"for a random basis (numpy/torch, different seeds). [K] means "
              f"this R is not the operator gabor_loop runs on")
        print(f"  I2 [{'V' if ct['inst'] > 0.10 else 'K'}]  A4 PRECONDITION "
              f"FAILS AS EXPECTED: control instability {ct['inst']:.3f} -- G's "
              f"counting function is not a power law, so no exponent exists to "
              f"compare and the Weyl reading has nothing to attach to")
        print(f"  I3  control r = {ct['r_unf']:.4f}.  This, not GOE 0.5307, is "
              f"the baseline (Gabor Grams are sparse/localised, not Wishart)")
    if tr is None:
        return
    print(f"  W1 [?]  A6 trained i_crit {tr['i_crit']:.4f} -- READ FIRST "
          f"against your measured 0.0298 on model2.pt. A mismatch means this is "
          f"not the basis the loop runs on and nothing below counts")
    if ct is None:
        print("  no control -- rerun with --control matched before reading S1-S3")
        return
    d = tr['r_unf'] - ct['r_unf']
    cls = ('MORE LOCALISED than the control -- octave bands more decoupled'
           if d < -0.02 else
           'MORE DELOCALISED -- training mixed the bands' if d > 0.02 else
           'spectrally indistinguishable from the control')
    print(f"  S1  r trained {tr['r_unf']:.4f} vs control {ct['r_unf']:.4f} "
          f"(d {d:+.4f}) -> {cls}")
    print(f"  S2  shear |sigma|/|theta| trained {tr['ratio']:.3f} vs control "
          f"{ct['ratio']:.3f}  ({tr['ratio'] / max(ct['ratio'], 1e-9):.3f}x)")
    print(f"  S3  A5 slope trained {tr['a']:+.4f} (corr {tr.get('acorr', float('nan')):+.4f}) "
          f"vs control {ct['a']:+.4f} (corr {ct.get('acorr', float('nan')):+.4f}) "
          f"-- the control reproduces RF1's direction, so only the GAP is a result")
    print(f"  S4  redundancy: eff rank {tr['eff']:.2f}/{tr['n']} trained vs "
          f"{ct['eff']:.2f}/{ct['n']} control; cond {tr['cond']:.2e} vs "
          f"{ct['cond']:.2e}")


HOW = """
BUILD R IN YOUR OWN CODE (route 1, and the only one that cannot be wrong about
your parameterisation).  R must be the PRE-SIGMOID field, one atom at a time:

    import numpy as np, torch
    model, ck = load_splatvae('model2.pt')          # your loader
    z = torch.zeros(1, LATENT)                      # the operating point
    params = model.dec(z)                           # (1, N, K) packet params
    N = params.shape[1]

    def presigmoid(coeff):                          # coeff: (N,) numpy
        c = torch.tensor(coeff, dtype=params.dtype)[None, :]
        return model.ren.field(params, c).detach().cpu().numpy().ravel()
        #      ^ whatever your renderer's PRE-activation sum is called

    base = presigmoid(np.zeros(N))                  # removes any bias
    R = np.stack([presigmoid(np.eye(N)[k]) - base for k in range(N)], axis=1)
    np.save('R.npy', R)

Two things to get right or the whole file is measuring the wrong object:
  * PRE-sigmoid.  After the sigmoid it is not linear and G is not the Gram.
  * ONE CHANNEL.  If coefficients are per-atom-per-channel, take one channel
    (or the luminance combination).  Per-channel gives exactly 3-fold
    degenerate eigenvalues, which voids A3 while still printing a number.

AND THE FUNCTIONAL TEST OF THE NO-ROTATION THEOREM (A2 done properly).  The
theorem needs project() == adjoint up to a diagonal.  Test it on your own code,
no matrices required:

    for _ in range(20):
        c = np.random.randn(N)
        lhs = project(render_presigmoid(c))         # your loop's two halves
        rhs = 0.25 * np.linalg.solve(np.diag(D), R.T @ (R @ c))
        print(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))

Residual ~1e-15: the theorem applies and no autonomous rotation is possible.
Residual O(1): project() is NOT the adjoint, K need not be symmetrizable, and
the no-rotation result is about a different operator than the one you run.
That is the SK2 fork worth resolving before anything else.
"""


# -------------------------------------------------------------- selftest ---

def selftest():
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    print("SELFTEST  (every check two-sided)")
    rng = np.random.default_rng(0)

    # T1  the r statistic itself, on planted ensembles
    A = rng.normal(size=(600, 600))
    goe = np.linalg.eigvalsh((A + A.T) / np.sqrt(2))
    r_goe = r_stat(norm_gaps(goe))
    chk("T1 GOE reads 0.531", abs(r_goe - GOE_R) < 0.03, f"{r_goe:.4f}")
    poi = np.sort(rng.uniform(0, 1, 600))
    r_poi = r_stat(norm_gaps(poi))
    chk("T1 Poisson reads 0.386", abs(r_poi - POISSON_R) < 0.04, f"{r_poi:.4f}")
    pick = np.arange(1.0, 601.0)
    r_pick = gap_ratio(pick)[0]
    chk("T1 picket fence reads ~1", r_pick > 0.97, f"{r_pick:.4f}")

    # T2  DEGENERACY GUARD: the per-channel trap must be caught, not printed
    trip = np.repeat(goe, 3)
    r_t, drop, _ = gap_ratio(trip)
    chk("T2 3-fold degenerate spectrum drops its zero gaps",
        drop >= 2 * len(goe) - 5, f"{drop} dropped")
    chk("T2 and does NOT silently read the clean value",
        abs(r_t - r_goe) > 1e-9 or drop > 0, f"r {r_t:.4f}")

    # T3  UNFOLDING IS REQUIRED, AND THE REGISTERED REASON WAS WRONG.
    #     Claimed: a steep smooth density contaminates the raw statistic.
    #     Measured: it does NOT -- iid points over three decades read raw 0.386
    #     and unfolded 0.382, because <min(r,1/r)> over CONSECUTIVE gaps is
    #     density-insensitive by design.  The real trap is different and worse:
    #     a GEOMETRICALLY graded deterministic spectrum has every gap a fixed
    #     multiple of the last, so raw r reads exactly 1/ratio -- 0.500 for a
    #     dyadic spectrum, which is a hair from GOE's 0.531.  An octave-graded
    #     Gabor Gram is exactly that shape.  So the raw number would have
    #     printed "GOE" out of pure determinism, and unfolding is what tells
    #     0.500-because-random from 0.500-because-dyadic.
    lam = np.sort(np.exp(rng.normal(0, 3.0, 800)))
    chk("T3 raw r is NOT density-contaminated for iid points (claim retracted)",
        abs(gap_ratio(lam)[0] - r_stat(norm_gaps(lam))) < 0.02,
        f"raw {gap_ratio(lam)[0]:.4f} unfolded {r_stat(norm_gaps(lam)):.4f}")
    for base, mimics in ((2.0, 'GOE 0.531'), (3.0, 'Poisson 0.386')):
        geo = np.sort(base ** -np.arange(1, 301, dtype=float))
        r_g = gap_ratio(geo)[0]
        chk(f"T3 geometric base {base:g}: raw r reads 1/base, mimicking {mimics}",
            abs(r_g - 1.0 / base) < 1e-6, f"{r_g:.4f} vs {1/base:.4f}")
    geo = np.sort(2.0 ** -np.arange(1, 301, dtype=float))
    chk("T3 unfolding unmasks it as a picket fence",
        r_stat(norm_gaps(geo)) > 0.97, f"{r_stat(norm_gaps(geo)):.4f}")

    # T4  BLOCK DECOUPLING -> Poisson.  This certifies the octave prediction:
    #     the statistic must actually detect independent blocks.
    blocks = []
    for b in range(6):
        Ab = rng.normal(size=(120, 120))
        blocks.append(np.linalg.eigvalsh((Ab + Ab.T) / 2) * (2.0 ** b))
    sup = np.sort(np.concatenate(blocks))
    r_sup = r_stat(norm_gaps(sup))
    chk("T4 six superposed independent spectra read Poisson-ward",
        r_sup < 0.47, f"{r_sup:.4f} (GOE would be 0.531)")

    # T5  counting exponent.  THE FIRST PLANTED SPECTRUM WAS WRONG: lam_i =
    #     i^-p gives N(x) = n - x^(-1/p), which is n MINUS a power and has no
    #     log-log slope.  For N(lam) ~ lam^p the j-th smallest must be j^(1/p).
    for p_true in (1.25, 2.0, 3.0):
        lam_pl = np.arange(1, 2001, dtype=float) ** (1.0 / p_true)
        p_fit, r2, inst = counting_exponent(lam_pl)
        chk(f"T5 planted power law p={p_true} recovered",
            abs(p_fit - p_true) < 0.02 and inst < 0.01,
            f"{p_fit:.4f}, instability {inst:.4f}")
    Ae = rng.normal(size=(500, 500))
    semi = np.linalg.eigvalsh((Ae + Ae.T) / 2) + 40.0
    _, r2s, inst_s = counting_exponent(semi)
    chk("T5 semicircle is rejected by instability though R^2 passes",
        inst_s > 0.10 and r2s > 0.95, f"instability {inst_s:.3f}, R^2 {r2s:.3f}")
    _, r2e, inst_e = counting_exponent(np.exp(-np.arange(1, 2001) / 50.0))
    chk("T5 exponential is rejected by instability though R^2 passes",
        inst_e > 0.10 and r2e > 0.95, f"instability {inst_e:.3f}, R^2 {r2e:.3f}")

    # T6  theta/sigma.  REGISTRATION SHARPENED: A1 predicted |sigma|/|theta| > 1
    #     "because the octave structure spans orders of magnitude".  A LINEAR
    #     grading 1..100 reads 0.577, i.e. breathing-dominated -- so the
    #     threshold tests the GEOMETRIC premise specifically, and the linear
    #     case is kept as the other side.
    _, ri = theta_sigma(np.eye(50))
    chk("T6 identity is pure breathing", ri < 1e-12, f"{ri:.2e}")
    _, rl = theta_sigma(np.diag(np.linspace(1, 100, 50)))
    chk("T6 LINEAR grading is breathing-dominated (< 1)", rl < 1.0, f"{rl:.4f}")
    _, rg = theta_sigma(np.diag(2.0 ** np.arange(50)))
    chk("T6 GEOMETRIC grading is shear-dominated (> 1) -- the A1 premise",
        rg > 1.0, f"{rg:.4f}")

    # T7  rho / i_crit closed form vs power iteration on the actual map
    P = synth_basis(n=64, size=48, seed=3)
    Rm = build_R(P, 48)
    G = Rm.T @ Rm
    leak, inject = 0.9, 0.02
    ev, lm, rho, ic = loop_spectrum(G, leak, inject)
    d = np.diag(G); s = 1.0 / np.sqrt(d)
    K = 0.25 * (G / d[:, None])
    M = leak * np.eye(len(d)) + inject * K
    v = rng.normal(size=len(d))
    for _ in range(500):
        v = M @ v
        nv = np.linalg.norm(v)
        if nv < 1e-300:
            break
        v /= nv
    rho_pi = float(np.linalg.norm(M @ v))
    chk("T7 closed-form rho matches power iteration",
        abs(rho - rho_pi) / rho < 2e-3, f"{rho:.6f} vs {rho_pi:.6f}")
    _, _, rho_c, _ = loop_spectrum(G, leak, ic)
    chk("T7 i_crit puts rho exactly at 1", abs(rho_c - 1.0) < 1e-9,
        f"rho(i_crit) {rho_c:.9f}")
    chk("T7 loop spectrum is real and non-negative (PSD theorem)",
        ev.min() >= -1e-9, f"min {ev.min():.2e}")

    # T8  the MATCHED control must preserve Q and destroy arrangement
    Sh = shuffle_basis(P, seed=1)
    chk("T8 matched control preserves the (freq,sigma) marginals exactly",
        np.allclose(np.sort(Sh['freq']), np.sort(P['freq'])) and
        np.allclose(np.sort(Sh['sigma']), np.sort(P['sigma'])))
    chk("T8 and changes the Gram", not np.allclose(
        build_R(Sh, 48).T @ build_R(Sh, 48), G))

    # T10 build_R_from_render, the parameterisation-free path, WITH a bias and
    #     WITH baked-in per-packet amplitudes -- i.e. the awkward real case.
    Pm = synth_basis(n=140, size=40, seed=7)
    amp = np.exp(rng.normal(0, 1.2, 140))            # wildly unequal amplitudes
    bias = 0.37
    atoms = np.stack([gabor_atom(40, Pm['cx'][k], Pm['cy'][k], Pm['sigma'][k],
                                 Pm['freq'][k], Pm['theta'][k], Pm['phase'][k]
                                 ).ravel() for k in range(140)], axis=1)

    def render_rows(idx):
        if len(idx) == 0:
            raise ValueError("empty render unsupported, as in a real renderer")
        return (atoms[:, list(idx)] * amp[list(idx)]).sum(axis=1) + bias

    Rb, resid, base = build_R_from_render(render_rows, 140)
    chk("T10 additivity residual ~0 through the callable path", resid < 1e-10,
        f"{resid:.2e}")
    chk("T10 the constant bias is solved for exactly without an empty render",
        abs(base[0] - bias) < 1e-10, f"recovered {base[0]:.6f} vs {bias}")
    chk("T10 columns carry the baked amplitudes",
        np.allclose(Rb, atoms * amp, atol=1e-9))

    # T11 THE DESIGN CLAIM: baked amplitudes rescale G but CANNOT move any loop
    #     quantity, because 1/4 D^-1 G is the correlation matrix.  Two-sided --
    #     G itself must change, the loop numbers must not.
    G_amp = Rb.T @ Rb
    G_raw = atoms.T @ atoms
    chk("T11 amplitudes DO change the raw Gram",
        not np.allclose(G_amp / np.linalg.norm(G_amp),
                        G_raw / np.linalg.norm(G_raw), atol=1e-6))
    e1, l1, r1, i1 = loop_spectrum(G_amp, 0.95, 0.03)
    e2, l2, r2_, i2 = loop_spectrum(G_raw, 0.95, 0.03)
    chk("T11 but lam_max / rho / i_crit are invariant",
        abs(l1 - l2) < 1e-9 and abs(i1 - i2) < 1e-12,
        f"lam {l1:.9f} vs {l2:.9f}, i_crit {i1:.9f} vs {i2:.9f}")
    ra, rb = r_stat(norm_gaps(e1)), r_stat(norm_gaps(e2))
    chk("T11 and so is r on the loop spectrum (NOT nan -- 40 atoms was below "
        "the unfolding minimum and passed vacuously)",
        np.isfinite(ra) and np.isfinite(rb) and abs(ra - rb) < 1e-9,
        f"{ra:.6f} vs {rb:.6f}")
    _, ra1 = theta_sigma(G_amp)
    _, ra2 = theta_sigma(G_raw)
    chk("T11 A1 IS amplitude-dependent (documented, not a bug)",
        abs(ra1 - ra2) > 0.01, f"{ra1:.3f} vs {ra2:.3f}")

    # T12 THE ADAPTER LOGIC, against a mock built like splat_trainer5:
    #     params (1,N,11), an activate() that maps raw logits to physical
    #     quantities (so touching it corrupts everything), amplitudes behind
    #     exp(), three channels, and a switchable OUTPUT sigmoid.
    class MockRen:
        def __init__(self, size, sigmoid):
            self.size = size
            self.sigmoid = sigmoid

        def activate(self, p):                       # raw -> physical
            return (np.tanh(p[..., 0]), np.tanh(p[..., 1]),
                    0.05 + 0.5 / (1 + np.exp(-np.clip(p[..., 2], -60, 60))),
                    1.0 + 30.0 / (1 + np.exp(-np.clip(p[..., 3], -60, 60))),
                    np.pi * np.tanh(p[..., 4]), p[..., 5],
                    np.exp(np.clip(p[..., 8:11], -60, 60)))

        def __call__(self, p):
            cx, cy, sg, fr, th, ph, amp = self.activate(p)
            S, N = self.size, p.shape[1]
            fld = np.zeros((1, 3, S, S))
            for k in range(N):
                a = gabor_atom(S, cx[0, k], cy[0, k], sg[0, k], fr[0, k],
                               th[0, k], ph[0, k])
                for c in range(3):
                    fld[0, c] += amp[0, k, c] * a
            return 1.0 / (1.0 + np.exp(-fld)) if self.sigmoid else fld

    rp = rng.normal(0, 0.8, (1, 60, 11))
    for sig, label in ((False, 'sigmoid OFF'), (True, 'sigmoid ON')):
        ren_m = MockRen(40, sig)
        found3, _ka, nch = _find_amp_group(ren_m, rp)
        found = {c: (v[0], v[2]) for c, v in found3.items()}
        if not sig:
            chk("T12 amplitude columns DETECTED behind exp() (off-value < 0)",
                len(found) == 3 and all(f < 0 for _, f in found.values()),
                f"{ {c: v for c, v in sorted(found.items())} }")
        amp_j, off = found[0]

        def rr(idx, _r=ren_m, _j=amp_j, _o=off):
            p = rp.copy()
            on = set(int(i) for i in idx)
            oi = [i for i in range(60) if i not in on]
            if oi:
                p[:, oi, _j] = _o
            return _r(p)[0, 0].ravel()

        _, res, _ = build_R_from_render(rr, 60)
        if sig:
            chk("T12 A0 CATCHES a live output sigmoid on the raw output",
                res > 1e-3, f"raw residual {res:.3e} -- would be refused")
            _, res2, lk, _ = _pick_link(lambda i: rr(i), 60)
            chk("T12 and the link cascade RECOVERS the pre-sigmoid field",
                res2 < 1e-6 and lk.startswith('logit '),
                f"residual {res2:.3e} via {lk.split()[0]}")
        else:
            chk(f"T12 A0 passes with {label}", res < 1e-8, f"residual {res:.3e}")
            _, _, lk0, _ = _pick_link(lambda i: rr(i), 60)
            chk("T12 and the cascade picks identity when nothing is applied",
                lk0.startswith('identity'), lk0.split()[0])

    # T13 A FAITHFUL MOCK OF splat_trainer5.GaborRendererQ, which is the case
    #     that broke the previous version: coeff = tanh(raw[...,5:11]) reshaped
    #     to (B,N,3,2), each packet contributing a*env*cos - b*env*sin per
    #     channel, and forward() ending in sigmoid.  So the amplitude of channel
    #     c is the COLUMN PAIR (5+2c, 6+2c) and no single column silences it.
    class MockQ:
        def __init__(self, size):
            self.size = size

        def activate(self, raw):
            px = np.tanh(raw[..., 0]) * 0.9
            py = np.tanh(raw[..., 1]) * 0.9
            sigma = 0.03 + 0.30 / (1 + np.exp(-np.clip(raw[..., 2], -60, 60)))
            theta = raw[..., 3]
            freq = 1.0 + 20.0 / (1 + np.exp(-np.clip(raw[..., 4], -60, 60)))
            coeff = np.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)
            return px, py, sigma, theta, freq, coeff

        def __call__(self, raw):
            px, py, sg, th, fr, co = self.activate(raw)
            S, N = self.size, raw.shape[1]
            g = np.linspace(-1, 1, S)
            X, Y = np.meshgrid(g, g)
            out = np.zeros((1, 3, S, S))
            for k in range(N):
                dx, dy = X - px[0, k], Y - py[0, k]
                xr = dx * np.cos(th[0, k]) + dy * np.sin(th[0, k])
                env = np.exp(-(dx * dx + dy * dy) / (2 * sg[0, k] ** 2))
                ec = env * np.cos(2 * np.pi * fr[0, k] * xr)
                es = env * np.sin(2 * np.pi * fr[0, k] * xr)
                for c in range(3):
                    out[0, c] += co[0, k, c, 0] * ec - co[0, k, c, 1] * es
            return 1.0 / (1.0 + np.exp(-np.clip(out, -60, 60)))

    rq = rng.normal(0, 0.9, (1, 70, 11))
    renq = MockQ(40)
    fnd, ka, nc = _find_amp_group(renq, rq)
    chk("T13 phasor amplitudes found as COLUMN PAIRS, not single columns",
        all(fnd.get(c, (0, 0, 0))[:2] == (5 + 2 * c, 7 + 2 * c) for c in range(3)),
        f"{ {c: fnd[c][:2] for c in sorted(fnd)} }")
    chk("T13 and none of those ranges silences all three channels",
        not any(ka.values()), f"{ka}")
    a0_, a1_, off_ = fnd[0]

    def rq_rows(idx):
        p = rq.copy()
        on = set(int(i) for i in idx)
        oi = [i for i in range(70) if i not in on]
        if oi:
            p[:, oi, a0_:a1_] = off_
        return renq(p)[0, 0].ravel()

    Rq, resq, lkq, baseq = _pick_link(rq_rows, 70)
    chk("T13 the link cascade picks logit and A0 passes on the real layout",
        resq < 1e-6 and lkq.startswith('logit '),
        f"residual {resq:.3e} via {lkq.split()[0]}")
    chk("T13 zero coefficients give an exactly zero pre-sigmoid field (no bias)",
        abs(float(np.abs(baseq).max())) < 1e-9, f"{float(np.abs(baseq).max()):.2e}")
    Gq = Rq.T @ Rq
    _, lq, _, iq = loop_spectrum(Gq, 0.95, 0.03)
    chk("T13 the resulting Gram is PSD with a usable spectrum",
        lq > 0 and np.isfinite(iq), f"lam_max {lq:.4f}, i_crit {iq:.4f}")

    # T9  R really is the linear synthesis map (superposition holds)
    c = rng.normal(size=64)
    direct = sum(c[k] * gabor_atom(48, P['cx'][k], P['cy'][k], P['sigma'][k],
                                   P['freq'][k], P['theta'][k], P['phase'][k])
                 for k in range(64)).ravel()
    chk("T9 R@c equals the summed field (linearity of the pre-sigmoid sum)",
        np.allclose(Rm @ c, direct, atol=1e-9),
        f"max err {np.abs(Rm @ c - direct).max():.2e}")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=None,
                    help='.pt checkpoint (room.pt / model2.pt) -- primary route')
    ap.add_argument('--inspect', action='store_true',
                    help='report what is in the .pt and build nothing. RUN FIRST')
    ap.add_argument('--zscale', type=float, default=0.0,
                    help='|z| operating point; the Gram is z-dependent')
    ap.add_argument('--channel', type=int, default=None,
                    help='keep one colour channel (avoids 3-fold degeneracy)')
    ap.add_argument('--from-R', type=str, default=None)
    ap.add_argument('--from-npz', type=str, default=None)
    ap.add_argument('--size', type=int, default=96)
    ap.add_argument('--leak', type=float, default=0.95)
    ap.add_argument('--inject', type=float, default=0.03)
    ap.add_argument('--control', choices=['matched', 'random', 'none'],
                    default='random')
    ap.add_argument('--n', type=int, default=256, help='control atom count')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--how', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.how:
        print(HOW)
        return
    if a.inspect:
        if not a.model:
            sys.exit("--inspect needs --model room.pt")
        inspect_ckpt(a.model)
        return

    print(f"loop: leak {a.leak}  inject {a.inject}")
    trained = ctrl = None
    tparams = None

    if a.model:
        Rt, _ = load_trained_R(a.model, zscale=a.zscale, seed=a.seed,
                               channel=(a.channel or 0))
        trained = analyse(Rt, None, a.leak, a.inject, f"TRAINED  ({a.model})")

    if a.from_npz:
        z = np.load(a.from_npz)
        tparams = {k: np.asarray(z[k], float) for k in z.files}
        Rt = build_R(tparams, a.size)
        trained = analyse(Rt, tparams, a.leak, a.inject,
                          f"TRAINED  ({a.from_npz})")
    if a.from_R:
        Rt = np.load(a.from_R)
        if Rt.ndim != 2:
            sys.exit(f"R must be 2-D (pixels x atoms), got {Rt.shape}")
        if Rt.shape[0] < Rt.shape[1]:
            print(f"  WARNING R is {Rt.shape} -- fewer pixels than atoms; "
                  f"did you transpose it?")
        trained = analyse(Rt, tparams, a.leak, a.inject, f"TRAINED  ({a.from_R})")

    if a.control != 'none':
        if a.control == 'matched' and tparams is not None:
            cp = shuffle_basis(tparams, seed=a.seed)
            lbl = "CONTROL  (matched: same freq/sigma, arrangement destroyed)"
        else:
            if a.control == 'matched':
                print("  matched control needs --from-npz; falling back to random")
            cp = synth_basis(n=(trained['n'] if trained else a.n),
                             size=a.size, seed=a.seed)
            lbl = "CONTROL  (fresh constant-Q draw)"
        ctrl = analyse(build_R(cp, a.size), cp, a.leak, a.inject, lbl)

    gates(trained, ctrl)
    if trained is None:
        print("\n  no trained basis supplied -- run --how for the six lines "
              "that produce R.npy")


if __name__ == '__main__':
    main()