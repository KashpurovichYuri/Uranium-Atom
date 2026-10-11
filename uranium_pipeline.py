"""
uranium_pipeline.py — shared engine for the U-atom case studies.

Drivers (e.g. 14e/run_14e.py, 32e/...) configure this module and call main():
    import uranium_pipeline as up
    up.configure(ATOM=..., BASIS=..., ECP=..., ECP_KIND='std'|'grpp', SPIN=4,
                 CORE_MAP=..., CAS_MULTIPLETS=..., STAGE1=..., STAGE3=..., ...)
    up.main()

Pipeline (per run):
  stage1_scf    — SCF variants from STAGE1 (strategy mini-language)
  stage2_cas    — declarative seed -> CAS pick (subshell completion) ->
                  CASCI pre-flight row -> CASSCF + NOs + configuration histogram
  spectroscopy  — optional multi-root CASCI scan on frozen seed-row MOs;
                  per-root E, dE (eV/cm^-1) and COMPUTED <S^2> (no hardcoded
                  spin labels: septet leakage into an (na,nb) sector becomes
                  visible); L^2 labels pending the full-space L^2 fix
  stage3_scf    — SCF experiments from STAGE3: seeded or cold
  report        — final table (text + results.md)

Analysis contract (who carries what):
  * CHARACTER (l-map): Lowdin weights on any orthonormal set — ROHF MOs
    (pick_cas), CASSCF MOs (histogram labels); occupations are never
    attached to CASSCF MOs.
  * POPULATIONS (valence fingerprint): character x occupations, valid only
    where the 1-RDM is diagonal — SCF MOs directly; NOs+NOONs for CAS
    objects (the only frame with invariant occupations).
  * GUESS (stage 3): NOs + NOON ranking -> dm0; a practical reuse of the
    analysis frame, not an analysis itself (= mc.make_rdm1s() alternative).

Session-verified PySCF 2.13.1 facts:
  * pyscf.scf.smearing.smearing(mf, sigma=...); smearing an ROHF object
    recasts it to the RHF class (no ROHF smearing); SMEAR/SMEARFS/CLEAN
    rows are therefore never warm-seeded (mu re-optimization reshuffles
    the fractional frontier; the tiny verify budget cannot re-flag).
  * mcscf chkfile: data under the 'mcscf' key (dump_mcscf); CI vector
    ('ci_vector') only when mc.chk_ci = True (may still be absent in the
    stored file — regenerate via CASCI when needed).
  * CASCI.make_rdm1 returns the density in the non-orthogonal AO
    representation: occupations require eigh(S dm S, b=S).

FUNDAMENTALS (angular momentum of an atomic CAS):
  * A term (L, S) is a coherent superposition over m-occupations: the
    M-ladder components of one term occupy DIFFERENT m-functions (p^2
    triplet: M_L=1 is |+1,0>, M_L=0 is (|+1,-1>+|-1,+1>)/sqrt2 — the
    |-1> function appears although the top component never touches it).
  * An L^2 eigenvector exists in a CAS iff the active one-electron space
    is l-closed (carries an SO(3) representation: a sum of complete
    l-multiplets, e.g. 1s + 5d + 7f).
  * The Hamiltonian is a rotational scalar => all M_L components of a term
    are degenerate => the energy is blind to which m's were dropped. That
    is why state-specific CASSCF trades multiplet completeness for cheap
    correlators (7p) regardless of the pick: energetically fine, L-labels
    lost.
  * Division of labor: energy/fingerprint from the relaxed CASSCF (honest
    variational result); term labels from the frozen multiplet-complete
    span. Canonical check: carbon CAS(2,3) <L^2> = 2.000000, CAS(2,2)
    (truncated) != 2.
"""
import os, sys, time, inspect
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
from scipy.linalg import eigh as _sc_eigh
from pyscf import gto, scf, dft, mcscf, fci
from pyscf.mcscf import addons
from pyscf.scf.smearing import smearing as _smear_fn
from pyscf.lib import chkfile as lib_chk

from lmom.expectation import L2Lz_expectation_dm   # verified evaluator (lmom)

# ============================ DEFAULTS (drivers override) =====================
ECP_KIND = 'std'                 # 'std': gto.M + ecp dict; 'grpp': GRPPMole
ATOM = 'U 0 0 0'
BASIS = None                     # {'U': parsed basis}
ECP = None                       # {'U': NWChem-format string} (std) / GRECP (grpp)
UNIT, CHARGE, SPIN = 'B', 0, 4
CORE_MAP = {0: 0, 1: 0, 2: 0}    # closed shells subtracted from populations
NELECAS = (5, 1)                 # active electrons (na, nb)
NCAS = 13
CAS_MANUAL = None                # 1-based MO list to override pick_cas
CAS_MULTIPLETS = {'s': 1, 'd': 5, 'f': 7}   # subshell-completion targets
TARGET_CFG = {'s': 2.0, 'd': 1.0, 'f': 3.0}
CFG_TOL, CFG_PRINT_TOL = 0.3, 0.05
XC, SMEAR_SIGMA = 'pbe', 0.02
WARM, WARM_MAX_SCF, WARM_MAX_MACRO = True, 3, 6
FORCE_COLD_CAS = False
MAX_DIIS, MAX_SOSCF, MAX_MACRO = 100, 50, 50
DIIS_START, CONV_TOL = 1, 1e-9
VERBOSE, SCF_VERBOSE = 4, 3
CSF_CUTOFF, PRINT_CSF = 0.05, True
HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.63
TERM_LETTERS = 'SPDFGHIKLMNOQRTU'
STAGE1 = [('ROHF', 'ROHF', ('DIIS', 'SOSCF'))]
STAGE2_SEED = 'rohf'
STAGE3 = [('ROHF<-CAS', 'ROHF', 'casscf', ('DIIS', 'SOSCF'))]
ROOT_SECTORS = []                # e.g. [((5,1), 20), ((6,0), 6)]; empty = off
SPECTROSCOPY = True
CAS_CHK = 'casscf.chk'
CHAR_EVERY = 10                  # CHAR lock: heartbeat interval (calls)

_CONFIGURABLE = [k for k, v in list(globals().items())
                 if k.isupper() and not k.startswith('_') and k != '_CONFIGURABLE']


def configure(**kw):
    """Override module-level configuration; unknown keys are rejected."""
    unknown = set(kw) - set(_CONFIGURABLE)
    if unknown:
        raise KeyError(f'unknown config keys: {sorted(unknown)}')
    globals().update(kw)


# ============================ BOOKKEEPING =====================================
class CycleCounter:
    """Per-iteration bookkeeping via callback(envs): cycles/macros spent,
    last |dE|, last |grad|. mcscf fires the callback twice per macro and
    carries 'imacro' -> distinct-imacro count is the true macro total.
    Cells are pipe-free ('|' breaks markdown table columns)."""
    KEYS_E = ('e_tot', 'etot')
    KEYS_G = ('norm_g', 'norm_gorb', 'grad_norm', 'gnorm')

    def __init__(self):
        self.n = 0; self.e_last = self.e_prev = None; self.g_last = None
        self.macros = set()

    def __call__(self, envs):
        self.n += 1
        im = envs.get('imacro')
        if im is not None:
            self.macros.add(int(im))
        e = next((envs[k] for k in self.KEYS_E if k in envs), None)
        if e is not None:
            self.e_prev, self.e_last = self.e_last, float(e)
        g = next((envs[k] for k in self.KEYS_G if k in envs), None)
        if g is not None:
            self.g_last = float(g)

    @property
    def prec(self):
        parts = []
        if None not in (self.e_last, self.e_prev):
            parts.append(f'dE {abs(self.e_last - self.e_prev):.1e}')
        if self.g_last is not None:
            parts.append(f'g {self.g_last:.1e}')
        return ', '.join(parts) if parts else '—'

    def summary(self):
        if self.macros:
            return f'{len(self.macros)} macro'
        return f'{self.n} cyc' if self.n else None


def _counted_kernel(mf, *kargs):
    cc = CycleCounter(); mf.callback = cc
    t0 = time.perf_counter()
    mf.kernel(*kargs)
    return cc, time.perf_counter() - t0


def _stage(label, cc):
    s = cc.summary()
    return f'{label} {s}' if s else label


def slug(name):
    """Row name -> chkfile/seed tag; duplicate underscores collapsed."""
    s = (name.lower().replace('<', '_').replace('-', '_')
         .replace('*', '').replace(' ', '_'))
    while '__' in s:
        s = s.replace('__', '_')
    return s


def fmt_cfg(cfg):
    return ' '.join(f'{l}^{n:.2f}' for l, n in cfg.items()) if cfg else '(none)'


def config_matches(cfg):
    if not cfg:
        return False
    return all(l in cfg and abs(cfg[l] - n) <= CFG_TOL
               for l, n in TARGET_CFG.items())


# ============================ ANALYSIS ========================================
def ao_l_list(mol):
    """Angular momentum l of every AO column."""
    out = []
    for b in range(mol.nbas):
        l = mol.bas_angular(b)
        out += [l] * (2 * l + 1) * mol.bas_nctr(b)
    return np.array(out)


def l_weights(ctx, mo):
    """Lowdin l-composition: w[l,i] = fraction of MO i's norm carried by AOs
    of momentum l. Squared coefficients are taken in the symmetrically
    orthogonalized AO frame (C' = S^(1/2) C, C'^T C' = I), so the partition
    is exact as bookkeeping: w >= 0, sum_l w[l,i] = 1, and the electron
    count sum_l n_l = N_elec holds by construction. For a single atom S is
    block-diagonal in l, so the orthogonalization keeps l-blocks separate
    and the l-attribution of each MO is unambiguous; the split among
    functions of one l remains a statement about this basis (populations
    are not observables). UHF spin blocks are handled in valence_config,
    not here ((C_a + C_b)/2 is not S-orthonormal). Character labeling is
    frame-safe: any orthonormal set works (SCF MOs, CASSCF MOs, NOs);
    electron counting is deliberately NOT done here — see valence_config."""
    if mo is None:
        raise RuntimeError('mo_coeff is None — SCF object was never run!')
    mo = np.asarray(mo, dtype=float)
    assert mo.ndim == 2, f'expect one (nao,nmo) matrix, got {mo.shape}'
    assert mo.shape[0] == ctx.l_ao.size == ctx.S.shape[0]
    s2, U = np.linalg.eigh(ctx.S)
    Co = (U * np.sqrt(s2)) @ U.T @ mo
    W = np.stack([(Co[ctx.l_ao == l]**2).sum(0)
                  for l in range(int(ctx.l_ao.max()) + 1)])
    assert np.allclose(W.sum(0), 1.0, atol=1e-8)
    return W


def valence_config(ctx, occ, mo, etol=CFG_PRINT_TOL):
    """Open-valence electrons per angular momentum,
        n_l = sum_i n_i * w[l,i]  -  CORE_MAP[l]        (Lowdin populations)
    with sum_l n_l = sum_i n_i = N_elec exactly (assert). Contract: occ must
    be occupations in a frame where the 1-RDM is diagonal — SCF MOs qualify
    directly; for CAS objects only NOONs do (CASSCF MOs carry no
    occupations). n_i = occupations: RHF/ROHF 2/1/0; UHF/UKS (2,nmo) with
    per-spin Lowdin weights then summed; CASSCF: NOONs. The CASSCF result
    is the DENSITY's shell population, i.e. an ensemble average over the CI
    determinants (hence soft values)."""
    occ, mo = np.asarray(occ, dtype=float), np.asarray(mo, dtype=float)
    if occ.ndim == 2:
        assert mo.ndim == 3 and mo.shape[0] == occ.shape[0]
        nel = sum(o @ l_weights(ctx, c).T for o, c in zip(occ, mo))
    else:
        assert mo.ndim == 2
        nel = occ @ l_weights(ctx, mo).T
    assert abs(nel.sum() - ctx.mol.nelectron) < 1e-6, \
        f'{nel.sum()} != {ctx.mol.nelectron}'
    tags = 'spdfgh'
    raw = {tags[l]: round(float(n), 2) for l, n in enumerate(nel) if n > etol}
    val = {tags[l]: round(float(n - CORE_MAP.get(l, 0)), 2)
           for l, n in enumerate(nel) if n - CORE_MAP.get(l, 0) > etol}
    print('  per-l raw:', raw, '\n  valence  :', val)
    return val


def char_occ_counts(ctx):
    """Per-l, per-spin electron counts that hold TARGET_CFG (e.g.
    7s^2 6d^1 5f^3) on top of the CORE_MAP core: spin-paired core takes
    CORE_MAP[l]/2 per spin; closed valence subshells pair up; open
    subshells stay alpha unless the beta budget (nb - core) demands
    pairing (Hund: the 7s^2 6d^1 5f^3 quintet -> s:2a/2b incl. core,
    d:1a, f:3a). Single source of truth for the fixed_occ_config
    template and the CHAR per-cycle lock. Returns (cnt_a, cnt_b),
    dicts l -> electron count."""
    tags = 'spdfgh'
    assert all(n % 2 == 0 for n in CORE_MAP.values()), \
        'CORE_MAP subshells must be spin-paired'
    nl_core = {l: CORE_MAP.get(l, 0) for l in range(len(tags))}
    nl_val = {tags.index(k): int(round(v)) for k, v in TARGET_CFG.items()}
    nb_val = ctx.nb - sum(n // 2 for n in CORE_MAP.values())
    for l, nv in nl_val.items():          # closed subshells pair up first
        if nv % 2 == 0:
            nb_val -= nv // 2
    assert nb_val >= 0, 'negative beta budget — check CORE_MAP/SPIN/TARGET_CFG'
    cnt_a, cnt_b = {}, {}
    for l in sorted(set(nl_core) | set(nl_val)):
        nc, nv = nl_core.get(l, 0) // 2, nl_val.get(l, 0)
        nb_l = nv // 2 if nv % 2 == 0 else min(nv // 2, nb_val)
        if nv % 2:
            nb_val -= nb_l
        cnt_a[l], cnt_b[l] = nc + nv - nb_l, nc + nb_l
    return cnt_a, cnt_b


def pick_by_character(ctx, mo, rank, cnt_a, cnt_b, verbose=False, label=''):
    """Occupation vectors from a Lowdin-character pick: per l, the
    cnt_a/cnt_b lowest-`rank` MOs of dominant l character (rank =
    orbital energies, ascending; -NOON for natural orbitals; a 2D rank
    is spin-averaged first). Stateless assignment rule shared by the
    fixed_occ_config template and the CHAR per-cycle lock — same counts,
    same rule. A character shortfall (not enough MOs of some l to hold
    the electrons) is a hard error; everything else about the pick is
    printed only when verbose. Returns (ferwe, ferdo, W, dom): the occ
    vectors and the l-weight view for printing."""
    mo = np.asarray(mo, dtype=float)
    assert mo.ndim == 2, 'expect one (nao,nmo) matrix'
    nmo = mo.shape[1]
    rank = np.asarray(rank, dtype=float)
    if rank.ndim == 2:                    # UHF mo_energy -> spin-averaged
        rank = rank.mean(0)
    assert rank.shape == (nmo,)
    W = l_weights(ctx, mo)
    dom = W.argmax(0)
    tags = 'spdfgh'
    ferwe, ferdo = np.zeros(nmo), np.zeros(nmo)
    if verbose:
        print(f'\n--- character occ picks ({label}) ---')
    for l in range(len(tags)):
        ca, cb = cnt_a.get(l, 0), cnt_b.get(l, 0)
        if not ca and not cb:
            continue
        cand = sorted((i for i in range(nmo) if dom[i] == l),
                      key=lambda i: rank[i])
        if len(cand) < max(ca, cb):
            raise RuntimeError(f'character pick: only {len(cand)} '
                               f'{tags[l]}-character MOs, need {max(ca, cb)}')
        picked = cand[:max(ca, cb)]
        for j, i in enumerate(picked):
            if j < ca:
                ferwe[i] = 1.0
            if j < cb:
                ferdo[i] = 1.0
        if verbose:
            for i in picked:
                flag = '  <-- LOW PURITY' if W[l, i] < 0.5 else ''
                print(f'  MO {i+1:3d}  rank={rank[i]:+.4f}  char={tags[l]}  '
                      f'purity={W[l, i]:.2f}  occ={ferwe[i] + ferdo[i]:.1f}'
                      f'{flag}')
    assert ferwe.sum() == ctx.na and ferdo.sum() == ctx.nb, \
        f'character pick sums ({ferwe.sum():.0f},{ferdo.sum():.0f}) != ' \
        f'(na,nb) = ({ctx.na},{ctx.nb})'
    return ferwe, ferdo, W, dom


def fixed_occ_config(ctx, mo, rank, label=''):
    """Fixed-occupancy template for TARGET_CFG (e.g. 7s^2 6d^1 5f^3) on a
    given orbital set: char_occ_counts + pick_by_character, ranked by
    ascending `rank` (mo_energy for SCF MOs, -NOON for natural orbitals).
    Validation is warning-only (deviating template config, low purity).
    Returns (ferwe, ferdo), 1.0/0.0 per MO per spin."""
    ferwe, ferdo, W, dom = pick_by_character(
        ctx, mo, rank, *char_occ_counts(ctx), verbose=True, label=label)
    val = valence_config(ctx, np.array([ferwe, ferdo]), np.array([mo, mo]))
    if not config_matches(val):
        print(f'  WARNING: template config {fmt_cfg(val)} deviates from '
              f'TARGET_CFG {fmt_cfg(TARGET_CFG)} (tol {CFG_TOL})')
    return ferwe, ferdo


def make_char_occ(ctx, label, occ_template, nmo):
    """CHAR occupation lock for an SCF object: a get_occ that re-derives
    the occupations from the CURRENT MOs at every cycle — pick_by_character
    with the fixed char_occ_counts, ranked by current orbital energy. The
    held quantity is the subshell composition itself (e.g. 7s^2 6d^1 5f^3):
    MO reordering is absorbed, and a gradual d/f band drift cannot change
    the occupied character set (contrast MOM, which follows the band and
    can slip across a seam, and FIXOCC, which pins indices and silently
    changes meaning on reordering). Audit trail: full assignment table on
    call 1 (warned if it differs from the fixed_occ_config template), a
    FLIP block whenever the assignment changes, a heartbeat every
    CHAR_EVERY calls, and a report(final_mf) closing table. Returns
    (get_occ, report); report returns the flip count."""
    tags = 'spdfgh'
    cnt_a, cnt_b = char_occ_counts(ctx)
    state = dict(calls=0, flips=0, prev=None)

    def _assign(mo_coeff, mo_energy):
        mo_coeff = np.asarray(mo_coeff, dtype=float)
        mo_energy = np.asarray(mo_energy, dtype=float)
        if mo_coeff.ndim == 2:            # restricted-form -> spin pair
            mo_coeff = np.array([mo_coeff, mo_coeff])
        if mo_energy.ndim == 1:
            mo_energy = np.array([mo_energy, mo_energy])
        occ = np.zeros((2, nmo))
        view = []
        for s in (0, 1):
            fa, fb, W, dom = pick_by_character(ctx, mo_coeff[s],
                                               mo_energy[s], cnt_a, cnt_b)
            occ[s] = fa if s == 0 else fb
            view.append((W, dom, np.asarray(mo_energy[s], dtype=float)))
        return occ, view

    def _mo_line(s, i, view, occ):
        W, dom, e_s = view[s]
        l = int(dom[i])
        return (f'{"a" if s == 0 else "b"} MO {i+1:3d}  {tags[l]}  '
                f'e={e_s[i]:+.4f}  purity={W[l, i]:.2f}  occ={occ[s, i]:.0f}')

    def get_occ(mo_energy=None, mo_coeff=None):
        occ, view = _assign(mo_coeff, mo_energy)
        state['calls'] += 1
        if state['prev'] is None:
            print(f'\n--- CHAR occ assignment, call 1 ({label}) ---')
            for s in (0, 1):
                for i in np.where(occ[s] > 0)[0]:
                    print('  ' + _mo_line(s, i, view, occ))
            if not np.array_equal(occ, occ_template):
                print('  WARNING: call-1 assignment differs from the '
                      'fixed_occ_config template (seed MOs already drifted)')
        elif not np.array_equal(occ, state['prev']):
            state['flips'] += 1
            print(f'\nCHAR FLIP #{state["flips"]} at call {state["calls"]} '
                  f'({label}):')
            for s in (0, 1):
                for i in np.where((occ[s] > 0) & (state['prev'][s] <= 0))[0]:
                    print('  ' + _mo_line(s, i, view, occ) + '  ENTERS')
                for i in np.where((occ[s] <= 0) & (state['prev'][s] > 0))[0]:
                    print('  ' + _mo_line(s, i, view, occ) + '  leaves')
        elif state['calls'] % CHAR_EVERY == 0:
            print(f'CHAR: stable through call {state["calls"]} '
                  f'({label}; flips {state["flips"]})')
        state['prev'] = occ
        get_occ.flips = state['flips']
        return occ

    def report(mf_now):
        """Closing audit: assignment recomputed on the final MOs."""
        occ, view = _assign(mf_now.mo_coeff, mf_now.mo_energy)
        print(f'\n--- CHAR final assignment ({label}); calls '
              f'{state["calls"]}, flips {state["flips"]} ---')
        for s in (0, 1):
            for i in np.where(occ[s] > 0)[0]:
                print('  ' + _mo_line(s, i, view, occ))
        return state['flips']

    get_occ.flips = 0
    return get_occ, report


def large_ci_cut(ci, ncas, nelecas, cut=CSF_CUTOFF):
    """fci.addons.large_ci with the cutoff keyword located by signature."""
    params = inspect.signature(fci.addons.large_ci).parameters
    for kw in ('tol', 'lim', 'cutoff'):
        if kw in params:
            return fci.addons.large_ci(ci, ncas, nelecas, **{kw: cut})
    return fci.addons.large_ci(ci, ncas, nelecas, cut)


def build_system():
    """Molecule + analysis context. ECP_KIND='grpp': GRPPMole + integral
    cross-check against PySCF; 'std': plain gto.M with an ECP dict."""
    if ECP_KIND == 'grpp':
        import libgrpp4pyscf
        mol = libgrpp4pyscf.GRPPMole()
        mol.build(atom=ATOM, basis=BASIS, ecp=ECP, spin=SPIN, charge=CHARGE,
                  unit=UNIT, verbose=VERBOSE)
        libgrpp4pyscf.compare_integrals_with_pyscf(mol, verbose=True,
                                                   gradients=False)
    else:
        mol = gto.M(atom=ATOM, basis=BASIS, ecp=ECP, spin=SPIN, charge=CHARGE,
                    unit=UNIT, verbose=VERBOSE)
    l_ao = ao_l_list(mol)
    assert not mol.cart and (l_ao == 3).sum() >= 7, 'need spherical f functions'
    return SimpleNamespace(
        mol=mol, S=mol.intor('int1e_ovlp'), l_ao=l_ao,
        na=(mol.nelectron + SPIN) // 2,
        nb=(mol.nelectron - SPIN) // 2,
        rows=[], mf_ref=None, mc=None, noon=None, natorb=None, cas_idx=None)


def make_scf(ctx, kind, xc=None):
    mf = {'ROHF': ctx.mol.ROHF, 'UHF': ctx.mol.UHF, 'UKS': ctx.mol.UKS}[kind]()
    if xc:
        mf.xc = xc
    mf.verbose = SCF_VERBOSE
    return mf


def seed_from_chk(mf, tag):
    """WARM: previous run's converged orbitals as the starting guess; the
    short re-run certifies the stored solution (stationarity check)."""
    path = f'{tag}.chk'
    if not (WARM and os.path.exists(path)):
        return False, None
    try:
        mol_chk, rec = scf.chkfile.load_scf(path)
        if (mol_chk.nao_nr() != mf.mol.nao_nr()
                or mol_chk.nelectron != mf.mol.nelectron):
            print(f'  [{tag}] chk belongs to a different system -> cold start')
            return False, None
        mf.mo_coeff, mf.mo_occ = rec['mo_coeff'], rec['mo_occ']
        if rec.get('mo_energy') is not None:
            mf.mo_energy = rec['mo_energy']
        print(f'  [{tag}] warm seed (E_prev={rec["e_tot"]:.8f})')
        return True, rec.get('e_tot')
    except Exception as ex:
        print(f'  [{tag}] chk seed failed ({ex}) -> cold start')
        return False, None


def run_scf(ctx, name, kind, strategy=('DIIS', 'SOSCF'), xc=None, dm0=None,
            occ_seed=None):
    """One SCF variant via the strategy mini-language. Warm seeding
    overrides the initial dm and shortens budgets, and the re-run doubles
    as a stationarity check — except SMEAR/SMEARFS/CLEAN strategies, which
    always run cold (module docstring explains why).

    Fixed-occupancy modifiers (run-to-run reoccupation lock, e.g. to hold
    7s^2 6d^1 5f^3 on the UKS branch where PBE prefers 5f^4):
      FIXOCC — index-pinned get_occ (user-supplied occ vector verbatim);
      MOM    — scf.addons.mom_occ: every cycle the occupied set maximizes
               overlap with the seed's occupied columns (ordering-safe,
               but follows the orbital band and can drift across a seam);
      CHAR   — make_char_occ: every cycle the occupations are re-derived
               from the current MOs by Lowdin character with fixed
               per-l/per-spin counts — the held quantity is the subshell
               composition itself.
    All need occ_seed: a row slug run earlier, or 'casscf' (NOs/NOONs).
    They force a cold start from the seed-row MOs (a chk warm seed would
    contaminate the template) and hand SOSCF (mo_coeff, mo_occ) instead of
    a dm, so the occupations stay ours (newton discards re-occupation)."""
    if isinstance(strategy, str):
        raise TypeError(f"strategy for '{name}' is a string — add the "
                        f"trailing comma: ({strategy!r},)")
    tag = slug(name)
    mf = make_scf(ctx, kind, xc)
    fix_mode = next((st for st in strategy
                     if st in ('FIXOCC', 'MOM', 'CHAR')), None)
    fix_note, occ_mo, ferwe, ferdo = None, None, None, None
    char_occ_fn = char_report = None
    if fix_mode:
        if [st for st in strategy
                if st in ('FIXOCC', 'MOM', 'CHAR')] != [fix_mode] \
                or strategy.index(fix_mode) != 0:
            raise ValueError(f"'{name}': {fix_mode} must appear exactly "
                             'once, first in the strategy')
        if occ_seed is None:
            raise ValueError(f"'{name}': {fix_mode} needs an occ-template "
                             'seed (STAGE1 4th field / STAGE3 seed field)')
        if occ_seed == 'casscf':
            if ctx.natorb is None:
                raise RuntimeError("'casscf' occ seed requested before "
                                   'CASSCF has run')
            occ_mo, occ_rank = ctx.natorb, -ctx.noon
        else:
            row = next((r for r in ctx.rows
                        if slug(r['name']) == occ_seed), None)
            if row is None:
                raise KeyError(f'occ-template seed row {occ_seed!r} not '
                               f'found — it must run EARLIER than {name!r}')
            occ_mo = np.asarray(row['mf'].mo_coeff, dtype=float)
            occ_rank = row['mf'].mo_energy
            if occ_mo.ndim == 3:          # UHF-class seed: template on the
                occ_mo, occ_rank = occ_mo[0], np.asarray(  # alpha MO set
                    occ_rank, dtype=float)[0]
        ferwe, ferdo = fixed_occ_config(ctx, occ_mo, occ_rank, label=name)
        occ2 = np.array([ferwe, ferdo])
        # UHF-class make_rdm1/mom_occ need the (2,nao,nmo) spin form
        occ_orb = occ_mo if occ_mo.ndim == 3 else np.array([occ_mo, occ_mo])
        dm0 = mf.make_rdm1(occ_orb, occ2)
        if fix_mode == 'MOM':
            mf = scf.addons.mom_occ(mf, occ_orb, occ2)
        elif fix_mode == 'CHAR':
            char_occ_fn, char_report = make_char_occ(
                ctx, name, occ2, occ2.shape[1])
            mf.get_occ = char_occ_fn
        else:
            def _fixed_occ(mo_energy=None, mo_coeff=None,
                           fw=ferwe.copy(), fd=ferdo.copy()):
                return np.array([fw, fd])
            mf.get_occ = _fixed_occ
        fix_note = (f'{fix_mode}-locked occ {fmt_cfg(TARGET_CFG)}')
    # Smeared rows always run cold (chk warm-seeding disabled). The
    # frontier — here the nearly degenerate valence shell cluster — holds
    # fractional occupations under smearing; which orbital takes which
    # fraction is set by a chemical potential re-solved every cycle. On a
    # warm restart a ~1e-7 orbital-energy drift re-deals the fractions, the
    # density jumps, and the tiny verify budget cannot re-converge; cold
    # these rows take <1 minute.
    if (any(st in ('SMEAR', 'SMEARFS', 'CLEAN') for st in strategy)
            or fix_mode):
        seeded, e_prev = False, None
    else:
        seeded, e_prev = seed_from_chk(mf, tag)
    if seeded:
        dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        notes = ['warm verify']
    else:
        notes = []
    if fix_note:
        notes.append(f'{fix_note}; cold start from occ-template dm')
    mf.chkfile, mf.conv_tol = f'{tag}.chk', CONV_TOL
    stages, wall, cc_last = [], 0.0, None
    soscf_done = False
    for st in strategy:
        if st in ('FIXOCC', 'MOM', 'CHAR'):
            continue                      # applied above, before seeding
        if st == 'DIIS':
            if soscf_done:
                # DIIS after SOSCF: the newton wrapper would swallow the
                # positional dm0 as mo_coeff (its kernel then re-runs
                # second-order from the drifted state) — unwrap to the
                # base SCF, sync the wrapper's final state onto it, and
                # rebuild the start density from the current MOs/occ
                # (CHAR: occupations re-assigned on the final MOs).
                mf._scf.mo_coeff = mf.mo_coeff
                mf._scf.mo_energy = mf.mo_energy
                mf._scf.mo_occ = mf.mo_occ
                mf = mf._scf
                occ_start = (char_occ_fn(mf.mo_energy, mf.mo_coeff)
                             if fix_mode == 'CHAR' else mf.mo_occ)
                dm0 = mf.make_rdm1(mf.mo_coeff, occ_start)
                soscf_done = False
            mf.diis_start_cycle = DIIS_START
            mf.max_cycle = WARM_MAX_SCF if seeded else MAX_DIIS
            cc, t = _counted_kernel(mf, dm0)
            stages.append(_stage('DIIS', cc))
        elif st == 'SOSCF':
            mf = mf.newton()
            mf.max_cycle = WARM_MAX_SCF if seeded else MAX_SOSCF
            # fixed-occ rows: pass MOs+occupations (not a dm) so the SOSCF
            # start honors the template; newton keeps occ fixed (it calls
            # get_occ only to discard the result). CHAR: re-assign on the
            # DIIS-final MOs so the handed-over occ matches their order.
            if fix_mode == 'CHAR':
                kargs = (mf.mo_coeff,
                         char_occ_fn(mf.mo_energy, mf.mo_coeff))
            elif fix_mode:
                kargs = (occ_orb, np.array([ferwe, ferdo]))
            else:
                kargs = (dm0,)
            cc, t = _counted_kernel(mf, *kargs)
            soscf_done = True
            stages.append(_stage('SOSCF', cc))
        elif st == 'SMEAR':
            mf = _smear_fn(mf, sigma=SMEAR_SIGMA)
            # smearing recasts ROHF as an RHF-class object (PySCF applies
            # ONE chemical potential to ONE set of orbital energies; ROHF
            # has two Fock matrices) — its kernel wants the spin-summed
            # total dm, not the spin-resolved (2,nao,nao) of a native ROHF:
            # rebuild it from the current (possibly fractional) occ —
            # ROHF.make_rdm1 would binarize them and break the electron
            # count (smeared UKS stays UHF-like: 3D ok)
            if (dm0 is not None and np.ndim(dm0) == 3
                    and not mf.istype('UHF')):
                dm0 = np.dot(mf.mo_coeff * mf.mo_occ, mf.mo_coeff.T)
            mf.max_cycle = WARM_MAX_SCF if seeded else MAX_DIIS
            cc, t = _counted_kernel(mf, dm0)
            stages.append(_stage('SMEAR', cc))
            notes.append(f'gaussian smear sigma={SMEAR_SIGMA}')
        elif st == 'SMEARFS':
            mf = _smear_fn(mf, sigma=SMEAR_SIGMA, fix_spin=True)
            mf.max_cycle = MAX_DIIS
            cc, t = _counted_kernel(mf, dm0)
            stages.append(_stage('SMEARFS', cc))
            notes.append(f'gaussian smear sigma={SMEAR_SIGMA}, fix_spin')
        elif st == 'CLEAN':
            dm = mf.make_rdm1()
            mf = make_scf(ctx, kind, xc)
            mf.chkfile, mf.conv_tol = f'{tag}.chk', CONV_TOL
            mf.max_cycle = MAX_DIIS
            cc, t = _counted_kernel(mf, dm)
            stages.append(_stage('CLEAN', cc))
            notes.append('integer restart from smeared dm')
        else:
            raise ValueError(f'unknown strategy stage: {st}')
        wall += t
        cc_last = cc
        if mf.converged:
            # CHAR: a mid-strategy convergence may still have morphed the
            # locked composition (newton freezes index-based occupations
            # and lets their character drift) — re-check and, if drifted,
            # fall through to the remaining stages (e.g. a trailing DIIS
            # that re-engages the lock).
            drifted = (fix_mode == 'CHAR' and char_occ_fn is not None
                       and st != strategy[-1]
                       and not np.array_equal(
                           np.asarray(mf.mo_occ, dtype=float),
                           char_occ_fn(mf.mo_energy, mf.mo_coeff)))
            if drifted:
                print(f"CHAR: '{st}' converged but drifted from the locked "
                      'composition — re-locking with the remaining stages')
            else:
                break
    if fix_mode == 'CHAR':
        mf._char_flips = char_report(mf)
    if seeded and e_prev is not None and mf.converged:
        d = abs(mf.e_tot - e_prev)
        notes.append(f'stationary: dE_vs_chk {d:.1e}')
        if d > 1e-7:
            notes.append('CHECK: differs from chk!')
    mf._meta = dict(cycles=' + '.join(stages), t=wall,
                    prec=cc_last.prec if cc_last else '—')
    return mf, '; '.join(notes) or '->'.join(strategy)


def add_row(ctx, name, mf, note, s2=None, l2=float('nan'),
            lz=float('nan'), cfg=None):
    """Register one variant. <Sz> from occupations: per-spin occ
    (UHF/UKS class, incl. fix_spin smearing) give the exact value; ROHF and
    CAS(n) have (na-nb)/2; spin-summed smeared ensembles have none."""
    conv = bool(getattr(mf, 'converged', False))
    if conv and s2 is None:
        try:
            s2 = mf.spin_square()[0]
        except Exception:
            s2 = float('nan')
    if conv and cfg is None:
        cfg = valence_config(ctx, mf.mo_occ, mf.mo_coeff)
    occ = np.asarray(mf.mo_occ, dtype=float)
    if occ.ndim == 2:
        szv = float(occ[0].sum() - occ[1].sum()) / 2
    elif hasattr(mf, 'ncas'):
        szv = (ctx.na - ctx.nb) / 2
    elif hasattr(mf, 'istype') and mf.istype('ROHF'):
        szv = (ctx.na - ctx.nb) / 2
    else:
        szv = None
    meta = getattr(mf, '_meta', {})
    nflip = getattr(mf, '_char_flips', None)
    if nflip is not None:
        note = f'{note}; CHAR flips {nflip}'
    ctx.rows.append(dict(name=name, mf=mf, conv=conv, E=mf.e_tot, S2=s2,
                         L2=l2, Lz=lz, Sz=szv, cfg=cfg, note=note,
                         cycles=meta.get('cycles', '—'),
                         t=meta.get('t', 0.0),
                         prec=meta.get('prec', '—')))


# ============================ STAGE 1 =========================================
def stage1_scf(ctx):
    for entry in STAGE1:
        name, kind, strategy = entry[:3]
        occ_seed = entry[3] if len(entry) > 3 else None
        mf, note = run_scf(ctx, name, kind, strategy,
                           xc=XC if kind == 'UKS' else None,
                           occ_seed=occ_seed)
        add_row(ctx, name, mf, note)


# ============================ STAGE 2 =========================================
def pick_cas(ctx, mf):
    """Subshell-completion rule: gather the valence electrons from the top
    of the occupied spectrum, split them by dominant Lowdin character, then
    complete each requested l-multiplet (CAS_MULTIPLETS) with the
    lowest-energy virtuals of the same character. Yields an SO(3)-complete
    active space (required for exact L^2 spectroscopy); heuristic in
    general."""
    e, occ = mf.mo_energy, mf.mo_occ
    W = l_weights(ctx, mf.mo_coeff)
    dom = W.argmax(0)
    occ_idx = sorted((i for i in range(ctx.mol.nao) if occ[i] > 1e-8),
                     key=lambda i: -e[i])
    val_occ, ne = [], 0.0
    for i in occ_idx:
        val_occ.append(i)
        ne += occ[i]
        if ne >= sum(NELECAS) - 1e-8:
            break
    picked = []
    for l_char, target in CAS_MULTIPLETS.items():
        li = 'spdfgh'.index(l_char)
        occ_l = [i for i in val_occ if dom[i] == li]
        virt_l = sorted((i for i in range(ctx.mol.nao)
                         if occ[i] <= 1e-8 and dom[i] == li),
                        key=lambda i: e[i])
        picked += occ_l + virt_l[:max(0, target - len(occ_l))]
    picked = sorted(picked)
    assert len(picked) == NCAS, \
        f'CAS completion gave {len(picked)} orbitals, expected {NCAS}'
    print('\n--- CAS picks (1-based) ---')
    for i in picked:
        d = dom[i]
        print(f'  MO {i+1:3d}  occ={occ[i]:.1f}  e={e[i]:+.4f}'
              f'  char={"spdfgh"[d]}  purity={W[d, i]:.2f}')
    comp = {}
    for i in picked:
        comp[dom[i]] = comp.get(dom[i], 0) + 1
    print('  CAS composition: '
          + ' '.join(f'{"spdfgh"[l]}^{comp[l]}' for l in sorted(comp)))
    return [i + 1 for i in picked]


def csf_config_histogram(ctx, mc, cutoff=CSF_CUTOFF):
    """Aggregate determinant weights c_I^2 over configuration classes,
    labeled by the dominant Lowdin character of each active MO (bit j of
    the string integer = active orbital j in the sort_mo feed order; bin()
    is MSB-first, so int(str(sa), 2) puts orbital 0 in bit 0). The bit -> l
    map is derived at runtime from the active orbitals. NB: this is a
    post-optimization MEASUREMENT — the pick's multiplet design (e.g.
    s1d5f7) is generally NOT preserved by state-specific CASSCF, which
    rotates cheap correlators (7p) into the span; expect drifted labels
    like s1d4f5p3. Determinant-level: Hund-coupled atomic terms bundle
    several determinants, so class weights are lower bounds of term
    weights."""
    act = mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    dom = np.array(list('spdfgh'))[l_weights(ctx, act).argmax(0)]
    print('  active-orbital l-map:', ' '.join(dom))
    hist, betas = defaultdict(float), set()
    for c, sa, sb in large_ci_cut(mc.ci, mc.ncas, mc.nelecas, cutoff):
        a = int(str(sa), 2)
        cnt = {l: 0 for l in 'spdfgh'}
        for j in range(mc.ncas):
            if (a >> j) & 1:
                cnt[dom[j]] += 1
        key = ' '.join(f'{l}^{cnt[l]}' for l in 'spdf' if cnt[l])
        hist[key] += c * c
        betas.add(str(sb))
    return dict(sorted(hist.items(), key=lambda kv: -kv[1])), betas


def run_casci(ctx, mf_ref, cas_idx):
    """CASCI on frozen seed-row MOs — seconds; adds a table row.
    CASCI.make_rdm1 returns the density in the non-orthogonal AO
    representation, so occupations require the generalized problem
    eigh(S dm S, b=S) (bare eigenvalues of dm are meaningless);
    eigenvectors are already AO coefficients for the Lowdin analysis."""
    mci = mcscf.CASCI(mf_ref, len(cas_idx), NELECAS)
    mci.verbose = SCF_VERBOSE
    cc, t = _counted_kernel(mci, mci.sort_mo(cas_idx))
    dm_ao = mci.make_rdm1()
    noon, natorb = _sc_eigh(ctx.S @ dm_ao @ ctx.S, ctx.S)
    noon, natorb = noon[::-1], natorb[:, ::-1]
    ss, _ = mci.fcisolver.spin_square(mci.ci, len(cas_idx), NELECAS)
    cfg = valence_config(ctx, noon, natorb)
    core_idx = [i for i, o in enumerate(mf_ref.mo_occ) if o == 2
                and i not in cas_idx]
    lm = calc_L_moments(ctx, mf_ref.mo_coeff, core_idx, cas_idx,
                        mci.ci, NELECAS, mo_occ=mf_ref.mo_occ)
    if PRINT_CSF:
        # design view: frozen seed-row span, s/d/f-only classes expected
        hist, betas = csf_config_histogram(ctx, mci)
        print('CASCI configuration histogram (frozen span):')
        for k, v in hist.items():
            print(f'   {k:16s}  sum_w = {v:.3f}')
        print('   listed total =', round(sum(hist.values()), 3),
              ' beta strings:', betas)
    mci._meta = dict(cycles='—' if not cc.n else _stage('casci', cc),
                     t=t, prec=cc.prec)
    add_row(ctx, 'CASCI', mci, f'frozen seed-row MOs; Term: {lm["term"]}',
            s2=ss, l2=lm['L2'], lz=lm['Lz'], cfg=cfg)
    return mci.e_tot


def run_casscf(ctx, mf_ref, cas_idx):
    """CASSCF solve + diagnostics. Warm start: mo0 (orbitals) and ci0 (CI
    vector) from casscf.chk (unless FORCE_COLD_CAS); the short re-run's
    stationarity certifies the stored solution. Natural orbitals:
    per-orbital occupations exist only in the frame that diagonalizes the
    1-RDM, so NOs/NOONs are computed for valence_config and the stage-3
    guess — a representation change, no new information; the equivalent raw
    spin-dm seeding (make_rdm1s) is noted in stage3_scf."""
    mc = mcscf.CASSCF(mf_ref, len(cas_idx), NELECAS)
    mc.chkfile = CAS_CHK
    mc.chk_ci = True
    mc.conv_tol = CONV_TOL
    mc.verbose = VERBOSE
    mo0, ci0, e_prev = mc.sort_mo(cas_idx), None, None
    warm = WARM and not FORCE_COLD_CAS and os.path.exists(CAS_CHK)
    if warm:
        try:
            rec = lib_chk.load(CAS_CHK, 'mcscf')
            if rec.get('mo_coeff') is not None:
                mo0 = rec['mo_coeff']; ci0 = rec.get('ci_vector')
                e_prev = rec.get('e_tot')
                print(f'  [casscf] warm seed (E_prev={e_prev})')
            else:
                print('  [casscf] chk has no orbitals -> cold start')
                warm = False
        except Exception as ex:
            print('  [casscf] chk seed failed:', ex); warm = False
    mc.max_cycle_macro = WARM_MAX_MACRO if warm else MAX_MACRO
    cc = CycleCounter(); mc.callback = cc
    t0 = time.perf_counter()
    mc.kernel(mo0) if ci0 is None else mc.kernel(mo0, ci0)
    t_cas = time.perf_counter() - t0
    if e_prev is not None:
        d = mc.e_tot - e_prev
        print(f'  [casscf] stationarity: E - E_chk = {d:.2e}'
              + ('' if abs(d) < 1e-7 else '  <- raise WARM_MAX_MACRO'))

    # Natural orbitals: a representation change only (see docstring)
    noon, natorb = addons.make_natural_orbitals(mc)
    ss, _ = addons.spin_square(mc)
    cfg = valence_config(ctx, noon, natorb)
    # (4) sort_mo/canonicalization reorder mc.mo_coeff freely: locate the
    # relaxed core/active columns by overlap with the seed-row columns
    core0 = [i for i, o in enumerate(mf_ref.mo_occ) if o == 2
             and i not in cas_idx]
    ov = mf_ref.mo_coeff[:, core0 + cas_idx].conj().T @ ctx.S @ mc.mo_coeff
    assigned, used = {}, set()
    for i, c0 in enumerate(core0 + cas_idx):
        j = int(np.argmax(np.abs(ov[i])))
        while j in used:
            ov[i, j] = 0.0
            j = int(np.argmax(np.abs(ov[i])))
        used.add(j)
        assigned[c0] = j
    core_idx = [assigned[c] for c in core0]
    act_idx = [assigned[c] for c in cas_idx]
    lm = calc_L_moments(ctx, mc.mo_coeff, core_idx, act_idx, mc.ci,
                        NELECAS, mo_occ=None)
    print(f'\nE(CASSCF) = {mc.e_tot:.8f}   <S^2> = {ss:.4f}')
    print('CASSCF NOONs:', np.round(noon[::-1], 3))

    note = f'CAS({sum(NELECAS)},{len(cas_idx)}) from seed-row MOs'
    if warm and e_prev is not None:
        note += f'; dE_vs_chk {abs(mc.e_tot - e_prev):.1e}'
    if PRINT_CSF:
        try:
            hist, betas = csf_config_histogram(ctx, mc)
            print('configuration histogram (alpha classes, |c| > cutoff):')
            for k, v in hist.items():
                print(f'   {k:16s}  sum_w = {v:.3f}')
            print('   listed total =', round(sum(hist.values()), 3),
                  ' beta strings:', betas)
        except Exception as ex:
            print('  (histogram unavailable:', ex, ')')
    mc._meta = dict(cycles=_stage('casscf', cc), t=t_cas, prec=cc.prec)
    add_row(ctx, 'CASSCF', mc, note, s2=ss, l2=lm['L2'], lz=lm['Lz'], cfg=cfg)
    ctx.mc, ctx.noon, ctx.natorb = mc, noon, natorb


def stage2_cas(ctx):
    """Declarative stage 2 (STAGE2_SEED): the seed row supplies MOs
    (fail-fast), the CAS is picked once (subshell completion), then
    CASCI pre-flight row + CASSCF."""
    ref = next((r for r in ctx.rows if slug(r['name']) == STAGE2_SEED), None)
    if ref is None or not ref['conv']:
        raise RuntimeError(f'STAGE2 seed row {STAGE2_SEED!r} missing or not '
                           'converged — fix stage 1 first.')
    ctx.mf_ref = ref['mf']
    if CAS_MANUAL is not None:
        assert len(set(CAS_MANUAL)) == NCAS
        cas_idx = list(CAS_MANUAL)
    else:
        cas_idx = pick_cas(ctx, ref['mf'])
    ctx.cas_idx = cas_idx
    print('\nCAS candidate (1-based):', cas_idx)
    run_casci(ctx, ref['mf'], cas_idx)
    print(f"\n** CASSCF reference: {ref['name']}  "
          f"E={ref['mf'].e_tot:.8f} **")
    run_casscf(ctx, ref['mf'], cas_idx)


# ============================ ANGULAR MOMENTUM ================================
# (the evaluator lives in lmom.expectation; the theorem and its premise
#  certificates are documented there. Premises: span/core l-invariance,
#  l-purity, core closure, core<->act disjointness, spherical H.)
def calc_L_moments(ctx, mo_basis, core_idx, act_idx, ci, nelecas,
                   mo_occ=None):
    """mo_basis: ORIGINAL canonical MO columns (never a sort_mo-reordered
    copy); core_idx/act_idx: explicit doubly-occupied/active columns in
    that basis; mo_occ: occupations of mo_basis (None for CASSCF-relaxed
    columns, where the core is structurally doubly occupied)."""
    lm = L2Lz_expectation_dm(ctx.mol, mo_basis, act_idx, ci, nelecas,
                    core_cols=core_idx, mo_occ=mo_occ)
    print(f'  [L^2] <L^2> = {lm["L2"]:.6f}  <Lz> = {lm["Lz"]:+.6f}  '
          f'L_eff = {lm["L_eff"]:.3f} Term: {lm["term"]}  '
          f'ladder closure = {lm["closure"]:.1e}')
    n_el = 2 * len(core_idx) + sum(nelecas)
    if abs(lm['N2'] - n_el ** 2) > 1e-6:
        print(f'  [L^2] WARNING: <N^2> = {lm["N2"]:.4f} != {n_el**2} '
              '— check the core/active columns')
    return lm


# ============================ SPECTROSCOPY ====================================
def spectroscopy(ctx):
    """Multi-root CASCI on the FROZEN seed-row MOs (frozen: the scan stays
    on a fixed, preferably multiplet-complete span; CASSCF relaxation is
    free to rotate the span out of l-invariance). Per root: E, dE (eV,
    cm^-1) and COMPUTED <S^2> — never hardcoded, so leakage of other-spin
    states into a sector is visible. L^2 labels: pending the full-space
    L^2 implementation."""
    if not (SPECTROSCOPY and ROOT_SECTORS):
        return
    mf_ref, cas_idx = ctx.mf_ref, ctx.cas_idx
    ncas = len(cas_idx)
    print('\n' + '=' * 25 + ' SPECTROSCOPIC SCAN (frozen seed-row MOs) '
          + '=' * 25)
    allroots = []
    for nelecas, nroots in ROOT_SECTORS:
        mci = mcscf.CASCI(mf_ref, ncas, nelecas)
        mci.fcisolver.nroots = nroots
        mci.verbose = 0
        mci.kernel(mci.sort_mo(cas_idx))
        for k in range(nroots):
            ss, _ = mci.fcisolver.spin_square(mci.ci[k], ncas, nelecas)
            allroots.append((nelecas, k, mci.e_tot[k], ss))
    e0 = min(r[2] for r in allroots)
    for nelecas, nroots in ROOT_SECTORS:
        roots = [r for r in allroots if r[0] == nelecas]
        na, nb = nelecas
        print(f'\n--- sector (na, nb) = {nelecas}  ({nroots} roots) ---')
        print(f"{'root':<5} {'E, Eh':>16} {'dE, eV':>10} {'dE, cm^-1':>12} "
              f"{'<S^2>':>8}")
        print('-' * 55)
        for _, k, e, ss in roots:
            print(f'{k:<5} {e:16.8f} {(e - e0)*HARTREE2EV:+10.3f} '
                  f'{(e - e0)*HARTREE2CM:+12.1f} {ss:8.3f}')
    print('  (L^2 per root: pending the full-space L^2 implementation)')


# ============================ STAGE 3 =========================================
def spin_occs_from_noons(noon, na, nb):
    """NOON ranking -> binary (alpha, beta) occupations for the SCF guess
    (a guess convention only — the kernel re-derives occupations): beta
    fills the nb largest NOONs, alpha the na largest."""
    idx = np.argsort(-noon)
    oa = np.zeros(len(noon)); oa[idx[:na]] = 1.0
    ob = np.zeros(len(noon)); ob[idx[:nb]] = 1.0
    return oa, ob


def stage3_scf(ctx):
    """SCF experiments after stage 2 (STAGE3; not necessarily restarts).
    Seeds: None = cold start; 'casscf' = natural orbitals + NOON-ranked
    binary occupations; a slug of an earlier row = that row's converged dm.
    The NOON step is a guess convention — only dm0 matters for the SCF
    trajectory; the equivalent raw density, dm0 = ctx.mc.make_rdm1s(), is
    the documented alternative."""
    mo2 = np.asarray((ctx.natorb, ctx.natorb))
    occ2 = np.asarray(spin_occs_from_noons(ctx.noon, ctx.na, ctx.nb))
    for name, kind, seed, strategy in STAGE3:
        xc = XC if kind == 'UKS' else None
        if seed is None:
            dm0 = None                        # cold start, default guess
        elif seed == 'casscf':
            if kind == 'ROHF':
                # native ROHF dm (dma, dmb) built explicitly: do NOT route
                # through ROHF.make_rdm1(mo2, occ2) — it re-binarizes occ
                # (occ>0 / occ==2) with the (2,nmo) tuple and returns a
                # garbage-shaped density
                dm0 = np.array([
                    np.einsum('pi,i,pj->ij', mo2[s], occ2[s], mo2[s])
                    for s in range(2)])
            else:
                dm0 = make_scf(ctx, kind, xc).make_rdm1(mo2, occ2)
        else:
            row = next((r for r in ctx.rows
                        if slug(r['name']) == seed), None)
            if row is None:
                raise KeyError(f'seed row {seed!r} not found — it must run '
                               f'EARLIER in STAGE3 than {name!r}')
            dm0 = row['mf'].make_rdm1()
        mf, note = run_scf(ctx, name, kind, strategy, xc=xc, dm0=dm0,
                           occ_seed=seed)
        if kind == 'UHF':
            note += (f'; dE vs {STAGE2_SEED} '
                     f'{mf.e_tot - ctx.mf_ref.e_tot:+.2e}')
        add_row(ctx, name, mf, note)


# ============================ REPORT ==========================================
def report(ctx):
    """Final comparison table (text + results.md), same columns in both.
    No '|' inside cells — raw pipes break markdown table columns."""
    mc = ctx.mc
    hdr = (f"{'method':26} {'conv':>5} {'iterations':>26} {'tolerance':>24} "
           f"{'E, Eh':>16} {'dE,mEh':>9} {'E,eV':>13} {'dE,eV':>8} "
           f"{'<S^2>':>7} {'<Sz>':>6} {'<L^2>':>7} {'<Lz>':>6} {'match':>6}  "
           f"{'configuration':30} note")
    md = ['| Method | Conv | Iterations | Tolerance | E, Eh | ΔE, mEh | '
          'E, eV | ΔE, eV | ⟨S²⟩ | ⟨Sz⟩ | ⟨L²⟩ | ⟨Lz⟩ | Configuration | '
          'Match to ground | Note |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    print('\n' + '=' * 20 + ' final comparison ' + '=' * 20)
    print(hdr)
    for r in ctx.rows:
        conv = '✔' if r['conv'] else '✘'
        hit = '✔' if config_matches(r['cfg']) else '✘'
        dE_m = f"{(r['E'] - mc.e_tot)*1e3:+.2f}" if r['conv'] else 'n/a'
        dE_e = f"{(r['E'] - mc.e_tot)*HARTREE2EV:+.3f}" if r['conv'] else 'n/a'
        e_ev = f"{r['E'] * HARTREE2EV:.4f}"
        s2s = 'n/a' if r['S2'] is None or r['S2'] != r['S2'] else f"{r['S2']:.3f}"
        szs = 'n/a' if r['Sz'] is None else f"{r['Sz']:.2f}"
        l2s = 'n/a' if r['L2'] is None or r['L2'] != r['L2'] else f"{r['L2']:.1f}"
        lzs = 'n/a' if r['Lz'] is None or r['Lz'] != r['Lz'] else f"{r['Lz']:.2f}"
        print(f"{r['name']:26} {conv:>5} {r['cycles']:>26} {r['prec']:>24} "
              f"{r['E']:16.8f} {dE_m:>9} {e_ev:>13} {dE_e:>8} {s2s:>7} "
              f"{szs:>6} {l2s:>7} {lzs:>6} {hit:>6}  {fmt_cfg(r['cfg']):30} {r['note']}")
        md.append(f"| {r['name']} | {conv} | {r['cycles']} | {r['prec']} | "
                  f"{r['E']:.8f} | {dE_m} | {e_ev} | {dE_e} | {s2s} | {szs} | "
                  f"{l2s} | {lzs} | {fmt_cfg(r['cfg'])} | {hit} | {r['note']} |")
    with open('results.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\nmarkdown table written to results.md')
    print('  * = a SMEAR stage in the strategy (see note); dE vs CASSCF is')
    print('  comparable only among rows of the same Hamiltonian family.')


# ============================ MAIN ============================================
def main(ctx=None):
    if ctx is None:
        ctx = build_system()
    stage1_scf(ctx)
    stage2_cas(ctx)
    spectroscopy(ctx)
    stage3_scf(ctx)
    report(ctx)
    return ctx
