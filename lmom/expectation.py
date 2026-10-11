"""THE EXPECTATION APPROACH — <L^2>, <Lz>, <Lz^2> of a single state via
dm contraction (no operator matrix; eigenvalues live in lmom.moments).

THE THEOREM. For an l-invariant active span and a closed l-pure core:
(on a NON-invariant span the same formulas return the MIXING value
 sum_i w_i L_i(L_i+1) — a composition readout, not a term label; see
 README.md section 0.4)
    [P H P, L^2_span] = 0  exactly on the CASCI space, because for
    phi, chi in the span:  <phi|[P H P, L^2]|chi> = <phi|[H, L^2]|chi> = 0
    (P phi = phi, L^2 phi stays in the span). H and L^2 are therefore
    simultaneously diagonalizable: CASCI roots decompose into term states
    with exact integer L(L+1) = <L^2> (resolvable via the
moment machinery in lmom.moments (moment_matrix + eigvals).
    Active-only evaluation is exact: <L^2> = <L_a^2>, since
    <L_c^2> = 0 (closed l-pure core) and <L_c L_a> = 0 (the product
    state |Phi_c> x |Psi_a> factorizes; <Phi_c|L_k,c|Phi_c> = 0 for real
    MOs: l_k = i A_k with A_k real antisymmetric).
    For any REAL CI: <Lz> = i Psi^T A Psi = 0 identically (A
    antisymmetric) — <Lz> = 0 on every row carries no information;
    <Lz^2> is the informative partner.

THE PREMISES -> THE CERTIFICATES (all checked numerically per call):
  1. active span invariant under l ....... ||L[outside, act]|| ~ 0
  2. active MOs l-pure ................... <l^2> ~ l(l+1) per MO
  3. core: closed shells of l-pure MOs ... per-MO + ||L[outside, core]||
  4. core<->active disjointness .......... ||L[core, act]|| ~ 0
  5. core closure ........................ mo_occ == 2
  6. true H spherically symmetric ........ [H, L^2] = 0 (exact for
     all-electron atoms and spin-free AREP ECPs; BROKEN by spin-orbit
     ECPs; post-hoc check: (2L+1)-fold M_L multiplet completeness,
     see the term-
decomposition recipe in lmom/moments.py: moment_matrix over the
root set -> eigvalsh -> integer L(L+1) ladder

CAUSES OF UNEXPECTED <L^2> AND THE CURES (each verified on carbon)
-------------------------------------------------------------------
  span insufficiency (the active space misses complete shells, e.g.
    {px,py} without pz) -> the L-leakage, [H_CAS, L^2] != 0, the roots
    are term-mixtures.
    CURE: complete shells in the CAS (the subshell completion;
    CAS(6,16) = +7p^3 for U-32e).  DETECTOR: the span certificate.
  reference smearing (a Hund-broken open-shell ROHF has many
    near-degenerate solutions; the canonical MOs smear across
    l-character run-to-run) -> non-l-invariant spans, unstable picks.
    CURE: spherical_reference (the m-averaged density; pure-l MOs).
    DETECTOR: the core certificate + the run-to-run spread.
  optimizer drift (a state-specific CASSCF has no symmetry constraint;
    the orbitals drift out of pure-l) -> flagged spans, L-mixed roots.
    CURE: state-averaged CASSCF over the term's M_L-multiplet (the
    averaged density is rotation-invariant; the Schur argument then
    keeps the optimized orbitals pure-l).  DETECTOR: the certificates.
  degenerate-block mixing (LAPACK returns arbitrary combinations inside
    exactly-degenerate multi-L levels) -> the per-root <L^2> is a
    block-average; the spectrum over the COMPLETE block is still the
    exact integer ladder (the trace is invariant).
    CURE: decompose_states_op (the L^2-diagonalization inside the
    block; the energy is untouched — the rotation changes nothing
    physically).  DETECTOR: the multiplet certificate.

NOTE ON SOC: all of the above assumes [H, L^2] = 0 and real CI vectors
(spin-free H; <Lz> = i Psi^T A Psi = 0 identically for real Psi, A real
antisymmetric — a realness sanity check, NOT a term test). With
spin-orbit coupling: only J^2, Jz are conserved; <Lz> != 0 becomes
possible and the LS term analysis must be replaced by the J-based one
(J^2 = L^2 + S^2 + 2 L.S — the operator machinery generalizes; the
spin-flip bilinears are the required extension).

CONVENTIONS (pinned by exact measurements on carbon):
  * COMPLETE INTERMEDIATES in the one-body piece (radial-mixture leakage
    into virtual same-l shells otherwise lost);
  * two-body dm2 layout <a+_p a_q a+_r a_s> - delta_qr dm1[ps], i.e.
    pairing 'pqrs', sign +1 (pyscf native; verified to ~1e-13).
"""
import numpy as np
from pyscf import fci

from lmom.angular import get_L, l2_op

PAIRING = 'pqrs'   # dm2 index order, pinned by exact measurement
SIGN = +1          # two-body sign
TERM_LETTERS = 'SPDFGHIKLMNOQRTUVWXYZ'


# ------------------------------------------------------------------ certificates
def span_certificate(L, act_cols, l_max=5):
    """Is the active span invariant under l itself? Two checks:
    (a) l^2-block structure: columns grouped by nearest l(l+1); each
        group's l^2 block = l(l+1) x I (every MO pure-l). NOTE (a) alone
        is trivially true for ANY subspace of an l shell.
    (b) l-INVARIANCE: ||L[k][outside, act]|| ~ 0 — l must not map the
        span into its complement. This is the true closure condition: a
        pick mixing radial sets passes (a) and fails (b), and non-integer
        <L^2> is then legitimate span-mixing. With (a)+(b): every CASCI
        root in the span can be chosen an exact L-term; non-integer
        values come only from degenerate-block mixing (resolvable by
        spectrum.resolve_terms)."""
    nmo = L.shape[1]
    l2 = l2_op(L)
    d = l2[np.ix_(act_cols, act_cols)]
    diag = np.diag(d).real
    targets = np.array([l * (l + 1) for l in range(l_max + 1)])
    groups = {}
    for col, v in zip(act_cols, diag):
        l = int(np.argmin(np.abs(targets - v)))
        groups.setdefault(l, []).append(col)
    closure_err = 0.0
    for l, cols in groups.items():
        ev = np.linalg.eigvalsh(l2[np.ix_(cols, cols)])
        closure_err = max(closure_err, float(np.abs(ev - l * (l + 1)).max()))
    outside = [i for i in range(nmo) if i not in act_cols]
    invar_max, invar_fro = 0.0, 0.0
    for k in range(3):
        blk = L[k][np.ix_(outside, act_cols)]
        invar_max = max(invar_max, float(np.abs(blk).max()))
        invar_fro = max(invar_fro, float(np.linalg.norm(blk)))
    return dict(diag=diag, groups=groups, closure_err=closure_err,
                invar_err=invar_max, invar_fro=invar_fro,
                l_closed=bool(closure_err < 1e-6 and invar_max < 1e-6))


def core_certificate(L, mo_occ, core_cols, act_cols, l_max=5):
    """Theorem conditions on the core: (1) closed shells of l-pure
    orbitals forming l-invariant subspaces (a closed shell of MIXED-l
    orbitals is NOT an L=0 state); (2) core<->active disjointness (fails
    first when a relaxed active MO drifts into core character);
    (3) doubly occupied core columns."""
    nmo = L.shape[1]
    l2 = l2_op(L)
    diag = np.diag(l2[np.ix_(core_cols, core_cols)]).real
    targets = np.array([l * (l + 1) for l in range(l_max + 1)])
    purity_err = max(float(np.min(np.abs(targets - v))) for v in diag)
    outside = [i for i in range(nmo) if i not in core_cols]
    core_inv = max(float(np.linalg.norm(L[k][np.ix_(outside, core_cols)]))
                   for k in range(3))
    disjoint = max(float(np.linalg.norm(L[k][np.ix_(core_cols, act_cols)]))
                   for k in range(3))
    closed = True if mo_occ is None else \
        bool(all(mo_occ[i] == 2 for i in core_cols))  # None: CAS core is
        # structurally doubly occupied in every determinant
    ok = bool(purity_err < 0.05 and core_inv < 1e-6 and closed)
    return dict(purity_err=purity_err, core_inv=core_inv, disjoint=disjoint,
                closed=closed, ok=ok)


# ------------------------------------------------------------------ evaluator
def L2Lz_expectation_dm(mol, mo_coeff, act_cols, ci, nelecas, core_cols=None,
                        mo_occ=None):
    """Production evaluator (active-only; exact when the certificates
    pass). Returns <Lz>, <Lz^2>, <L^2>, ladder pieces, L_eff, term letter,
    closure, <N_a^2>, and the certificates."""
    L = get_L(mol, mo_coeff)
    La = L[:, act_cols][:, :, act_cols]
    ncas = len(act_cols)
    d1a, d1b = fci.direct_spin1.make_rdm1s(ci, ncas, nelecas)
    d1 = d1a + d1b
    # pyscf's spin-summed 2-RDM, native <p^+ q r^+ s> layout (verified)
    d2 = fci.direct_spin1.make_rdm12(ci, ncas, nelecas)[1]
    ix = np.ix_(act_cols, act_cols)

    def one(A, B):     # complete intermediates, active restriction after
        return np.einsum('ps,ps->', (A @ B)[ix], d1).real

    def two(X, Y):
        return SIGN * np.einsum('pq,rs,' + PAIRING + '->', X, Y, d2).real

    lx, ly, lz = L[0], L[1], L[2]
    lp, lmv = lx + 1j * ly, lx - 1j * ly
    eye = np.eye(ncas)
    r = {}
    r['N2'] = np.einsum('ps,ps->', eye, d1).real + two(eye, eye)
    r['Lz'] = np.einsum('pq,qp->', La[2], d1).real      # = 0 for real CI
    r['Lz2'] = one(lz, lz) + two(La[2], La[2])
    r['LpLm'] = one(lp, lmv) + two(La[0] + 1j * La[1], La[0] - 1j * La[1])
    r['LmLp'] = one(lmv, lp) + two(La[0] - 1j * La[1], La[0] + 1j * La[1])
    r['L2'] = sum(one(L[k], L[k]) + two(La[k], La[k]) for k in range(3))
    r['ladder'] = r['Lz2'] + 0.5 * (r['LpLm'] + r['LmLp'])
    r['closure'] = abs(r['ladder'] - r['L2'])
    r['L_eff'] = (-1.0 + np.sqrt(1.0 + 4.0 * max(0.0, r['L2']))) / 2.0
    li = int(round(r['L_eff']))
    r['term'] = TERM_LETTERS[li] if 0 <= li < len(TERM_LETTERS) else '??'
    r['span'] = span_certificate(L, act_cols)
    if core_cols is not None:
        r['core'] = core_certificate(L, mo_occ, core_cols, act_cols)
    return r


def L2_expectation_dm_full(mol, mo_coeff, core_cols, act_cols, ci, nelecas):
    """GATE FIXTURE ONLY: expanded-CI densities — an independent density
    source; if it ever disagreed with L2Lz_expectation_dm on an l-disjoint system,
    that would be an evaluator defect. Feasible while
    C(n,na)C(n,nb) <~ 1e9."""
    from math import comb
    cols = list(core_cols) + list(act_cols)
    n = len(cols)
    na = len(core_cols) + nelecas[0]
    nb = len(core_cols) + nelecas[1]
    if comb(n, na) * comb(n, nb) > 1e9:
        raise RuntimeError('full reference infeasible')
    from lmom.moments import _expand
    ci_full = _expand(len(act_cols), len(core_cols), ci, nelecas)
    L = get_L(mol, mo_coeff)
    Lc = L[:, cols][:, :, cols]
    dm1, dm2 = fci.direct_spin1.make_rdm12(ci_full, n, (na, nb))
    one = sum(np.einsum('ps,ps->', (L[k] @ L[k])[np.ix_(cols, cols)],
                        dm1).real for k in range(3))
    two = sum(SIGN * np.einsum('pq,rs,pqrs->', Lc[k], Lc[k], dm2).real
              for k in range(3))
    return one + two


def print_lm(tag, r):
    print(f'--- {tag} ---')
    print(f"    <L^2> = {r['L2']:+.8f}   <Lz> = {r['Lz']:+.6f}   "
          f"<Lz^2> = {r['Lz2']:+.6f}")
    print(f"    L_eff = {r['L_eff']:.4f}  Term: {r['term']}   "
          f"ladder closure = {r['closure']:.1e}   <N_a^2> = {r['N2']:.6f}")
    s = r['span']
    print(f"    span certificate: l-closed = {s['l_closed']} "
          f"(l^2 closure err {s['closure_err']:.1e}, l-invariance "
          f"{s['invar_err']:.1e});  per-MO <l^2> = "
          f"{np.round(np.sort(s['diag']), 3)}")
    if 'core' in r:
        c = r['core']
        print(f"    core certificate: purity err = {c['purity_err']:.1e}  "
              f"core l-invariance = {c['core_inv']:.1e}  "
              f"core<->act = {c['disjoint']:.1e}  closed = {c['closed']}"
              f"  -> {'OK' if c['ok'] else 'FLAGGED'}")
