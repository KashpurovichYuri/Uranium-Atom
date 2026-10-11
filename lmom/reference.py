"""Reference-orbital machinery: the corrections that make frozen-MO CASCI
gates reproducible. (The spectrum approach in spectrum.py does not need
this module; the expectation evaluator benefits from it.)"""
import numpy as np

from lmom.angular import get_L, l2_op
from scipy.linalg import eigh
from pyscf import gto, scf


def spherical_reference(mf, m_separable=False, deg_tol=1e-6):
    """Pure-l, run-stable MO reference from a Hund-broken open-shell ROHF.
    The SCF converges to a non-spherical density (e.g. 2 of 3 degenerate
    p MOs occupied); its canonical MOs near degenerate manifolds are
    run-dependently smeared, which poisons frozen-MO CASCI gates. Fix:
    m-average the converged density within each l (the real-harmonic AO
    blocks ARE m-bases), zero cross-l blocks, re-diagonalize. [F, l^2] = 0
    then makes every eigenvector l-pure (Schur) and the l-shell spans
    exact and degenerate, so any CASCI state in a full l shell is an exact
    L-term, independent of LAPACK's degenerate eigenvector.

    m_separable (EXPERIMENTAL, default False): diagonalize Lz within
    each degenerate eigenvalue cluster. STATUS: FAILED in a real AO
    basis — m-eigenstates are complex (Y_lm), every real column has
    <Lz> = 0 exactly, and a real cluster rotation cannot m-diagonalize;
    the attempt also degraded orthonormality (an inconsistency between
    the cluster-Lz metric and get_L's transform — unresolved). The
    span-closure criterion stands as stated (L is radial-diagonal and
    m-mixing only, so the span needs the L-images of all its columns);
    the one-shot reference satisfies it empirically (invar ~1e-12).
    A correct route would complexify the shell manifolds, diagonalize
    Lz, and rotate back — deferred.

    Needed for reproducible frozen-MO term analysis of open-shell atoms;
    unnecessary for closed shells or whole-space checks."""
    mol = mf.mol
    dm = np.asarray(mf.make_rdm1())
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    dm_sph = m_average_density(mol, dm)
    F = scf.hf.get_hcore(mol) + scf.hf.get_veff(mol, dm_sph)
    mo_e, mo_u = eigh(F, mol.intor('int1e_ovlp'))
    if m_separable:
        mo_u = _m_separate(mol, mo_u, mo_e, deg_tol)
    return mo_u, mo_e


def _m_separate(mol, mo_u, mo_e, tol=1e-6):
    """Diagonalize Lz within each degenerate eigenvalue cluster of the
    spherical Fock. After the rotation every column is (radial-combo, m)
    with complete m-sets per cluster, so L-images provably stay in the
    shell span (L is radial-diagonal and m-mixing only). The rotation is
    unitary within the cluster: S-orthonormality and the l-structure are
    preserved."""
    L = get_L(mol, mo_u)
    lz = L[2]                          # the MO-basis Lz (Hermitian)
    order = np.argsort(mo_e)
    clusters = [[order[0]]]
    for k in order[1:]:
        if mo_e[k] - mo_e[clusters[-1][-1]] < tol:
            clusters[-1].append(k)
        else:
            clusters.append([k])
    for cl in clusters:
        if len(cl) < 2:
            continue
        cl = np.array(cl)
        sub = lz[np.ix_(cl, cl)]
        w, V = np.linalg.eigh(0.5 * (sub + sub.conj().T))
        mo_u[:, cl] = mo_u[:, cl] @ V
    return mo_u


def m_average_density(mol, dm):
    """The rotational group average of a density matrix, exact in the
    shell-block representation: every separately-outlined (2l+1) block
    -> (tr/n) x I (same-m coupling preserved), cross-l blocks zeroed
    (the group average cannot contain cross-l content)."""
    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    nao = mol.nao
    loc = [int(x) for x in mol.ao_loc_nr()]
    ranges = []
    for i in range(mol.nbas):
        a0, a1, la = loc[i], loc[i + 1], int(mol._bas[i, gto.ANG_OF])
        n = 2 * la + 1
        assert (a1 - a0) % n == 0, 'shell size not a multiple of 2l+1'
        for c in range((a1 - a0) // n):   # merged shells: per-copy blocks
            ranges.append((a0 + c * n, a0 + (c + 1) * n, la))
    assert all(a1 > a0 for a0, a1, _ in ranges) and loc[-1] == nao
    dm_sph = np.zeros_like(dm)
    for a0, a1, la in ranges:
        for b0, b1, lb in ranges:
            if la != lb:
                continue          # cross-l blocks: zero (block-diag in l)
            if la == 0:
                dm_sph[a0:a1, b0:b1] = dm[a0:a1, b0:b1]   # single m = 0
            else:
                n = a1 - a0                      # m-average within the shell
                dm_sph[a0:a1, b0:b1] = (np.trace(dm[a0:a1, b0:b1]) / n) * np.eye(n)
    return dm_sph


def force_config(mol, mf, l_occs, core_n):
    '''2-pass: detect the valence l-columns from the first ROHF, then
    re-run ROHF with the m-averaged occupations {l: occ} (the Aufbau
    sometimes converges to an excited configuration, e.g. 4p^1 instead
    of 3d^1 for Sc/sto-3g).

    OCCUPATION PINNING: mf2.get_occ is overridden to return the fixed
    occupation vector at EVERY SCF iteration, so occupations cannot
    drift during SCF (index-pinned, the same IDX philosophy as the 14e
    driver rows). MOs may rotate/reorder; the occupation pattern per
    index does not change. mf2.mo_occ is set at the end only for
    bookkeeping (active-space picks).'''
    L = get_L(mol, mf.mo_coeff)
    l2 = np.diag(l2_op(L)).real
    targets = np.array([l * (l + 1) for l in range(7)])
    l_of = np.array([int(np.argmin(np.abs(targets - v)))
                     for v in l2])
    occ = np.zeros(len(l_of))
    occ[:core_n] = 2.0
    for l, o in l_occs.items():
        cols = [i for i in range(core_n, len(l_of)) if l_of[i] == l]
        # m-averaged Hund config: spread over the first (2l+1) columns
        # (the lowest radial set of that l); higher radial ghosts stay 0
        nsh = min(2 * l + 1, len(cols))
        for i in cols[:nsh]:
            occ[i] = o / nsh
    mf2 = scf.ROHF(mol)
    mf2.get_occ = lambda mo_e=None, mo=None: occ
    mf2.kernel()
    if not mf2.converged:
        mf2 = mf2.newton()
        mf2.get_occ = lambda mo_e=None, mo=None: occ
        mf2.kernel()
    mf2.mo_occ = occ
    return mf2
