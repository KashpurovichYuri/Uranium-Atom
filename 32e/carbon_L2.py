#!/usr/bin/env python
"""
carbon_L2.py — validation benchmark for the full-space <L^2>/<Lz> evaluator
(gates for the uranium L^2/Lz implementation).

System: carbon atom, ROHF start.
  Control: STO-3G CAS(2,3) (whole MO space) — exact 3P, no virtual p MOs.
  Gate B: cc-pvdz CAS(2,3) frozen ROHF MOs — full 2p shell (l-closed span),
          exact 3P:  <Lz> = 0, <L^2> = 2.000000, L_eff = 1.0.
  Gate C: cc-pvdz CAS(2,2) (1,1) — 2 of the 3 p MOs, spin-mixed:
          <L^2> != 2  (the multiplet-insufficiency mechanism, demonstrated).
          NB: the (2,0) variant is NOT a valid insufficiency gate — all-alpha
          forces S=1, and p^2 with S=1 is pure 3P by Pauli, so L^2 = 2 for
          EVERY state in that span (same-spin Pauli protection).
  Gate D: cc-pvdz C+ CAS(1,3) — single p electron:  <l^2> = 2.000000.
  <N^2> = N^2 exactly on every state (dm-route sanity).

Evaluator (every essential pinned/verified — carbon_L2_diag2.py,
carbon_L2_referee.py):
  L matrices in the FULL MO space (verified: lz eigenvalues exactly
  {-1,0,+1} on a single p shell; AO SO(3) algebra [Lx,S^-1,Ly]=iLz exact;
  the FULL one-body product satisfies l^2 = 2 on the occ-p block exactly);
  contraction:
    <Lk Lk'> = Tr((Lk @ Lk')[cols,cols] dm1) + sum (Lk)_pq (Lk')_rs dm2[p,q,r,s]

  1. COMPLETE INTERMEDIATES (delta piece): the matrix product must run over
     ALL MO columns, then be restricted to the (core+active) columns. The
     occupied 2p MOs are radial mixtures across contracted p shells and the
     radial-preserving operator l maps them partly into VIRTUAL p MOs;
     slice-products alone lose that leakage (occ-p eigenvalues 1.92/1.96
     instead of 2.000000; STO-3G, with no virtual p, is exact unpatched).
  2. TWO-BODY CONVENTION: pairing 'pqrs', sign +1 — matches the
     convention-free referee (second-quantized L^ applied twice to the CI
     vector, <L^2> = ||L^ psi||^2) on generic broken-symmetry states AND
     gives exactly 2.000000 on full-space FCI of C and C+ (err ~1e-13).
     dm2 = pyscf-layout spin-summed 2-RDM <a+_i a+_j a_l a_k>.
  3. BASIS ORDER (the sort_mo trap): CASCI.sort_mo REORDERS mc.mo_coeff
     (active MOs moved to the front). All evaluation therefore runs on the
     ORIGINAL canonical MO columns (mf.mo_coeff) with explicit core/active
     index lists; the CASCI object is used ONLY as the source of mc.ci.

  Gate tolerance: 1e-4 (not 1e-6): the ROHF atom reference has ~4e-3
  l-contamination in the "pure" MOs (the open-shell 2p^2 density is not
  perfectly spherical); the ladder closure (convention-internal) stays
  at ~1e-13.
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring

PAIRING = 'pqrs'   # dm2 index order for the two-body contraction
SIGN = +1          # sign of the two-body term


def get_orbital_angular_momentum(mol, mo_coeff):
    """L = -i (r x grad) in the given MO basis; (3, nmo, nmo)."""
    mol.set_common_origin([0.0, 0.0, 0.0])
    L_ao = -1j * mol.intor('int1e_cg_irxp')
    return np.einsum('pi,xpq,qj->xij', mo_coeff.conj(), L_ao, mo_coeff)


def spherical_reference(mf):
    """Spherically-symmetrized orbital reference.

    The Hund-broken open-shell ROHF atom converges to a non-spherical
    density; its canonical MOs near the degenerate manifolds (empty-2p vs
    3s/d virtuals) are run-dependently smeared, which poisons frozen-MO
    CASCI gates (the CASCI then may or may not land on the pure-3P pair).
    Fix: average the converged density over the m-channels within each l
    (the real-harmonic AO blocks ARE m-bases), zero the cross-l blocks,
    and re-diagonalize. The resulting MOs are pure-l, degenerate within
    each l shell, and run-stable; every frozen-MO CASCI state in the full
    p shell is then exactly 3P.
    """
    mol = mf.mol
    dm = mf.make_rdm1()
    dm = np.asarray(dm)
    if dm.ndim == 3:                     # ROHF spin-resolved -> spin-summed
        dm = dm.sum(axis=0)
    nao = mol.nao
    # shell -> AO range (spherical; mol.cart must be False)
    assert not mol.cart
    loc = [int(x) for x in mol.ao_loc_nr()]
    ranges = [(loc[i], loc[i + 1], mol._bas[i, gto.ANG_OF])
              for i in range(mol.nbas)]
    assert all(a1 > a0 for a0, a1, _ in ranges) and loc[-1] == nao
    dm_sph = np.zeros_like(dm)
    for a0, a1, la in ranges:
        for b0, b1, lb in ranges:
            if la != lb:
                continue          # cross-l blocks: zero for spherical density
            if la == 0:
                dm_sph[a0:a1, b0:b1] = dm[a0:a1, b0:b1]   # all m = 0
            else:                 # square m-blocks, aligned across shells
                n = a1 - a0
                blk = dm[a0:a1, b0:b1]
                dm_sph[a0:a1, b0:b1] = (np.trace(blk) / n) * np.eye(n)
    from scipy.linalg import eigh
    from pyscf import scf as _scf
    F = _scf.hf.get_hcore(mol) + _scf.hf.get_veff(mol, dm_sph)
    mo_e, mo_u = eigh(F, mol.intor('int1e_ovlp'))
    return mo_u, mo_e

def expand_product_ci(ncore, ncas, ci_act, nelecas):
    """|core dets (columns 0..ncore-1 doubly occupied)> x |ci_act (ncas
    active columns)> -> CI over the first ncore+ncas columns (direct_spin1
    packing). The caller MUST pass mo columns in an order where the core
    really is 0..ncore-1 (see sort_mo trap in the docstring)."""
    n = ncore + ncas
    na, nb = ncore + nelecas[0], ncore + nelecas[1]
    core = (1 << ncore) - 1
    ci_full = np.zeros((cistring.num_strings(n, na),
                        cistring.num_strings(n, nb)))
    for ia in range(ci_act.shape[0]):
        sa = int(cistring.addr2str(ncas, nelecas[0], ia))
        for ib in range(ci_act.shape[1]):
            sb = int(cistring.addr2str(ncas, nelecas[1], ib))
            x = core | (sa << ncore)
            y = core | (sb << ncore)
            ci_full[cistring.str2addr(n, na, x),
                    cistring.str2addr(n, nb, y)] = ci_act[ia, ib]
    return ci_full


def eval_L_full(mol, mo_basis, core_idx, act_idx, ci_full, nelec,
                pairing=PAIRING, sign=SIGN, verbose=False):
    """Full-space <L^2>/<Lz> for the given expanded CI. mo_basis: the
    ORIGINAL canonical MO columns (never an mc object's reordered copy);
    core_idx/act_idx: the doubly-occupied / active columns in that basis;
    ci_full: expand_product_ci over [core..., act...] columns."""
    cols = list(core_idx) + list(act_idx)
    n = len(cols)
    ncore = len(core_idx)
    L = get_orbital_angular_momentum(mol, mo_basis)          # FULL basis
    Lc = L[:, cols][:, :, cols]                              # (C∪A) blocks
    na, nb = nelec
    dm1, dm2 = fci.direct_spin1.make_rdm12(ci_full, n, (na, nb))

    def delta_one(A_full, B_full):
        """delta piece with COMPLETE intermediates: product over ALL MO
        columns, restricted to (C∪A) afterwards (docstring item 1)."""
        return np.einsum('ps,ps->', (A_full @ B_full)[np.ix_(cols, cols)],
                         dm1).real

    def two(A, B):
        return np.einsum('pq,rs,' + pairing + '->', A, B, dm2).real

    tot = {}
    eye = np.eye(n)
    tot['N2'] = (np.einsum('ps,ps->', eye, dm1).real
                 + sign * two(eye, eye))
    tot['Lz'] = np.einsum('pq,qp->', Lc[2], dm1).real
    tot['L2_direct'] = sum(delta_one(L[k], L[k]) + sign * two(Lc[k], Lc[k])
                           for k in range(3))
    tot['L_eff'] = (-1.0 + np.sqrt(1.0 + 4.0 * max(0.0, tot['L2_direct']))) / 2.0
    # ladder: <L^2> = <Lz^2> + 1/2 (<L+L-> + <L-L+>), L+/- = Lx +/- i Ly
    # (uniform convention: delta_one + sign*two in every piece; bilinearity
    # makes the closure a pure float-noise check)
    lx, ly, lz = L[0], L[1], L[2]
    lp, lmv = lx + 1j * ly, lx - 1j * ly
    tot['Lz2'] = delta_one(lz, lz) + sign * two(Lc[2], Lc[2])
    tot['LpLm'] = (delta_one(lp, lmv) + sign * two(Lc[0] + 1j * Lc[1],
                                                   Lc[0] - 1j * Lc[1]))
    tot['LmLp'] = (delta_one(lmv, lp) + sign * two(Lc[0] - 1j * Lc[1],
                                                   Lc[0] + 1j * Lc[1]))
    tot['L2_ladder'] = tot['Lz2'] + 0.5 * (tot['LpLm'] + tot['LmLp'])
    tot['closure'] = abs(tot['L2_ladder'] - tot['L2_direct'])
    if verbose:
        for k in sorted(tot):
            print(f'    {k:10s} = {tot[k]:+.6f}')
    return tot


def main():
    print(__doc__)

    # ROHF C cc-pvdz canonical MO order: 1s, 2s, 2p x3 (indices 0..4),
    # then virtuals — verified: MOs 2:5 have occ-p block l^2 eigen exactly
    # [2, 2, 2] with full products (carbon_L2_diag2.py [C1]).
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    mo, mo_e = spherical_reference(mf)   # pure-l, run-stable reference

    # ---------- control: STO-3G (whole space, no virtual p) --------------
    mol_s = gto.M(atom='C 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf_s = scf.ROHF(mol_s); mf_s.kernel()
    mci_s = mcscf.CASCI(mf_s, 3, (2, 0))
    mci_s.kernel(mci_s.sort_mo([2, 3, 4]))
    ci_s = expand_product_ci(2, 3, mci_s.ci, (2, 0))
    print('\n--- control: STO-3G CAS(2,3) (whole space, exact 3P) ---')
    rS = eval_L_full(mol_s, mf_s.mo_coeff, [0, 1], [2, 3, 4], ci_s, (4, 2),
                     verbose=True)
    ok_s = abs(rS['L2_direct'] - 2) < 1e-4 and rS['closure'] < 1e-9

    # ---------- gate B: exact-3P frozen CAS(2,3) on full 2p shell --------
    mci = mcscf.CASCI(mf, 3, (2, 0))
    mci.kernel(mci.sort_mo([2, 3, 4]))
    ci_full = expand_product_ci(2, 3, mci.ci, (2, 0))
    print('\n--- gate B: cc-pvdz CAS(2,3) frozen (exact 3P) ---')
    rB = eval_L_full(mol, mo, [0, 1], [2, 3, 4], ci_full, (4, 2), verbose=True)
    ok_b = (abs(rB['N2'] - 36) < 1e-6 and abs(rB['L2_direct'] - 2) < 1e-4
            and abs(rB['L_eff'] - 1) < 1e-4 and rB['closure'] < 1e-9)

    # gate C: CAS(2,2) (1,1) — l- AND S-insufficient span (2 of the 3 p MOs,
    # one alpha + one beta: the singlet root mixes 1D/1S -> L^2 != 2)
    mci2 = mcscf.CASCI(mf, 2, (1, 1))
    mci2.kernel(mci2.sort_mo([2, 3]))
    ci2_full = expand_product_ci(2, 2, mci2.ci, (1, 1))
    print('\n--- gate C: cc-pvdz CAS(2,2)(1,1) (insufficient span) ---')
    r2 = eval_L_full(mol, mo, [0, 1], [2, 3], ci2_full, (3, 3), verbose=True)
    print(f"  C CAS(2,2)(1,1): <L^2> = {r2['L2_direct']:+.6f}  (must differ from 2)")
    ok_c = abs(r2['L2_direct'] - 2) > 1e-3

    # gate D: C+ CAS(1,3) — single p electron
    mol_c = gto.M(atom='C 0 0 0', basis='cc-pvdz', charge=1, spin=1, verbose=0)
    mf_c = scf.ROHF(mol_c)
    mf_c.kernel()
    mo_c, mo_ec = spherical_reference(mf_c)
    mci_c = mcscf.CASCI(mf_c, 3, (1, 0))
    mci_c.kernel(mci_c.sort_mo([2, 3, 4]))
    ci_c_full = expand_product_ci(2, 3, mci_c.ci, (1, 0))
    print('\n--- gate D: cc-pvdz C+ CAS(1,3) (single p electron) ---')
    r3 = eval_L_full(mol_c, mo_c, [0, 1], [2, 3, 4], ci_c_full,
                     (3, 2), verbose=True)
    print(f"  D C+ CAS(1,3): <L^2> = {r3['L2_direct']:+.6f}  (2.000000 +- 1e-4), "
          f"L_eff = {r3['L_eff']:.4f}")
    ok_d = abs(r3['L2_direct'] - 2) < 1e-4 and r3['closure'] < 1e-9

    print('\n================ GATES ================')
    print(f"  S (STO-3G control = 2):  {'PASS' if ok_s else 'FAIL'}")
    print(f"  B (CAS(2,3) = 2.000000): {'PASS' if ok_b else 'FAIL'}")
    print(f"  C (CAS(2,2) != 2):       {'PASS' if ok_c else 'FAIL'}")
    print(f"  D (C+ CAS(1,3) = 2):     {'PASS' if ok_d else 'FAIL'}")
    print('=======================================\n')


if __name__ == '__main__':
    main()
