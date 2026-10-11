#!/usr/bin/env python
"""
carbon_L2_diag2.py — decisive diagnostics for the <L^2> evaluator.

Hypothesis under test (leakage mechanism): the occupied 2p MOs are radial
mixtures of both contracted p shells; l is radial-preserving and maps them
partly into VIRTUAL p MOs, so one-body products over only the (core+active)
MO slice lose the leakage (~0.079 per gate D). Full products over ALL MO
columns restore exactness. Differential prediction:
  STO-3G (one contracted p shell, no virtual p): sliced == full == 2.000000
  cc-pvdz (two p shells, 3 virtual p MOs):       sliced ~1.92, full = 2.000000
If STO-3C gate D FAILS instead -> integral/component defect -> pivot.

Battery:
  [S1] STO-3G C: MO p-block lz eigenvalues (must be {-1, 0, +1} exact)
  [S2] STO-3G C: AO SO(3) algebra [Lx,S^-1,Ly] - iLz = 0
  [S3] STO-3G gate B CAS(2,3) and gate D C+ CAS(1,3) with SLICED products
       (current carbon_L2 code): must be exactly 2.000000
  [S4] STO-3G C FCI full-space reference: <L^2> = 2.000000
  [C1] cc-pvdz: p-block eigen/diagonal of SLICED product sum_k L5[k]@L5[k]
       vs FULL product (sum_k L[k]@L[k])[:5,:5]
  [C2] cc-pvdz: per-component leakage ||P_virt L_k P_occ-p|| and
       s-block cleanliness ||L_k rows/cols on 1s/2s||
  [C3] cc-pvdz gate D: ladder table (Lz2, LpLm, LmLp, closure) sliced vs full
  [C4] cc-pvdz gate D: full-product one-body (expect 2.000000); dm2 == 0
  [C5] cc-pvdz C FCI full-space reference: scan all (pairing, sign) combos;
       combos giving exactly 2.000000 are the valid two-body conventions
  [C6] cc-pvdz gate B/C ladder closure under the pinned convention
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from carbon_L2 import (get_orbital_angular_momentum, expand_product_ci,
                       PAIRING, SIGN)

np.set_printoptions(precision=8, suppress=True, linewidth=130)

PAIRINGS = ('pqrs', 'prqs', 'psrq', 'prsq', 'pqsr', 'psqr')


def L_mo(mol, mo):
    return get_orbital_angular_momentum(mol, mo)


def two(name, A, B, dm2):
    return np.einsum('pq,rs,' + name + '->', A, B, dm2).real


def one_sliced(L, k, dm1, n):
    return np.einsum('ps,ps->', L[k][:n, :n] @ L[k][:n, :n], dm1).real


def one_full(L, k, dm1, n):
    return np.einsum('ps,ps->', (L[k] @ L[k])[:n, :n], dm1).real


def cross_full(L, kA, sA, kB, sB, dm1, n):
    A = L[kA] * sA
    B = L[kB] * sB
    return np.einsum('ps,ps->', (A @ B)[:n, :n], dm1).real


def cross_sliced(L, kA, sA, kB, sB, dm1, n):
    A = L[kA][:n, :n] * sA
    B = L[kB][:n, :n] * sB
    return np.einsum('ps,ps->', A @ B, dm1).real


def ladder(L, dm1, dm2, n, pairing, sign, mode):
    f = one_full if mode == 'full' else one_sliced
    cf = cross_full if mode == 'full' else cross_sliced
    lz, lx, ly = L[2], L[0], L[1]
    lp, lmv = lx + 1j * ly, lx - 1j * ly
    r = {}
    r['Lz2'] = f(L, 2, dm1, n) + sign * two(pairing, lz[:n, :n], lz[:n, :n], dm2)
    r['LpLm'] = (cf(L, 0, 1, 1, -1, dm1, n)
                 + sign * two(pairing, lp[:n, :n], lmv[:n, :n], dm2))
    r['LmLp'] = (cf(L, 0, -1, 1, 1, dm1, n)
                 + sign * two(pairing, lmv[:n, :n], lp[:n, :n], dm2))
    r['L2'] = sum(f(L, k, dm1, n) for k in range(3)) + sign * sum(
        two(pairing, L[k][:n, :n], L[k][:n, :n], dm2) for k in range(3))
    r['ladder'] = r['Lz2'] + 0.5 * (r['LpLm'] + r['LmLp'])
    r['closure'] = abs(r['ladder'] - r['L2'])
    return r


def gate_setup(mf, ncas, nelecas, act):
    mci = mcscf.CASCI(mf, ncas, nelecas)
    mci.kernel(mci.sort_mo(act))
    return mci


def main():
    # ============================ STO-3G battery =========================
    print('=' * 30, 'STO-3G battery', '=' * 30)
    mol_s = gto.M(atom='C 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf_s = scf.ROHF(mol_s); mf_s.kernel()
    nmo_s = mol_s.nao
    print(f'STO-3G: nao = {nmo_s} (1s, 2s, 2p x3)')
    L_s = L_mo(mol_s, mf_s.mo_coeff)

    lz_p = mf_s.mo_coeff[:, 2:].conj().T @ (L_s[2] @ mf_s.mo_coeff[:, 2:])
    ev = np.sort_complex(np.linalg.eigvals(lz_p)).real
    print(f"[S1] p-block lz eigenvalues: {np.round(ev, 10)}  (want -1, 0, +1)")

    S = mol_s.intor('int1e_ovlp'); Sinv = np.linalg.inv(S)
    L_ao = L_mo(mol_s, np.eye(nmo_s))
    comm = L_ao[0] @ Sinv @ L_ao[1] - L_ao[1] @ Sinv @ L_ao[0] - 1j * L_ao[2]
    print(f"[S2] AO algebra max|[Lx,S^-1,Ly]-iLz| = {np.abs(comm).max():.3e}")

    # gate B STO-3G: whole space is 5 MOs; sliced == full by construction
    mci_s = gate_setup(mf_s, 3, (2, 0), [2, 3, 4])
    ci_s = expand_product_ci(2, 3, mci_s.ci, (2, 0))
    dm1s, dm2s = fci.direct_spin1.make_rdm12(ci_s, 5, (4, 2))
    L2_s = sum(one_sliced(L_s, k, dm1s, 5) for k in range(3)) + SIGN * sum(
        two(PAIRING, L_s[k][:5, :5], L_s[k][:5, :5], dm2s) for k in range(3))
    print(f"[S3a] STO-3G gate B CAS(2,3) sliced-evaluator <L^2> = {L2_s:+.8f}"
          f"  (want 2.000000)")

    mol_sc = gto.M(atom='C 0 0 0', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf_sc = scf.ROHF(mol_sc); mf_sc.kernel()
    mci_sc = gate_setup(mf_sc, 3, (1, 0), [2, 3, 4])
    ci_sc = expand_product_ci(2, 3, mci_sc.ci, (1, 0))
    dm1sc, dm2sc = fci.direct_spin1.make_rdm12(ci_sc, 5, (3, 2))
    L2_sc = sum(one_sliced(L_mo(mol_sc, mf_sc.mo_coeff), k, dm1sc, 5)
                for k in range(3))
    print(f"[S3b] STO-3G gate D C+ CAS(1,3) sliced-evaluator <L^2> = {L2_sc:+.8f}"
          f"  (want 2.000000; dm2 max = {np.abs(dm2sc).max():.1e})")

    cis = fci.FCI(mf_s); e_s, ci_fci_s = cis.kernel()
    na, nb = (mol_s.spin + mol_s.nelectron) // 2, (mol_s.nelectron - mol_s.spin) // 2
    dm1f, dm2f = fci.direct_spin1.make_rdm12(ci_fci_s, nmo_s, (na, nb))
    L2_fci = sum(one_full(L_s, k, dm1f, nmo_s) for k in range(3)) + SIGN * sum(
        two(PAIRING, L_s[k], L_s[k], dm2f) for k in range(3))
    print(f"[S4] STO-3G FCI <L^2> = {L2_fci:+.8f} (E = {e_s:+.6f}; want 2.000000;"
          f" <N> = {np.trace(dm1f).real:.6f})")

    # ============================ cc-pvdz battery ========================
    print('\n' + '=' * 30, 'cc-pvdz battery', '=' * 30)
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    nmo = mol.nao
    print(f'cc-pvdz: nao = {nmo}; occ MOs: 1s, 2s, 2p x3 -> slice n = 5')
    L = L_mo(mol, mf.mo_coeff)

    M_sliced = sum(L[k][:5, :5] @ L[k][:5, :5] for k in range(3))
    M_full = sum((L[k] @ L[k])[:5, :5] for k in range(3))
    ev_s = np.linalg.eigvalsh(M_sliced[2:5, 2:5])
    ev_f = np.linalg.eigvalsh(M_full[2:5, 2:5])
    print(f"[C1] occ-p block eigen of sliced product: {np.round(ev_s, 6)}")
    print(f"     occ-p block eigen of full   product: {np.round(ev_f, 8)}"
          "  (want 2.000000)")

    print('[C2] per-component leakage ||P_virt L_k P_occ-p||_F (k = x,y,z):')
    leak = [np.linalg.norm(L[k][5:, 2:5]) for k in range(3)]
    for k, name in enumerate('xyz'):
        print(f'     L_{name}: {leak[k]:.6f}   (squared sum -> expected deficit)')
    print(f"     total leak^2 = {sum(l**2 for l in leak):.6f}  "
          f"(gate D deficit was 2 - 1.9213 = {2 - 1.9213:.4f})")
    s_clean = max(np.abs(L[k][:2, :]).max() for k in range(3))
    print(f"     s-block (1s/2s) L max element = {s_clean:.1e}  (want ~0)")

    # gate D cc-pvdz
    mol_c = gto.M(atom='C 0 0 0', basis='cc-pvdz', charge=1, spin=1, verbose=0)
    mf_c = scf.ROHF(mol_c); mf_c.kernel()
    mci_c = gate_setup(mf_c, 3, (1, 0), [2, 3, 4])
    ci_c = expand_product_ci(2, 3, mci_c.ci, (1, 0))
    dm1c, dm2c = fci.direct_spin1.make_rdm12(ci_c, 5, (3, 2))
    L_c = L_mo(mol_c, mf_c.mo_coeff)
    print(f"[C3] gate D ladder (dm2 max = {np.abs(dm2c).max():.1e}):")
    for mode in ('sliced', 'full'):
        r = ladder(L_c, dm1c, dm2c, 5, PAIRING, SIGN, mode)
        print(f"     {mode:7s}: Lz2 = {r['Lz2']:+.6f}  LpLm = {r['LpLm']:+.6f}  "
              f"LmLp = {r['LmLp']:+.6f}  <L^2> = {r['L2']:+.6f}  "
              f"ladder = {r['ladder']:+.6f}  closure = {r['closure']:.2e}")
    L2_full_D = sum(one_full(L_c, k, dm1c, 5) for k in range(3))
    print(f"[C4] gate D full-product one-body <L^2> = {L2_full_D:+.8f}"
          f"  (want 2.000000)")

    # FCI refs + convention scan
    print('[C5] FCI full-space scan (pairing, sign) -> <L^2> (want 2.000000):')
    for basis, spin, chg, tag in (('cc-pvdz', 2, 0, 'C'), ('cc-pvdz', 1, 1, 'C+')):
        molf = gto.M(atom='C 0 0 0', basis=basis, spin=spin, charge=chg, verbose=0)
        mff = scf.ROHF(molf); mff.kernel()
        cisf = fci.FCI(mff); ef, cif = cisf.kernel()
        n = molf.nao
        naf = (molf.spin + molf.nelectron) // 2
        nbf = (molf.nelectron - molf.spin) // 2
        dm1, dm2 = fci.direct_spin1.make_rdm12(cif, n, (naf, nbf))
        Lf = L_mo(molf, mff.mo_coeff)
        good = []
        for pairing in PAIRINGS:
            for sign in (-1, 1):
                v = sum(one_full(Lf, k, dm1, n) for k in range(3)) + sign * sum(
                    two(pairing, Lf[k], Lf[k], dm2) for k in range(3))
                mark = ' <== exact' if abs(v - 2) < 1e-6 else ''
                good.append((abs(v - 2), pairing, sign))
                print(f"     {tag} {pairing} sign{sign:+d}: {v:+.8f}{mark}")
        good.sort()
        print(f"     {tag} FCI: best = {good[0][1]} sign{good[0][2]:+d} "
              f"(err {good[0][0]:.2e}); N2 check: {np.trace(dm1).real + good[0][2] * two(good[0][1], np.eye(n), np.eye(n), dm2):+.6f} (want {molf.nelectron ** 2})")

    # gate B/C ladder under pinned convention
    print(f"[C6] gate B/C ladder (pinned {PAIRING}, sign{SIGN:+d}):")
    mci_b = gate_setup(mf, 3, (2, 0), [2, 3, 4])
    ci_b = expand_product_ci(2, 3, mci_b.ci, (2, 0))
    dm1b, dm2b = fci.direct_spin1.make_rdm12(ci_b, 5, (4, 2))
    for mode in ('sliced', 'full'):
        r = ladder(L, dm1b, dm2b, 5, PAIRING, SIGN, mode)
        print(f"     B {mode:7s}: <L^2> = {r['L2']:+.6f}  ladder = {r['ladder']:+.6f}"
              f"  closure = {r['closure']:.2e}")
    mci_c22 = gate_setup(mf, 2, (2, 0), [2, 3])
    ci_c22 = expand_product_ci(2, 2, mci_c22.ci, (2, 0))
    dm1c22, dm2c22 = fci.direct_spin1.make_rdm12(ci_c22, 4, (4, 2))
    r22 = ladder(L, dm1c22, dm2c22, 4, PAIRING, SIGN, 'full')
    print(f"     C CAS(2,2) full : <L^2> = {r22['L2']:+.6f}  ladder = {r22['ladder']:+.6f}"
          f"  closure = {r22['closure']:.2e}  (must differ from 2)")

    # verdict
    print('\n================ VERDICT ===============')
    ok_s = abs(L2_sc - 2) < 1e-6 and abs(L2_s - 2) < 1e-6
    leak_evid = abs(ev_f.mean() - 2) < 1e-6 and ev_s.mean() < 1.99
    if ok_s and leak_evid:
        print('  LEAKAGE MECHANISM CONFIRMED: STO-3G control exact, cc-pvdz')
        print('  sliced < 2 / full = 2.000000 -> fix = full products.')
    elif not ok_s:
        print('  STO-3G CONTROL FAILED -> integral/component defect;')
        print('  inspect [S1]/[S2]/[C3] per-channel table for the culprit.')
    else:
        print('  INCONCLUSIVE - inspect tables above.')
    print('========================================\n')


if __name__ == '__main__':
    main()
