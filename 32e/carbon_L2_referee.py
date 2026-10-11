#!/usr/bin/env python
"""
carbon_L2_referee.py — convention-free referee for the <L^2> contraction.

Method: apply the SECOND-QUANTIZED one-body operator L^ = sum_k (l_k)_pq
a+_p a_q directly to the CI vector (both spin sectors), twice, and use
<L^2> = ||L^ psi||^2 (L^ is Hermitian). This involves NO dm1/dm2, NO
index-pairing conventions — the norm is invariant under any sign
convention of the fermionic moves. The referee value is the ground truth;
the (pairing, sign) scan then selects the contraction convention that
reproduces it on ALL systems simultaneously.

Systems (all with the fixed complete-intermediates one-body):
  B_STO  : STO-3G  C CAS(2,3) frozen, exact 3P       -> referee 2.000000
  B_pvdz : cc-pvdz C CAS(2,3) frozen, exact 3P       -> 2 +- 1e-4 (ROHF imp.)
  D_STO  : STO-3G  C+ CAS(1,3)                      -> 2.000000
  D_pvdz : cc-pvdz C+ CAS(1,3)                      -> 2 +- 1e-4
  FCI_C  : cc-pvdz C full-space FCI 3P              -> 2.000000 exact
  FCI_Cp : cc-pvdz C+ full-space FCI                -> 2.000000 exact
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from carbon_L2 import get_orbital_angular_momentum, expand_product_ci

PAIRINGS = ('pqrs', 'prqs', 'psrq', 'prsq', 'pqsr', 'psqr')


def one_body_T(n, nocc, lk):
    """Determinant-basis matrix of the one-body operator sum_pq lk[p,q] a+_p a_q
    (any consistent fermionic sign convention; the norm is convention-free)."""
    num = cistring.num_strings(n, nocc)
    T = np.zeros((num, num), dtype=complex)
    for ia in range(num):
        s = int(cistring.addr2str(n, nocc, ia))
        occ = [q for q in range(n) if (s >> q) & 1]
        for q in occ:
            sign1 = -1.0 if bin(s & ((1 << q) - 1)).count('1') % 2 else 1.0
            s1 = s ^ (1 << q)
            for p in range(n):
                if (s1 >> p) & 1:
                    continue
                sign2 = -1.0 if bin(s1 & ((1 << p) - 1)).count('1') % 2 else 1.0
                ja = cistring.str2addr(n, nocc, s1 | (1 << p))
                T[ja, ia] += lk[p, q] * sign1 * sign2
    return T


def referee_L2(L, ci, na, nb):
    """<L^2> = sum_k || L^_k psi ||^2 via direct operator application."""
    va = np.zeros((ci.shape[0], ci.shape[1]), dtype=complex)
    vb = np.zeros_like(va)
    Ta = [one_body_T(L[k].shape[0], na, L[k]) for k in range(3)]
    Tb = [one_body_T(L[k].shape[0], nb, L[k]) for k in range(3)]
    tot = 0.0
    for k in range(3):
        out = Ta[k] @ ci + ci @ Tb[k].T
        tot += float(np.linalg.norm(out) ** 2)
    return tot.real


def two_val(name, A, B, dm2, sign):
    return sign * np.einsum('pq,rs,' + name + '->', A, B, dm2).real


def one_full(L, k, dm1, n):
    return np.einsum('ps,ps->', (L[k] @ L[k])[:n, :n], dm1).real


def contract_scan(L, dm1, dm2, n, referee):
    ob = sum(one_full(L, k, dm1, n) for k in range(3))
    print(f'    one-body (complete int.) = {ob:+.6f};  referee <L^2> = {referee:+.8f}')
    ok = []
    for pairing in PAIRINGS:
        for sign in (-1, +1):
            v = ob + sum(two_val(pairing, L[k][:n, :n], L[k][:n, :n], dm2, sign)
                         for k in range(3))
            mark = '  <== matches referee' if abs(v - referee) < 1e-6 else ''
            if abs(v - referee) < 1e-6:
                ok.append((pairing, sign))
            print(f'    {pairing} sign{sign:+d}: {v:+.8f}{mark}')
    return ok


def gate(mf, mol, ncas, nelecas, act, n, nelec, tag):
    mci = mcscf.CASCI(mf, ncas, nelecas)
    mci.kernel(mci.sort_mo(act))
    ci = expand_product_ci(ncore=n - ncas, ncas=ncas, ci_act=mci.ci,
                           nelecas=nelecas)
    L = get_orbital_angular_momentum(mol, mf.mo_coeff[:, :n])   # ORIGINAL order
    dm1, dm2 = fci.direct_spin1.make_rdm12(ci, n, nelec)
    ref = referee_L2(L, ci, *nelec)
    print(f'\n=== {tag}: CAS({sum(nelecas)},{ncas}) ===')
    print(f'    dm1 diag = {np.diag(dm1).real.round(6)}')
    good = contract_scan(L, dm1, dm2, n, ref)
    return good


def main():
    winners = None

    # B_STO
    mol = gto.M(atom='C 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    w = gate(mf, mol, 3, (2, 0), [2, 3, 4], 5, (4, 2), 'B_STO')
    winners = w if winners is None else [x for x in winners if x in w]

    # D_STO
    mol = gto.M(atom='C 0 0 0', basis='sto-3g', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    w = gate(mf, mol, 3, (1, 0), [2, 3, 4], 5, (3, 2), 'D_STO')
    winners = [x for x in winners if x in w]

    # B_pvdz
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    w = gate(mf, mol, 3, (2, 0), [2, 3, 4], 5, (4, 2), 'B_pvdz')
    winners = [x for x in winners if x in w]

    # D_pvdz
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', charge=1, spin=1, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    w = gate(mf, mol, 3, (1, 0), [2, 3, 4], 5, (3, 2), 'D_pvdz')
    winners = [x for x in winners if x in w]

    # FCI_C and FCI_Cp (full space)
    for tag, spin, chg, nelec in (('FCI_C', 2, 0, (4, 2)), ('FCI_Cp', 1, 1, (3, 2))):
        mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=spin, charge=chg, verbose=0)
        mf = scf.ROHF(mol); mf.kernel()
        cis = fci.FCI(mf); e, ci = cis.kernel()
        n = mol.nao
        L = get_orbital_angular_momentum(mol, mf.mo_coeff)
        dm1, dm2 = fci.direct_spin1.make_rdm12(ci, n, nelec)
        ref = referee_L2(L, ci, *nelec)
        print(f'\n=== {tag}: full-space FCI ===')
        good = contract_scan(L, dm1, dm2, n, ref)
        winners = [x for x in winners if x in good] if winners else good

    print('\n================ VERDICT ================')
    print(f'  conventions matching the referee on ALL systems: {winners}')
    print('=========================================\n')


if __name__ == '__main__':
    main()
