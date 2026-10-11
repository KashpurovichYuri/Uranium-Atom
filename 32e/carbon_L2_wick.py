#!/usr/bin/env python
"""
carbon_L2_wick.py — settle the dm2 layout by DIRECT MEASUREMENT.

For the gate-B state (cc-pvdz C CAS(2,3)(2,0) expanded CI, a single-det
3P state where the two-body Hund term matters):
  1. TRUE <L^2> via the full-basis (14-MO) convention-free referee
     (second-quantized L^ applied twice)  -> expect 2.000000 (pure 3P).
  2. The EXACT tensor G4[p,q,r,s] = <a+_p a_q a+_r a_s> summed over spins,
     built by applying annihilation/creation operators to the CI vector
     directly (no rdm routines involved).
  3. Compare G4 against every candidate dm2 layout from
     fci.direct_spin1.make_rdm12 to identify the correct pairing/sign.
  4. Assemble <L^2> = one-body(complete int.) + true two-body and verify.
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from carbon_L2 import get_orbital_angular_momentum, expand_product_ci

PAIRINGS = ('pqrs', 'prqs', 'psrq', 'prsq', 'pqsr', 'psqr')


def apply_1body(vec, n, nocc, lk, dagger, annih):
    """Apply sum_pq lk[p,q] a+_p a_q (single spin sector) to vec."""
    num = cistring.num_strings(n, nocc)
    out = np.zeros_like(vec)
    for ia in range(num):
        c = vec[ia]
        if abs(c) < 1e-14:
            continue
        s = int(cistring.addr2str(n, nocc, ia))
        if annih:
            for q in range(n):
                if not (s >> q) & 1:
                    continue
                sgn1 = -1.0 if bin(s & ((1 << q) - 1)).count('1') % 2 else 1.0
                s1 = s ^ (1 << q)
                for p in range(n):
                    if (s1 >> p) & 1:
                        continue
                    sgn2 = -1.0 if bin(s1 & ((1 << p) - 1)).count('1') % 2 else 1.0
                    ja = cistring.str2addr(n, nocc, s1 | (1 << p))
                    out[ja] += lk[p, q] * sgn1 * sgn2 * c
        else:
            for q in range(n):
                if not (s >> q) & 1:
                    continue
                sgn1 = -1.0 if bin(s & ((1 << q) - 1)).count('1') % 2 else 1.0
                s1 = s ^ (1 << q)
                for p in range(n):
                    if (s1 >> p) & 1:
                        continue
                    sgn2 = -1.0 if bin(s1 & ((1 << p) - 1)).count('1') % 2 else 1.0
                    ja = cistring.str2addr(n, nocc, s1 | (1 << p))
                    out[ja] += lk[p, q].conj() * sgn1 * sgn2 * c
    return out


def referee_full(mol, mo_full, ci_act, ncore, nelecas, n_full):
    """TRUE <L^2>: apply the second-quantized L^ (full n_full-MO basis) to
    the expanded CI, twice; <L^2> = sum_k ||L^_k psi||^2. Convention-free."""
    na = ncore + nelecas[0]
    nb = ncore + nelecas[1]
    # embed the CAS CI in the FULL n_full-MO determinant space
    ncas = 3
    core = (1 << ncore) - 1
    na_a, nb_a = nelecas
    ci = np.zeros((cistring.num_strings(n_full, na),
                   cistring.num_strings(n_full, nb)), dtype=complex)

    def sector_str(ne, ia):
        return 0 if ne == 0 else int(cistring.addr2str(ncas, ne, ia))
    for ia in range(ci_act.shape[0]):
        sa = sector_str(na_a, ia)
        for ib in range(ci_act.shape[1]):
            sb = sector_str(nb_a, ib)
            x = core | (sa << ncore)
            y = core | (sb << ncore)
            ci[cistring.str2addr(n_full, na, x),
               cistring.str2addr(n_full, nb, y)] = ci_act[ia, ib]

    def move_matrix(n, nocc, lk):
        num = cistring.num_strings(n, nocc)
        T = np.zeros((num, num), dtype=complex)
        for ia in range(num):
            s = int(cistring.addr2str(n, nocc, ia))
            for q in range(n):
                if not (s >> q) & 1:
                    continue
                sgn1 = -1.0 if bin(s & ((1 << q) - 1)).count('1') % 2 else 1.0
                s1 = s ^ (1 << q)
                for p in range(n):
                    if (s1 >> p) & 1:
                        continue
                    sgn2 = -1.0 if bin(s1 & ((1 << p) - 1)).count('1') % 2 else 1.0
                    ja = cistring.str2addr(n, nocc, s1 | (1 << p))
                    T[ja, ia] += lk[p, q] * sgn1 * sgn2
        return T

    L = get_orbital_angular_momentum(mol, mo_full)      # FULL basis
    tot = 0.0
    for k in range(3):
        Ta = move_matrix(n_full, na, L[k])
        Tb = move_matrix(n_full, nb, L[k])
        out = Ta @ ci + ci @ Tb.T
        tot += float(np.linalg.norm(out) ** 2)
    return tot.real


def G4_direct(mol, mo_full, ci_act, ncore, nelecas):
    """Clean implementation: G4[p,q,r,s] = sum_{sigma,tau}
    <psi| a+_p(sigma) a_q(sigma) a+_r(tau) a_s(tau) |psi>.
    Uses <psi|A|psi> = <(a_p psi)| (a_q a+_r a_s psi)> on the same sector."""
    ncas = 3
    n = ncore + ncas
    ci = expand_product_ci(ncore, ncas, ci_act, nelecas)
    na = ncore + nelecas[0]
    nb = ncore + nelecas[1]
    dims = {0: (na, ci.shape[0]), 1: (nb, ci.shape[1])}

    def a_on(vec2d, sector, idx):
        nocc = dims[sector][0]
        n_ = n
        out = np.zeros_like(vec2d)
        num = vec2d.shape[sector]
        for ia in range(num):
            s = int(cistring.addr2str(n_, nocc, ia))
            if not (s >> idx) & 1:
                continue
            sgn = -1.0 if bin(s & ((1 << idx) - 1)).count('1') % 2 else 1.0
            ja = cistring.str2addr(n_, nocc, s ^ (1 << idx))
            if sector == 0:
                out[ja, :] += sgn * vec2d[ia, :]
            else:
                out[:, ja] += sgn * vec2d[:, ia]
        return out

    def ad_on(vec2d, sector, idx):
        nocc = dims[sector][0]
        out = np.zeros_like(vec2d)
        num = vec2d.shape[sector]
        for ia in range(num):
            s = int(cistring.addr2str(n, nocc, ia))
            if (s >> idx) & 1:
                continue
            sgn = -1.0 if bin(s & ((1 << idx) - 1)).count('1') % 2 else 1.0
            ja = cistring.str2addr(n, nocc, s | (1 << idx))
            if sector == 0:
                out[ja, :] += sgn * vec2d[ia, :]
            else:
                out[:, ja] += sgn * vec2d[:, ia]
        return out

    G4 = np.zeros((n,) * 4)
    for p in range(n):
        for q in range(n):
            for sigma in (0, 1):
                bra = a_on(ci, sigma, p)          # a_p |psi>
                nrm = np.linalg.norm(bra)
                if nrm < 1e-14:
                    continue
                for r in range(n):
                    for s in range(n):
                        for tau in (0, 1):
                            w = a_on(ci, tau, s)
                            if np.linalg.norm(w) < 1e-14:
                                continue
                            w = ad_on(w, tau, r)
                            w = a_on(w, sigma, q)
                            G4[p, q, r, s] += np.real(np.sum(np.conj(bra) * w))
    return G4


def main():
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    mci = mcscf.CASCI(mf, 3, (2, 0))
    mci.kernel(mci.sort_mo([2, 3, 4]))

    n_full = mol.nao
    print('[1] TRUE <L^2> (full 14-MO referee on the expanded CI):')
    ref = referee_full(mol, mf.mo_coeff, mci.ci, 2, (2, 0), n_full)
    print(f'    referee <L^2> = {ref:+.8f}   (pure-3P Pauli demand: 2.000000)')

    ci5 = expand_product_ci(2, 3, mci.ci, (2, 0))
    dm1, dm2 = fci.direct_spin1.make_rdm12(ci5, 5, (4, 2))
    print('[2] exact G4[p,q,r,s] = <a+_p a_q a+_r a_s> (spin-summed):')
    G4 = G4_direct(mol, mf.mo_coeff, mci.ci, 2, (2, 0))
    print(f'    G4 norm = {np.abs(G4).max():.4f}')

    print('[3] layout identification: G4 -=? dm2[layout]:')
    # delta part: delta_qr dm1[p,s]
    delta = np.einsum('qr,ps->pqrs', np.eye(5), dm1.real)
    resid = G4 - delta
    for layout in PAIRINGS:
        # dm2 indexed as dm2[p,q,r,s]; build permuted views
        if layout == 'pqrs':
            v = dm2
        elif layout == 'prqs':
            v = np.einsum('pqrs->prqs', dm2)
        elif layout == 'psrq':
            v = np.einsum('pqrs->psrq', dm2)
        elif layout == 'prsq':
            v = np.einsum('pqrs->prsq', dm2)
        elif layout == 'pqsr':
            v = np.einsum('pqrs->pqsr', dm2)
        elif layout == 'psqr':
            v = np.einsum('pqrs->psqr', dm2)
        for sign in (-1, +1):
            err = np.abs(resid - sign * v).max()
            mark = '  <== EXACT' if err < 1e-8 else ''
            print(f'    G4 = delta + sign*dm2[{layout}]: max|err| = {err:.2e}{mark}')

    print('[4] assemble <L^2> with the verified layout:')
    L = get_orbital_angular_momentum(mol, mf.mo_coeff)
    cols = [0, 1, 2, 3, 4]
    one = sum(np.einsum('ps,ps->', (L[k] @ L[k])[np.ix_(cols, cols)], dm1).real
              for k in range(3))
    Lc = L[:, cols][:, :, cols]
    # true two-body: sum_k sum_pqrs (l)_pq (l)_rs resid[p,q,r,s]
    two_true = sum(np.einsum('pq,rs,pqrs->', Lc[k], Lc[k], resid) for k in range(3))
    print(f'    one-body = {one:+.6f};  true-two-body = {two_true:+.6f};'
          f'  total = {one + two_true:+.8f}')


if __name__ == '__main__':
    main()
