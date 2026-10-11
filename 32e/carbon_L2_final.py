#!/usr/bin/env python
"""
carbon_L2_final.py — convention-free settlement of the <L^2> evaluator.

ALL second-quantized operator applications use pyscf's OWN sign tables
(fci.cistring.gen_cre_str_index / gen_des_str_index — the machinery behind
fci.spin_op.contract_ss), so there is NO sign-convention risk anywhere.

  [1] state validation: pyscf.spin_square on the gate-B CI  -> S^2 = 2 (3P)
  [2] TRUE <L^2>: L^ applied to the embedded full-basis CI with pyscf
      signs, <L^2> = sum_k ||L^_k psi||^2                    -> the verdict
  [3] exact G4[p,q,r,s] = <a+_p a_q a+_r a_s> (spin-summed) by direct
      operator application; layout scan of make_rdm12's dm2 vs
      G4 - delta_qr*dm1
  [4] assembly: one-body(complete intermediates) + true two-body vs [2]
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from pyscf.fci.spin_op import spin_square
from carbon_L2 import get_orbital_angular_momentum, expand_product_ci

PAIRINGS = ('pqrs', 'prqs', 'psrq', 'prsq', 'pqsr', 'psqr')


def _lookup(tab, norb, orb_col):
    """(addr, sign) lookup arrays indexed [string, orbital]; sign 0 = invalid.
    Valid rows have sign = +-1 (the rest are zero-padded)."""
    nstr = tab.shape[0]
    addr = np.zeros((nstr, norb), dtype=int)
    sgn = np.zeros((nstr, norb), dtype=int)
    for k in range(nstr):
        m = tab[k, :, 3] != 0                     # sign column: validity flag
        orbs = tab[k, m, orb_col]
        addr[k, orbs] = tab[k, m, 2]
        sgn[k, orbs] = tab[k, m, 3]
    return addr, sgn


def _tables(norb, ne, kind):
    if kind == 'des':
        tab = cistring.gen_des_str_index(range(norb), ne)
        return _lookup(tab, norb, 1)
    tab = cistring.gen_cre_str_index(range(norb), ne)
    return _lookup(tab, norb, 0)


def apply_1body(vec, na, nb, norb, T):
    """L^|vec> = (T on alpha sector) + (T on beta sector), pyscf signs."""
    out = np.zeros_like(vec, dtype=complex)
    for ne, axis in ((na, 0), (nb, 1)):
        if ne == 0:
            continue
        daddr, dsgn = _tables(norb, ne, 'des')
        caddr, csgn = _tables(norb, ne - 1, 'cre')
        num = vec.shape[axis]
        for ia in range(num):
            for q in range(norb):
                if dsgn[ia, q] == 0:
                    continue
                mid = daddr[ia, q]
                s1 = dsgn[ia, q]
                for p in range(norb):
                    if csgn[mid, p] == 0:
                        continue
                    amp = T[p, q] * s1 * csgn[mid, p]
                    if axis == 0:
                        out[caddr[mid, p], :] += amp * vec[ia, :]
                    else:
                        out[:, caddr[mid, p]] += vec[:, ia] * amp
    return out


def a_sector(vec, norb, ne, idx, axis):
    if ne == 0:
        return np.zeros(1)
    daddr, dsgn = _tables(norb, ne, 'des')
    nout = cistring.num_strings(norb, ne - 1)
    shape = (nout, vec.shape[1]) if axis == 0 else (vec.shape[0], nout)
    out = np.zeros(shape, dtype=vec.dtype)
    for ia in range(vec.shape[axis]):
        if dsgn[ia, idx] == 0:
            continue
        if axis == 0:
            out[daddr[ia, idx], :] += dsgn[ia, idx] * vec[ia, :]
        else:
            out[:, daddr[ia, idx]] += vec[:, ia] * dsgn[ia, idx]
    return out


def ad_sector(vec, norb, ne, idx, axis):
    if ne >= norb:
        return np.zeros(1)
    caddr, csgn = _tables(norb, ne, 'cre')
    nout = cistring.num_strings(norb, ne + 1)
    shape = (nout, vec.shape[1]) if axis == 0 else (vec.shape[0], nout)
    out = np.zeros(shape, dtype=vec.dtype)
    for ia in range(vec.shape[axis]):
        if csgn[ia, idx] == 0:
            continue
        if axis == 0:
            out[caddr[ia, idx], :] += csgn[ia, idx] * vec[ia, :]
        else:
            out[:, caddr[ia, idx]] += vec[:, ia] * csgn[ia, idx]
    return out


def main():
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    mci = mcscf.CASCI(mf, 3, (2, 0))
    mci.kernel(mci.sort_mo([2, 3, 4]))

    n_full = mol.nao
    na, nb = 4, 2
    print('[1] pyscf.spin_square of the gate-B CI (3 active MOs):')
    ss, mult = spin_square(mci.ci, 3, (2, 0))
    print(f'    S^2 = {ss:.8f}, multiplicity = {mult:.4f}  (3P demand: 2.0, 3)')
    ci5 = expand_product_ci(2, 3, mci.ci, (2, 0))
    ss5, mult5 = spin_square(ci5, 5, (na, nb))
    print(f'    S^2 of the expanded 5-MO CI = {ss5:.8f}  (2.0)')

    print('[2] TRUE <L^2> via pyscf-sign operator application (full 14-MO):')
    L = get_orbital_angular_momentum(mol, mf.mo_coeff)
    # embed into full space
    core = 0b11
    ci = np.zeros((cistring.num_strings(n_full, na),
                   cistring.num_strings(n_full, nb)), dtype=complex)
    for ia in range(3):
        sa = int(cistring.addr2str(3, 2, ia))
        x = core | (sa << 2)
        ci[cistring.str2addr(n_full, na, x), cistring.str2addr(n_full, nb, core)] = \
            mci.ci[ia, 0]
    tot = 0.0
    for k in range(3):
        w = apply_1body(ci, na, nb, n_full, L[k])
        tot += float(np.linalg.norm(w) ** 2)
    print(f'    TRUE <L^2> = {tot:.8f}')

    print('[3] exact G4 vs make_rdm12 layouts:')
    dm1, dm2 = fci.direct_spin1.make_rdm12(ci5, 5, (na, nb))
    n = 5
    G4 = np.zeros((n,) * 4)
    for p in range(n):
        for sig in (0, 1):
            nes = na if sig == 0 else nb
            bra = a_sector(ci5, n, nes, p, sig)
            if np.linalg.norm(bra) < 1e-14:
                continue
            for r in range(n):
                for s_ in range(n):
                    for tau in (0, 1):
                        net = na if tau == 0 else nb
                        w = a_sector(ci5, n, net, s_, tau)
                        if np.linalg.norm(w) < 1e-14:
                            continue
                        w = ad_sector(w, n, net - 1, r, tau)
                        for q in range(n):
                            wq = a_sector(w, n, nes, q, sig)
                            G4[p, q, r, s_] += float(
                                np.real(np.sum(np.conj(bra) * wq)))
    delta = np.einsum('qr,ps->pqrs', np.eye(n), dm1.real)
    resid = G4 - delta
    print(f'    |G4| max = {np.abs(G4).max():.4f},  |resid| max = {np.abs(resid).max():.4f}')
    perms = {'pqrs': (0, 1, 2, 3), 'prqs': (0, 2, 1, 3), 'psrq': (0, 3, 1, 2),
             'prsq': (0, 2, 3, 1), 'pqsr': (0, 1, 3, 2), 'psqr': (0, 3, 1, 2)}
    for name, perm in perms.items():
        v = np.transpose(dm2, perm)
        for sign in (-1, 1):
            err = np.abs(resid - sign * v).max()
            mark = '  <== EXACT' if err < 1e-9 else ''
            print(f'    resid = {sign:+d} dm2[{name}]: max|err| = {err:.2e}{mark}')

    print('[4] assembly:')
    cols = list(range(5))
    one = sum(np.einsum('ps,ps->', (L[k] @ L[k])[np.ix_(cols, cols)],
                        dm1).real for k in range(3))
    Lc = L[:, cols][:, :, cols]
    two = sum(np.einsum('pq,rs,pqrs->', Lc[k], Lc[k], resid) for k in range(3))
    print(f'    one-body = {one:+.6f}, true-two-body = {two:+.6f},'
          f'  total = {one + two:.8f}   (compare [2])')


if __name__ == '__main__':
    main()
