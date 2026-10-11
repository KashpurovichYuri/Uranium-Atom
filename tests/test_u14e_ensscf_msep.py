#!/usr/bin/env python
"""test_u14e_ensscf_msep.py — A2 completion: resolve the paradox
(l-pure ensemble basis yet span-broken) via the m-separability fix.

1. Rerun the ensemble-SCF (the locked protocol from test_u14e_ensscf.py).
2. DIAGNOSTIC: the <Lz> diagonal of the active columns — the paradox's
   fingerprint (l-pure columns must still be m-eigenstates; the naive
   purification leaves them m-mixed within degenerate manifolds).
3. Purify with spherical_reference v2 (m_separable=True) -> certificates,
   Lz-check, passport.
4. CASCI(6,13)(5,1) in the fixed basis -> the term check (<L^2> root0).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.reference import spherical_reference
from lmom.diag import purity_devs
from lmom.moments import moment_mats, moment_expectation
from battery_lib import mo_passport, lz_check, casci_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def main():
    mol = build_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo0 = d['mo_coeff']
    act = [int(c) for c in d['act_cols']]
    s7 = 4
    occ = np.zeros(mo0.shape[1])
    occ[:NCORE] = 2.0
    for c in act:
        occ[c] = 2.0 if c == s7 else \
            (3.0 / 7 if 22 <= c <= 28 else 1.0 / 5)
    S_ao = mol.intor('int1e_ovlp')
    shells = [([4], 2.0),
              (list(range(9, 14)), 1.0 / 5),
              (list(range(22, 29)), 3.0 / 7)]

    def shell_get_occ(mo_energy=None, mo_coeff=None):
        mo = mo_coeff
        occ_r = np.zeros(mo.shape[1])
        occ_r[:NCORE] = 2.0
        used = set(range(NCORE))
        for cols0, occval in shells:
            P = mo0[:, cols0]
            score = ((P.T @ S_ao @ mo) ** 2).sum(axis=0)
            idx = np.argsort(score)[::-1]
            take = 0
            for j in idx:
                if j in used:
                    continue
                occ_r[j] = occval
                used.add(j)
                take += 1
                if take == len(cols0):
                    break
        return occ_r

    def frac_rdm1(mo_coeff=None, mo_occ=None, **k):
        moc = mo0 if mo_coeff is None else np.asarray(mo_coeff)
        o = occ if mo_occ is None else np.asarray(mo_occ)
        return (moc * o) @ moc.T

    mf = scf.ROHF(mol)
    mf.mo_coeff = mo0
    mf.mo_occ = occ
    mf.get_occ = shell_get_occ
    mf.make_rdm1 = frac_rdm1
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    print(f'ensemble-SCF: converged = {mf.converged}, '
          f'E = {float(mf.e_tot):.8f}')

    # ---- 2. the DIAGNOSTIC: <Lz> per active column BEFORE the fix ----
    L = get_L(mol, mf.mo_coeff)
    sc0 = span_certificate(L, act)
    p0 = purity_devs(L, act)
    dev0, m0 = lz_check(mol, mf.mo_coeff, act, 'ensemble RAW')
    print(f'  BEFORE fix: span invar = {sc0["invar_err"]:.2e} '
          f'(l_closed {sc0["l_closed"]}), purity = {p0.max():.1e}')

    # ---- 3. the v2 purification ----
    mf_d = scf.ROHF(mol)
    mf_d.mo_coeff = mf.mo_coeff
    mf_d.mo_occ = occ
    mf_d.e_tot = 0.0
    mf_d.converged = True
    mo_p, mo_e_p = spherical_reference(mf_d, m_separable=True)
    mo_passport(mol, mo_p, None, act, None, None,
                expect_l=[0] + [2] * 5 + [3] * 7, label='ensemble v2')
    Lp = get_L(mol, mo_p)
    sc_p = span_certificate(Lp, act)
    p_p = purity_devs(Lp, act)
    dev_p, m_p = lz_check(mol, mo_p, act, 'ensemble v2')
    print(f'  AFTER fix: span invar = {sc_p["invar_err"]:.2e} '
          f'(l_closed {sc_p["l_closed"]}), purity = {p_p.max():.1e}')

    # ---- 4. the CASCI term check in the fixed basis ----
    mf_c = scf.ROHF(mol)
    mf_c.mo_coeff = mo_p
    mf_c.mo_occ = occ
    mf_c.e_tot = 0.0
    mf_c.converged = True
    cis, e_ci = casci_run(mol, mf_c, act, (NA, NB), NCORE, 5,
                          conv_tol=1e-12)
    mats = moment_mats(mol, mo_p, act, NA, NB, 'L2')
    from pyscf.fci import cistring
    nsa = cistring.num_strings(len(act), NA)
    nsb = cistring.num_strings(len(act), NB)
    q = [moment_expectation(mats, nsa, nsb, c) for c in cis]
    print(f'ensemble-CASCI (v2 basis): E = {e_ci[0]:.8f}; '
          f'<L^2> per root = {np.round(q, 5)}')
    ok = (sc_p['l_closed'] and dev_p < 1e-6
          and all(abs(v - 42) < 1e-4 for v in q))
    out = dict(e_ensemble=float(mf.e_tot),
               before=dict(invar=float(sc0['invar_err']),
                           lz_dev=dev0),
               after=dict(invar=float(sc_p['invar_err']),
                          l_closed=bool(sc_p['l_closed']),
                          lz_dev=dev_p, purity=float(p_p.max())),
               e_casci=float(e_ci[0]), l2_roots=[float(x) for x in q],
               verdict='PASS' if ok else 'FAIL')
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/u14e_ensscf_msep.meta.json'), 'w'), indent=1)
    print(f'\n==> A2 completion: {"PASS" if ok else "FAIL"} '
          f'(stored: u14e_ensscf_msep.meta.json)')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
