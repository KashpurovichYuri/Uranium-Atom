#!/usr/bin/env python
"""test_u14e_ensscf.py — A2: the self-consistent spherical-ENSEMBLE SCF
reference protocol (U-14e).

Protocol: pin the shell-m-averaged occupations (core 6s2 6p6 = 2.0;
7s = 2.0; 5f = 3/7 per column; 6d = 1/5 per column) through a FULL SCF
(index-pinned get_occ, start from the stored spherical basis), then:
  E_ensemble-SCF  — the variationally optimal energy WITHIN the
                    spherical-ensemble manifold
  span/purity     — the orbitals should stay l-pure (the density is
                    spherical by construction)
  one-shot spherical_reference afterwards = expected near-no-op
  CASCI(6,13)(5,1) in the ensemble basis: the ladder root check.

Position in the energy-legitimacy table:
  E(broken SCF) <= E(ensemble-SCF) <= E(one-shot averaged density)
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
from battery_lib import mo_passport, casci_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, spin=4,
                 verbose=0)   # spin=4 MANDATORY: unset spin silently
                 # converges the singlet Aufbau (the -50.1516 incident)


def main():
    mol = build_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo0 = d['mo_coeff']
    act = [int(c) for c in d['act_cols']]
    s7 = 4
    # the pinned m-averaged occ (aligned to the stored l-ordered basis)
    occ = np.zeros(mo0.shape[1])
    occ[:NCORE] = 2.0
    for c in act:
        occ[c] = 2.0 if c == s7 else \
            (3.0 / 7 if 22 <= c <= 28 else 1.0 / 5)

    # the self-consistent ensemble SCF (start from the spherical basis).
    # INDEX-pinning alone scrambles: each SCF cycle re-diagonalizes F,
    # columns swap identities, and the pinned occupations land on
    # different physical orbitals. MOM (maximum overlap) assigns the
    # occupations to the columns that overlap most with the START —
    # the same remedy as the 14e driver's MOM/IDX rows.
    # Shell-based MOM with fractional smearing (the minimal CHAR):
    # plain MOM integerizes occupations (1/0), which re-anisotropizes
    # the density; instead, per SCF cycle, match each reference shell
    # to its best-overlap current columns and assign the m-averaged
    # FRACTIONAL pattern to them.
    S_ao = mol.intor('int1e_ovlp')
    shells = [([4], 2.0),
              (list(range(9, 14)), 1.0 / 5),
              (list(range(22, 29)), 3.0 / 7)]

    def shell_get_occ(mo_energy=None, mo_coeff=None):
        mo = mo_coeff
        occ_r = np.zeros(mo.shape[1])
        occ_r[:NCORE] = 2.0                       # the frozen core
        used = set(range(NCORE))
        for cols0, occval in shells:
            P = mo0[:, cols0]
            score = (P.T @ S_ao @ mo) ** 2
            score = score.sum(axis=0)
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

    # The ensemble SCF runs on a UHF CARRIER: fractional (alpha,beta)
    # occupations are exact there (occa = clip(occ,0,1),
    # occb = clip(occ-1,0,1)) — pyscf's ROHF would threshold the 1-d
    # occ by occ>0 in its internal Fock build (the electron-count
    # corruption trap) and evaluate energies with the RHF total-dm
    # formula. The UHF carrier needs no overrides beyond get_occ.
    occa0 = np.clip(occ, 0, 1)
    occb0 = np.clip(occ - 1, 0, 1)
    mf = scf.UHF(mol)
    mf.max_cycle = 500
    mf.conv_tol = 1e-9
    mf.mo_coeff = np.stack([mo0, mo0])
    mf.mo_occ = np.stack([occa0, occb0])

    def shell_get_occ_uhf(mo_energy=None, mo_coeff=None, **k):
        moc = mf.mo_coeff if mo_coeff is None else np.asarray(mo_coeff)
        # lock the TOTAL 1-d pattern, then split per spin
        occ_tot = shell_get_occ(mo_energy=mo_energy, mo_coeff=moc[0])
        return np.stack([np.clip(occ_tot, 0, 1),
                         np.clip(occ_tot - 1, 0, 1)])

    mf.get_occ = shell_get_occ_uhf
    t0 = time.time()
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.get_occ = shell_get_occ_uhf
        mf.kernel()
    e_ens = float(mf.e_tot)
    occ = np.asarray(mf.mo_occ).sum(axis=0)
    print(f'ensemble-SCF: converged = {mf.converged}, '
          f'E = {e_ens:.8f} ({time.time() - t0:.0f} s)')

    # the passport: did the columns keep their shells?
    p1 = mo_passport(mol, np.asarray(mf.mo_coeff)[0], occ, act,
                     list(range(NCORE)), 14,
                     expect_l=[0] + [2] * 5 + [3] * 7,
                     label='ensemble-SCF')

    # certificates + purity on the ensemble orbitals
    L = get_L(mol, np.asarray(mf.mo_coeff)[0])
    sc = span_certificate(L, act)
    pdev = purity_devs(L, act)
    print(f'span: l_closed = {sc["l_closed"]}, invar = '
          f'{sc["invar_err"]:.2e}; purity max = {pdev.max():.2e}')

    # the one-shot purification afterwards: for an already-spherical
    # solution it must keep the span and the l-structure (the column
    # set may only rotate within degenerate shells; compare via the
    # certificate, not raw arrays — phases/orderings are arbitrary)
    mf_d = scf.ROHF(mol)
    mf_d.mo_coeff = np.asarray(mf.mo_coeff)[0]
    mf_d.mo_occ = occ
    mf_d.e_tot = 0.0
    mf_d.converged = True
    mo_p, _ = spherical_reference(mf_d)
    Lp = get_L(mol, mo_p)
    sc_p = span_certificate(Lp, act)
    print(f'one-shot purification: span l_closed = {sc_p["l_closed"]}, '
          f'invar = {sc_p["invar_err"]:.2e}')

    # CASCI in the ensemble basis
    cis, e_ci = casci_run(mol, mf, act, (NA, NB), NCORE, 3,
                          conv_tol=1e-12)
    mats = moment_mats(mol, mo_p, act, NA, NB, 'L2')
    nsa = __import__('pyscf.fci.cistring', fromlist=['cistring']) \
        .num_strings(len(act), NA)
    nsb = __import__('pyscf.fci.cistring', fromlist=['cistring']) \
        .num_strings(len(act), NB)
    q0 = moment_expectation(mats, nsa, nsb, cis[0])
    print(f'ensemble-CASCI: E = {e_ci[0]:.8f} (sph-ROHF ref '
          f'-50.164365); <L^2>(root0) = {q0:.6f}')

    # e_broken_scf: the TRUE unrestricted quintet ROHF (fresh, spin=4;
    # the old -50.15164379 was the SINGLET Aufbau solution — an unset
    # mol.spin — verified 2026: fresh spin-4 ROHF = -50.38185233,
    # fresh spin-0 ROHF = -50.15164379)
    mf_raw = scf.ROHF(mol)
    mf_raw.kernel()
    if not mf_raw.converged:
        mf_raw = mf_raw.newton()
        mf_raw.kernel()
    e_broken = float(mf_raw.e_tot)
    ref = dict(
        e_broken_scf=e_broken,
        e_ensemble_scf=e_ens,
        e_oneshot_avg_density=-49.71430012,  # exact evaluator (energy_ncheck)
        e_casci_sph_basis=-50.16436519,
        e_casci_ensemble=float(e_ci[0]),
        l2_root0=float(q0),
        l_closed=bool(sc['l_closed']), invar=float(sc['invar_err']),
        purity_max=float(pdev.max()),
        passport=p1,
        purif_l_closed=bool(sc_p['l_closed']),
        purif_invar=float(sc_p['invar_err']))
    json.dump(ref, open(os.path.join(ROOT,
              'lmom/data/u14e_ensscf.meta.json'), 'w'), indent=1)
    print('\nreference-energy table:')
    print(f'  broken SCF            {ref["e_broken_scf"]:.6f}')
    print(f'  ensemble SCF          {ref["e_ensemble_scf"]:.6f}')
    print(f'  one-shot avg density  {ref["e_oneshot_avg_density"]:.6f}')
    print(f'  CASCI (sph basis)     {ref["e_casci_sph_basis"]:.6f}')
    print('stored: u14e_ensscf.meta.json')
    return bool(p1['ok'] and sc['l_closed'] and abs(q0 - 42) < 1e-4)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
