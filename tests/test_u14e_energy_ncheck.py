#!/usr/bin/env python
"""test_u14e_energy_ncheck.py — U-14e items 0 + 1:

Item 0 (energy legitimacy): is the sph reference energetically
  equivalent to the raw ROHF?
  E(raw ROHF SCF density) vs E(sph MOs density) vs E_CASCI(raw basis)
  vs E_CASCI(sph basis); reported in mHa AND as a fraction of the
  smallest measured term gap (5I->5L from the 60-root study).

Item 1 (verification row, U part): electron-number checks on the
  stored 60 CASCI roots (dm-route Tr dm1, dm-route <N^2>, the
  number-operator route <N>/<N^2>) + CASSCF stationarity (orbital
  gradient norm at the returned solution; conv sensitivity 1e-7 vs
  1e-9 on ss-CASSCF(6,13); the three prior ss stationary points
  documented).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from lmom.moments import moment_mats, moment_expectation
from lmom.expectation import L2Lz_expectation_dm
from battery_lib import (n_expectation, energy_of_density, casci_run)

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
    out = {}
    mol = build_mol()
    d_sph = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                    allow_pickle=True)
    d_raw = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_raw.npz'),
                    allow_pickle=True)
    d60 = np.load(os.path.join(ROOT, 'lmom/data/u14e_roots60.npz'),
                  allow_pickle=True)

    # ---------- item 0: energy legitimacy (FRESH, reproducible SCF
    # pair; the stored occ vectors are unusable for densities — the
    # sph one sums to 22, not 14 — data-hygiene note recorded) ----------
    from lmom.reference import spherical_reference
    mf_raw = scf.ROHF(mol)
    mf_raw.kernel()
    if not mf_raw.converged:
        mf_raw = mf_raw.newton()
        mf_raw.kernel()
    e_raw = float(mf_raw.e_tot)
    dm = mf_raw.make_rdm1()
    if dm.ndim == 3:            # the spin-Kramer pair -> the total dm
        dm = dm.sum(axis=0)
    # the m-averaged density: the exact block code of
    # lmom/reference.py (proven), no re-diagonalization
    dm_sph = np.zeros_like(dm)
    loc = [int(x) for x in mol.ao_loc_nr()]
    ranges = []
    for i in range(mol.nbas):
        a0, a1 = loc[i], loc[i + 1]
        la = int(mol._bas[i, gto.ANG_OF])
        n = 2 * la + 1
        for c in range((a1 - a0) // n):
            ranges.append((a0 + c * n, a0 + (c + 1) * n, la))
    for a0, a1, la in ranges:
        for b0, b1, lb in ranges:
            if la != lb:
                continue
            if la == 0:
                dm_sph[a0:a1, b0:b1] = dm[a0:a1, b0:b1]
            else:
                n = a1 - a0
                dm_sph[a0:a1, b0:b1] = \
                    (np.trace(dm[a0:a1, b0:b1]) / n) * np.eye(n)
    e_sph = float(mf_raw.energy_tot(dm=dm_sph))
    print(f'E(raw ROHF SCF)            = {e_raw:.8f}')
    print(f'E(m-averaged density)      = {e_sph:.8f}   '
          f'dE = {(e_sph - e_raw) * 1000:+.3f} mHa')

    t0 = time.time()
    # the CASCI legitimacy row on the STORED bases (certificate-verified
    # column sets; the fresh Aufbau occ is integer-valued and would need
    # a different partition — the stored bases are the ones all our L^2
    # analyses used)
    mf_r = scf.ROHF(mol)
    mf_r.mo_coeff = d_raw['mo_coeff']
    mf_r.mo_occ = d_raw['mo_occ']
    mf_r.e_tot = 0.0
    mf_r.converged = True
    _, e_ci_raw = casci_run(mol, mf_r, [int(c) for c in
                                        d_raw['act_cols']],
                            (NA, NB), NCORE, 1)
    mf_s = scf.ROHF(mol)
    mf_s.mo_coeff = d_sph['mo_coeff']
    mf_s.mo_occ = d_sph['mo_occ']
    mf_s.e_tot = 0.0
    mf_s.converged = True
    _, e_ci_sph = casci_run(mol, mf_s, [int(c) for c in
                                        d_sph['act_cols']],
                            (NA, NB), NCORE, 1)
    e_ci_raw = float(np.atleast_1d(e_ci_raw)[0])
    e_ci_sph = float(np.atleast_1d(e_ci_sph)[0])
    print(f'E_CASCI(raw basis)  = {e_ci_raw:.8f}')
    print(f'E_CASCI(sph basis)  = {e_ci_sph:.8f}   '
          f'dE = {(e_ci_sph - e_ci_raw) * 1000:+.3f} mHa '
          f'({time.time() - t0:.0f} s)')

    e_roots = np.atleast_1d(d60['e_roots'])
    gap_5I_5L = float(e_roots[13] - e_roots[0])
    print(f'term-gap yardstick: E(5L)-E(5I) = {gap_5I_5L * 1000:.2f} mHa')
    print(f'dE_CASCI / term gap = '
          f'{abs(e_ci_sph - e_ci_raw) / gap_5I_5L:.3f}; '
          f'dE_ROHF / term gap = '
          f'{abs(e_sph - e_raw) / gap_5I_5L:.3f}')
    out['energy'] = dict(e_raw_density=e_raw, e_sph_density=e_sph,
                         dE_mHa=(e_sph - e_raw) * 1000,
                         e_casci_raw=e_ci_raw, e_casci_sph=e_ci_sph,
                         dE_casci_mHa=(e_ci_sph - e_ci_raw) * 1000,
                         term_gap_5I_5L_mHa=gap_5I_5L * 1000)

    # ---------- item 1a: electron-number checks on the stored roots ---
    mo = d60['mo_coeff']
    act = [int(c) for c in d60['act_cols']]
    cis60 = [np.asarray(c) for c in d60['cis']]
    matsN_used = False
    devs = []
    for k in (0, 1, 13, 30, 59):
        c = cis60[k]
        d1a, d1b = fci.direct_spin1.make_rdm1s(c, len(act), (NA, NB))
        tra, trb = float(np.trace(d1a).real), float(np.trace(d1b).real)
        n1, n2, n1x, n2x = n_expectation(mol, mo, act, NA, NB, c)
        r = L2Lz_expectation_dm(mol, mo, act, c, (NA, NB),
                                core_cols=list(range(NCORE)),
                                mo_occ=d_sph['mo_occ'])
        dev = max(abs(tra - NA), abs(trb - NB), abs(n1 - n1x),
                  abs(n2 - n2x), abs(float(r['N2']) - n2x))
        devs.append(dev)
        print(f'root {k:2d}: Tr dm1 = {tra:.6f}/{trb:.6f} (exact '
              f'{NA}/{NB}); <N> op-route = {n1:.6f} (exact {n1x}); '
              f'<N^2> op = {n2:.6f}, dm = {float(r["N2"]):.6f} '
              f'(exact {n2x}); max dev = {dev:.1e}')
    out['n_checks_max_dev'] = float(max(devs))

    # ---------- item 1b: CASSCF stationarity ----------
    print('\nss-CASSCF(6,13) stationarity:')
    stat = {}
    mf_sph = scf.ROHF(mol)
    mf_sph.mo_coeff = d_sph['mo_coeff']
    mf_sph.mo_occ = d_sph['mo_occ']
    mf_sph.e_tot = 0.0
    mf_sph.converged = True
    for conv in (1e-7, 1e-9):
        mc = mcscf.CASSCF(mf_sph, len(act), (NA, NB))
        mc.ncore = NCORE
        mc.conv = conv
        mc.max_cycle = 200
        mc.kernel(mf_sph.mo_coeff)
        g = None
        try:
            g = mc.get_grad()
            gnorm = float(np.linalg.norm(np.asarray(g).ravel()))
        except Exception as e:
            gnorm = None
        print(f'  conv = {conv:.0e}: converged = {mc.converged}, '
              f'E = {mc.e_tot:.8f}, |orb grad| = '
              f'{gnorm if gnorm is not None else "n/a"}')
        stat[f'conv{conv:.0e}'] = dict(
            converged=bool(mc.converged), e=float(mc.e_tot),
            grad_norm=gnorm)
    stat['known_stationary_points'] = [
        -50.43993256790306, -50.454807312211194, -50.459443]
    out['stationarity'] = stat

    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/u14e_energy_ncheck.meta.json'), 'w'), indent=1)
    ok = max(devs) < 1e-6 and all(
        v['grad_norm'] is not None and v['grad_norm'] < 1e-4
        for v in stat.values() if isinstance(v, dict)
        and 'grad_norm' in v)
    print(f'\nstored: u14e_energy_ncheck.meta.json; '
          f'checks {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
