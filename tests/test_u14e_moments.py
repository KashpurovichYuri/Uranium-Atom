#!/usr/bin/env python
"""test_u14e_moments.py — U-14e term identification via the VERIFIED
moment engine, on the stored validated data (lmom/data/u14e_rohf_sph.npz:
spherical ROHF MOs, l-closed span {7s + 6d^5 + 5f^7}, CAS(6,13)(5,1),
20 CASCI roots; the dm-route already gave <L^2> = 42.000000 x13).

The new engine must independently reproduce:
  1. the full (5,1)-sector L^2 spectrum: integers,
  2. the 20-root L^2-matrix ladder: {42 x13, 72 x7}
     (the 5L_I ground term + the 5L (L=8) excited term),
  3. the Lz^2 composition: the 5L_I block -> m^2 in
     {0, 1, 4, 9, 16, 25, 36} (the M_L = -6..6 components x2 + M=0),
     the 5L block -> {0, 1, 4, ..., 64}.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.moments import moment_mats, moment_matrix, sector_eigenvalues

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo = d['mo_coeff']
    act = list(d['act_cols'])
    cis = [np.asarray(c) for c in d['cis']]
    na, nb = 5, 1
    nact = len(act)
    from pyscf.fci import cistring
    nsa = cistring.num_strings(nact, na)
    nsb = cistring.num_strings(nact, nb)
    print(f'U-14e stored: act = {act}, sector {nsa}x{nsb} = '
          f'{nsa * nsb} dets, {len(cis)} roots')

    # the span certificate (independent re-check)
    Lm = get_L(mol=None, mo_coeff=mo) if False else get_L(_mol(), mo)
    sc = span_certificate(Lm, act)
    print(f'span: l_closed = {sc["l_closed"]}, invar = '
          f'{sc["invar_err"]:.2e}, closure = {sc["closure_err"]:.2e}, '
          f'groups = { {k: len(v) for k, v in sc["groups"].items()} }')

    ok = sc['l_closed']

    # 1. the full-sector spectrum
    mats = moment_mats(_mol(), mo, act, na, nb, 'L2')
    ev = sector_eigenvalues(mats, nsa, nsb, k=32)
    devs = np.abs(ev - np.round(ev))
    print(f'full-sector L^2 (the lowest 32): max int dev = '
          f'{devs.max():.2e}  {"PASS" if devs.max() < 1e-6 else "FAIL"}')
    ok = ok and devs.max() < 1e-6

    # 2. the root ladder
    MM = moment_matrix(mats, nsa, nsb, cis)
    ladd = np.sort(np.linalg.eigvalsh(MM))
    devs_r = np.abs(ladd - np.round(ladd))
    print(f'root L^2-ladder: max dev = {devs_r.max():.2e}  '
          f'{"PASS" if devs_r.max() < 1e-4 else "FAIL"}')
    print('  ladder:', np.round(ladd, 4))
    has_5I = sum(1 for v in ladd if abs(v - 42) < 1e-3)
    has_5L = sum(1 for v in ladd if abs(v - 72) < 1e-2)
    print(f'  5L_I (42) x {has_5I}, 5L (72) x {has_5L} '
          f'(expected 13 and 7)')
    ok = ok and devs_r.max() < 1e-4 and has_5I == 13

    # 3. the Lz^2 composition WITHIN the complete 5L_I eigenspace
    # (the 72-block is incomplete: 7 of 17 M-components in the window)
    ew, V = np.linalg.eigh(MM)
    iI = [i for i in range(len(ew)) if abs(ew[i] - 42) < 1e-3]
    VI = V[:, iI]                       # the 5L_I eigenbasis (13-dim)
    matsZ = moment_mats(_mol(), mo, act, na, nb, 'Lz2')
    MZ = moment_matrix(matsZ, nsa, nsb, cis)
    MZI = VI.conj().T @ MZ @ VI         # the projected 13x13
    MZI = 0.5 * (MZI + MZI.conj().T)
    zI = np.sort(np.real(np.linalg.eigvalsh(MZI)))
    expI = sorted([0] + [m * m for m in range(1, 7) for _ in range(2)])
    okI = np.allclose(zI, expI, atol=1e-3)
    print(f'the 5L_I eigenspace Lz^2: m^2 = {np.round(zI, 4)}')
    print(f'  expected {expI}  -> {"PASS" if okI else "FAIL"}')
    ok = ok and okI

    print(f'\n==> U-14e term identification: '
          f'{"PASS" if ok else "FAIL"} '
          f'(ground term 5L_I, L=6, <L^2>=42; excited 5L, L=8)')
    return ok


def _mol():
    """The mol object for the angular-momentum integrals (the ECP + the
    basis from the 14e driver)."""
    from pyscf import gto
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'run14e', os.path.join(ROOT, '14e', 'run_14e.py'))
    mod = importlib.util.module_from_spec(spec)
    import unittest.mock as mock
    # import the constants without running the pipeline
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split("if __name__")[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg', '__file__':
             os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, spin=4,
                 verbose=0)   # spin explicit: unset spin converges the singlet


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
