#!/usr/bin/env python
"""test_u14e_verify.py — B+ verification: the analytic trace invariants
against the full dense spectrum of the U-14e sector L^2 matrix.

  Tr W   (analytic from E1)  =  sum(all eigenvalues)
  Tr W^2 (Frobenius)         =  sum(all eigenvalues^2)
  stored 60-root ladder      ==  the lowest 60 of the fresh spectrum

SOC/NIST caveat (recorded in the meta): our terms are LS (spin-free);
NIST levels are J-resolved. Comparisons of term ORDERING must use NIST
centers of gravity (average over the J-levels of each term), since
5f SOC splittings (1000s of cm^-1) are comparable to term gaps.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf
from pyscf.fci import cistring
from lmom.moments import (moment_mats, sector_matrix, trace_moment,
                           trace_moment2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB = 5, 1


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def main():
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_roots60.npz'),
                allow_pickle=True)
    mo = d['mo_coeff']
    act = [int(c) for c in d['act_cols']]
    nact = len(act)
    nsa = cistring.num_strings(nact, NA)
    nsb = cistring.num_strings(nact, NB)
    mol = build_mol()

    mats = moment_mats(mol, mo, act, NA, NB, 'L2')
    tr_analytic = trace_moment(mats, nsa, nsb)
    print(f'Tr W (analytic)     = {tr_analytic:.8f}')

    t0 = time.time()
    W = sector_matrix(mats, nsa, nsb)
    W = 0.5 * (W + W.conj().T)
    tr_frob2 = float(np.real(np.sum(W.conj() * W)))
    print(f'Tr W^2 (Frobenius)  = {tr_frob2:.6f}  '
          f'({time.time() - t0:.0f} s to build W)')

    t0 = time.time()
    ev = np.linalg.eigvalsh(W)
    print(f'dense eigvalsh: {len(ev)} eigenvalues '
          f'({time.time() - t0:.0f} s)')
    tr_eig = float(ev.sum())
    tr2_eig = float((ev ** 2).sum())
    dev1 = abs(tr_analytic - tr_eig)
    dev2 = abs(tr_frob2 - tr2_eig)
    print(f'sum(eig)            = {tr_eig:.8f}   dev = {dev1:.2e}')
    print(f'sum(eig^2)          = {tr2_eig:.4f}   dev = {dev2:.2e}')
    dev_full = np.abs(ev - np.round(ev)).max()
    print(f'full-spectrum integrality: max dev = {dev_full:.2e}')

    # the stored ladder = the CASCI ROOT-WINDOW L^2 eigenvalues; every
    # one must APPEAR in the full spectrum (membership, not position:
    # the full spectrum contains additional low-L states from excited
    # configurations below the root window)
    lad_stored = np.sort(np.asarray(d['ladder']))
    devs_m = [float(np.abs(ev - v).min()) for v in lad_stored]
    dev_l = max(devs_m)
    print(f'stored root-window ladder ({len(lad_stored)} values): each '
          f'present in the full spectrum, max deviation = {dev_l:.2e}')

    ok = dev1 < 1e-6 and dev2 < 1e-4 and dev_full < 1e-6 and dev_l < 1e-4
    meta = dict(system='u14e_verify',
                trace=dict(analytic=tr_analytic, from_eig=tr_eig,
                           dev=dev1),
                trace2=dict(frobenius=tr_frob2, from_eig=tr2_eig,
                            dev=dev2),
                full_spectrum_max_int_dev=float(dev_full),
                stored_ladder_dev=float(dev_l),
                soc_nist_note='LS terms vs J-resolved NIST levels: '
                              'compare via centers of gravity',
                verdict='PASS' if ok else 'FAIL')
    json.dump(meta, open(os.path.join(ROOT,
              'lmom/data/u14e_verify.meta.json'), 'w'), indent=1)
    print(f'\n==> verification: {"PASS" if ok else "FAIL"} '
          f'(stored to lmom/data/u14e_verify.meta.json)')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
