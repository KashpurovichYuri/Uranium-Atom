#!/usr/bin/env python
"""test_frozen7s_invariants.py — A5 invariant completion for the frozen
7s^2 spaces: FULL dense spectra (both sectors are small: 495 and 1365
dets), the trace identities, the electron-number row, and <Sz>.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, fci
from pyscf.fci import cistring
from lmom.moments import (moment_mats, sector_matrix, trace_moment,
                           trace_moment2, moment_expectation)
from battery_lib import n_expectation, sz_expectation, casci_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P7 = [5, 6, 7]


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
    mo = d['mo_coeff']
    act13 = [int(c) for c in d['act_cols']]
    act12 = [c for c in act13 if c != 4]
    act15 = sorted(act12 + P7)
    mf = scf.ROHF(mol)
    mf.mo_coeff = mo
    mf.e_tot = 0.0
    mf.converged = True
    out = {}
    for tag, act, nelecas, ncore in (('CAS(4,12)', act12, (4, 0), 5),
                                     ('CAS(4,15)', act15, (4, 0), 5)):
        nact = len(act)
        nsa = cistring.num_strings(nact, nelecas[0])
        nsb = cistring.num_strings(nact, nelecas[1])
        cis, e = casci_run(mol, mf, act, nelecas, ncore, 5,
                           conv_tol=1e-12)
        mats = moment_mats(mol, mo, act, nelecas[0], nelecas[1], 'L2')
        W = sector_matrix(mats, nsa, nsb)
        W = 0.5 * (W + W.conj().T)
        ev = np.linalg.eigvalsh(W)
        tr_a = trace_moment(mats, nsa, nsb)
        tr2 = trace_moment2(mats, nsa, nsb)
        dev_tr = abs(tr_a - float(ev.sum()))
        dev_tr2 = abs(tr2 - float((ev ** 2).sum()))
        dev_int = float(np.abs(ev - np.round(ev)).max())
        n1, n2, n1x, n2x = n_expectation(mol, mo, act, nelecas[0],
                                         nelecas[1], cis[0])
        sz, szx = sz_expectation(mol, mo, act, nelecas[0], nelecas[1],
                                 cis[0])
        q0 = moment_expectation(mats, nsa, nsb, cis[0])
        print(f'=== {tag} ({nsa * nsb} dets) ===')
        print(f'  E = {e[0]:.8f}; <L^2>(r0) = {q0:.6f}')
        print(f'  Tr W: analytic {tr_a:.6f} vs sum(eig) {ev.sum():.6f}'
              f'  dev = {dev_tr:.1e}')
        print(f'  Tr W^2: {tr2:.4f} vs {float((ev**2).sum()):.4f}'
              f'  dev = {dev_tr2:.1e}')
        print(f'  full-spectrum integrality: max dev = {dev_int:.1e}')
        print(f'  N: Tr dm1 = 4/0; <N> op = {n1:.8f} (exact {n1x}); '
              f'<N^2> op = {n2:.6f} (exact {n2x})')
        print(f'  <Sz> op = {sz:.8f} (exact {szx})')
        out[tag] = dict(e_root0=float(e[0]), l2_root0=q0,
                        trace_dev=float(dev_tr),
                        trace2_dev=float(dev_tr2),
                        max_int_dev=float(dev_int),
                        n_dev=float(max(abs(n1 - n1x), abs(n2 - n2x))),
                        sz=float(sz), sz_exact=szx)
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/frozen7s_invariants.meta.json'), 'w'), indent=1)
    ok = all(out[t]['trace_dev'] < 1e-8 and out[t]['trace2_dev'] < 1e-6
             and out[t]['max_int_dev'] < 1e-6 and out[t]['n_dev'] < 1e-8
             and abs(out[t]['sz'] - 2.0) < 1e-8
             for t in out)
    print(f'\n==> invariants: {"PASS" if ok else "FAIL"} '
          f'(stored: frozen7s_invariants.meta.json)')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
