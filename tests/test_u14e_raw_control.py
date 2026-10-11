#!/usr/bin/env python
"""test_u14e_raw_control.py — the NEGATIVE CONTROL: the same moment
battery on the RAW (Hund-broken, non-spherical) ROHF MOs
(lmom/data/u14e_rohf_raw.npz). The span certificate must FLAG the
reference (l-closure fails) and the L^2-ladder must come out mixed /
non-integer — demonstrating that non-integer results are a REFERENCE
property, not an operator error (the spherical MOs give integers).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate, L2Lz_expectation_dm
from lmom.moments import (moment_mats, moment_matrix,
                           moment_expectation, trace_moment)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def main():
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_raw.npz'),
                allow_pickle=True)
    mo = d['mo_coeff']
    act = [int(c) for c in d['act_cols']]
    cis = [np.asarray(c) for c in d['cis']]
    nact = len(act)
    na, nb = 5, 1
    nsa = cistring.num_strings(nact, na)
    nsb = cistring.num_strings(nact, nb)
    print(f'RAW control: act = {act}, {len(cis)} roots, '
          f'sector {nsa * nsb}')

    mol = build_mol()
    Lm = get_L(mol, mo)
    sc = span_certificate(Lm, act)
    l2diag = np.diag(sum(k @ k for k in Lm)).real
    print(f'span certificate: l_closed = {sc["l_closed"]} '
          f'(FLAGGED = correct for raw), invar = {sc["invar_err"]:.2e}, '
          f'closure = {sc["closure_err"]:.2e}')
    print(f'l-groups: { {k: len(v) for k, v in sc["groups"].items()} }')
    print(f'l^2 diagonals of the active columns: '
          f'{np.round(l2diag[act], 3)}')

    mats = moment_mats(mol, mo, act, na, nb, 'L2')
    MM = moment_matrix(mats, nsa, nsb, cis)
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = np.abs(lad - np.round(lad)).max()
    print(f'root L^2-ladder: max int dev = {dev:.2e} '
          f'({"INTEGER" if dev < 1e-4 else "MIXED — as expected for raw"})')
    print('  ladder:', np.round(lad, 3))
    # the trace identity holds for ANY span (it is an operator identity)
    tr = trace_moment(mats, nsa, nsb)
    tr_l = float(np.real(np.trace(0.5 * (MM + MM.conj().T))))
    print(f'  trace identity: Tr(sector W) = {tr:.4f} (analytic); the '
          f'root-window trace = {tr_l:.4f} (a SUBSET — not equal on a '
          f'broken span)')
    # the dm-route per-root: on a broken span the act-only routes
    # DISAGREE — the disagreement itself is the drift demonstration
    q_dm = [float(L2Lz_expectation_dm(mol, mo, act, c, (na, nb),
                                      core_cols=[0, 1, 2, 3],
                                      mo_occ=d['mo_occ'])['L2'])
            for c in cis[:5]]
    q_new = [moment_expectation(mats, nsa, nsb, c) for c in cis[:5]]
    print(f'  dm-route <L^2> (first 5):  {np.round(q_dm, 3)}')
    print(f'  new-route <L^2> (first 5): {np.round(q_new, 3)}')
    print(f'  dm-vs-new dev = '
          f'{max(abs(a - b) for a, b in zip(q_dm, q_new)):.2e} '
          f'(nonzero = the act-only assumption broken, as expected)')

    flagged = (not sc['l_closed']) and dev > 1e-4
    print(f'\n==> RAW control: {"PASS" if flagged else "UNEXPECTED"} '
          f'(reference flagged, ladder mixed — the negative control '
          f'behaves as designed)')
    json.dump(dict(system='u14e_rohf_raw_control',
                   l_closed=bool(sc['l_closed']),
                   invar=float(sc['invar_err']),
                   ladder_dev=float(dev),
                   ladder=lad.tolist(),
                   verdict='flagged+mixed (expected)'),
              open(os.path.join(ROOT,
                   'lmom/data/u14e_rohf_raw_control.meta.json'), 'w'),
              indent=1)
    return flagged


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
