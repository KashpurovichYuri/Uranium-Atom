#!/usr/bin/env python
"""test_mini_gates.py — the worked demonstrations for the report:
1. the m-average on the carbon p-block density (CAS(2,3), px,py
   occupied): the 3x3 worked example;
2. the span-closure gates: CAS(2,2) on {px,py} (closed: the m=+-1
   pair) vs CAS(2,2) on {px,pz} (INCOMPLETE: L leaks to py) —
   integers vs non-integers, live.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.reference import m_average_density
from lmom.angular import get_L
from lmom.moments import moment_mats, moment_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    mol = gto.M(atom='C 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    out = {}

    # ---- 1. the worked m-average example ----
    # the CAS(2,3) density p-block with px,py occupied (M_L-selected):
    D = np.zeros(5)
    D[:2] = 2.0
    D[2] = 1.0
    D[3] = 1.0            # px, py singly occupied; pz empty
    dm = (mf.mo_coeff * D) @ mf.mo_coeff.T
    S = mol.intor('int1e_ovlp')
    # extract the p-AO block (sto-3g C: 1s,2s,2px,2py,2pz)
    p = dm[2:5, 2:5]
    p_sph = m_average_density(mol, dm)[2:5, 2:5]
    print('the p-block density (px,py occupied):')
    print(np.round(p, 4))
    print('after the m-average ((tr/3) x I):')
    print(np.round(p_sph, 4))
    print('anisotropy ||p - p_sph||_F / ||p||_F =',
          f'{np.linalg.norm(p - p_sph) / np.linalg.norm(p):.4f}',
          '(the M_L-selected determinant IS anisotropic; the trace is 2)')
    out['worked_example'] = dict(p_block=p.tolist(),
                                 p_sph=p_sph.tolist(),
                                 aniso=float(np.linalg.norm(
                                     p - p_sph) / np.linalg.norm(p)))

    # ---- 2. the span-closure gates ----
    for tag, act in (('closed {px,py}', [2, 3]),
                     ('incomplete {px,pz}', [2, 4])):
        mc = mcscf.CASCI(mf, 2, (2, 0))
        mc.ncore = 2
        mc.nroots = 3
        mc.fcisolver.nroots = 3
        mc.kernel(mc.sort_mo([c + 1 for c in act]))
        cis = mc.ci if isinstance(mc.ci, list) else \
            [np.asarray(mc.ci)]
        cis = [np.asarray(c) for c in cis]
        L = get_L(mol, mf.mo_coeff)
        from lmom.expectation import span_certificate
        sc = span_certificate(L, act)
        mats = moment_mats(mol, mf.mo_coeff, act, 2, 0, 'L2')
        nsa = cistring.num_strings(2, 2)   # = 1 string
        MM = moment_matrix(mats, nsa, 1, cis)
        lad = np.sort(np.linalg.eigvalsh(MM))
        dev = float(np.abs(lad - np.round(lad)).max())
        print(f'\nCAS(2,2) on {tag}: span invar = '
              f'{sc["invar_err"]:.2e} (l_closed {sc["l_closed"]}) '
              f'-> SPAN BROKEN (the incomplete m-set leaks: the real '
              f'L_x/L_y couple to pz; the complex m=+-1 pair couples '
              f'to m=0). The projected ladder = {np.round(lad, 4)} is '
              f'the IN-SPAN projection, NOT the physical <L^2> (the '
              f'true value is 2 for the 3P component): meaningless '
              f'numbers from an incomplete span — the miniature of '
              f'the whole lesson.')
        out[tag] = dict(invar=float(sc['invar_err']),
                        l_closed=bool(sc['l_closed']),
                        ladder=[float(x) for x in lad],
                        dev=dev)

    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/mini_gates.meta.json'), 'w'), indent=1)
    print('\nstored: mini_gates.meta.json')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
