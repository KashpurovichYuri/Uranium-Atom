#!/usr/bin/env python
"""test_carbon_verification.py — THE VERIFICATION OF THE CORRECTED
MOMENT ROUTE (the exact sector-basis construction in lmom/moments.py:
L^2 = (A^a)^2 + (A^b)^2 + 2 A^a A^b via one-string matrices; contains
ALL two-body pair terms, unlike the previous one-body-only route).

Targets (exact, carbon/cc-pvdz):
  gate B: CASCI(2,3)(2,0), 3 roots (the 3P multiplet):
     <L^2> per root = 2 exactly; the (4,2)-sector spectrum: integers.
  gate M: CASCI(2,3)(1,1), 9 roots = the full 2p^2 term ladder:
     <L^2> = {0 x1, 2 x3, 6 x5}; the sector spectrum: integers.
  gate Z: <Lz^2> spectra: integer m^2 values.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.reference import spherical_reference
from lmom.expectation import L2Lz_expectation_dm
from lmom.moments import (moment_mats, apply_moment, sector_matrix,
                           moment_expectation, sector_eigenvalues,
                           moment_matrix)
from lmom.moments import _expand


def is_int(v, tol=1e-6):
    return abs(v - round(v)) < tol


def main():
    ok = {}
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol); mf.kernel()
    mo, _ = spherical_reference(mf)
    cols = [0, 1, 2, 3, 4]      # 2 core + 3 active
    ncore, nact = 2, 3

    # ---------------- gate B: (2,0) sector, 3 roots ---------------------
    m = mcscf.CASCI(mf, nact, (2, 0))
    m.nroots = 3
    m.fcisolver.nroots = 3
    m.kernel(m.sort_mo([2, 3, 4]))
    cisB = m.ci if isinstance(m.ci, list) else [np.asarray(m.ci)]
    cisB = [np.asarray(c) for c in cisB]

    matsB = moment_mats(mol, mo, cols, 4, 2, 'L2')
    n_str = cistring.num_strings(5, 4)
    nb_str = cistring.num_strings(5, 2)
    qB = [moment_expectation(matsB, n_str, nb_str,
                             _expand(nact, ncore, c, (2, 0)))
          for c in cisB]
    evB = sector_eigenvalues(matsB, n_str, nb_str)
    dmb = [L2Lz_expectation_dm(mol, mo, [2, 3, 4], c, (2, 0),
                               core_cols=[0, 1],
                               mo_occ=mf.mo_occ)['L2'] for c in cisB]
    print('--- gate B: (2,0) sector, 3P ---')
    print('    new-route <L^2> per root =', np.round(qB, 8))
    print('    dm-route   <L^2> per root =', np.round(dmb, 8))
    print('    sector spectrum:', np.round(evB, 6))
    ok['B'] = (np.allclose(qB, 2.0, atol=1e-8)
               and np.allclose(dmb, 2.0, atol=1e-8)
               and all(is_int(v) for v in evB))

    # ---------------- gate M: (1,1) sector, 9 roots ---------------------
    m2 = mcscf.CASCI(mf, nact, (1, 1))
    m2.nroots = 9
    m2.fcisolver.nroots = 9
    m2.kernel(m2.sort_mo([2, 3, 4]))
    cisM = m2.ci if isinstance(m2.ci, list) else [np.asarray(m2.ci)]
    cisM = [np.asarray(c) for c in cisM]

    nsa = cistring.num_strings(5, 3)
    matsM = moment_mats(mol, mo, cols, 3, 3, 'L2')
    embM = [_expand(nact, ncore, c, (1, 1)) for c in cisM]
    MM = moment_matrix(matsM, nsa, nsa, embM)
    ladd = np.sort(np.linalg.eigvalsh(MM))
    evM = np.sort(sector_eigenvalues(matsM, nsa, nsa))
    print('--- gate M: (1,1) sector, 9 roots (2p^2 ladder) ---')
    print('    L^2-matrix eigenvalues =', np.round(ladd, 8))
    print('    (target: {0 x1, 2 x3, 6 x5} = 1S, 3P x3, 1D x5)')
    print('    sector spectrum:', np.round(evM, 6))
    ok['M'] = (np.allclose(ladd, [0] + [2] * 3 + [6] * 5, atol=1e-8)
               and all(is_int(v) for v in evM))

    # ---------------- gate Z: Lz^2 --------------------------------------
    matsZ = moment_mats(mol, mo, cols, 3, 3, 'Lz2')
    MZ = moment_matrix(matsZ, nsa, nsa, embM)
    zadd = np.sort(np.linalg.eigvalsh(MZ))
    evZ = np.sort(sector_eigenvalues(matsZ, nsa, nsa))
    print('--- gate Z: Lz^2, (1,1) sector ---')
    print('    Lz^2-matrix eigenvalues =', np.round(zadd, 8))
    print('    (target: {0 x3, 1 x4, 4 x2}: the M_L = 0 (1S, 3P, 1D),')
    print('     3P M=+-1, 1D M=+-1, 1D M=+-2)')
    print('    sector spectrum:', np.round(evZ, 6))
    ok['Z'] = (np.allclose(zadd, [0] * 3 + [1] * 4 + [4] * 2, atol=1e-8)
               and all(is_int(v) for v in evZ))

    print('\n================ VERIFICATION ================')
    for k in ('B', 'M', 'Z'):
        print(f"  {k}: {'PASS' if ok.get(k) else 'FAIL'}")
    print('==============================================\n')
    return all(ok.values())


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
