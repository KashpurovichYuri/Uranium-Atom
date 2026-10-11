#!/usr/bin/env python
"""
carbon_L2.py — validation benchmark for the full-space <L^2>/<Lz> evaluator
(gates for the uranium L^2 implementation; the evaluator is then mirrored
into uranium_pipeline.calc_L_moments).

System: carbon atom, ROHF start.
  CAS(2,3) on frozen ROHF MOs: the full 2p shell (l-closed span) —
      exact 3P within the basis.  Gate B: <Lz> = 0, <L^2> = 2.000000.
  CAS(2,2): 2 of the 3 p MOs — l-insufficient span.  Gate C: <L^2> != 2
      (the multiplet-insufficiency mechanism, demonstrated).
  C+ CAS(1,3): single p electron.  Gate D: <l^2> = 2.000000 exactly.
Gate A: <N^2> = N(N-1) + N exactly (pins the dm2 conventions end-to-end).

Evaluator (to be mirrored in uranium_pipeline.calc_L_moments for U):
  L matrices in the FULL MO space (get_orbital_angular_momentum);
  the CASSCF/CASCI state expanded into the full (C + A) determinant space
  (expand_product_ci); full-space dm1/dm2 from the trusted
  fci.direct_spin1.make_rdm12 (pyscf convention, pinned by gate A);
  contraction:
    <Lk Lk'> = einsum('pq,rs,pqrs->', Lk, Lk', dm2) + Tr(Lk Lk' dm1)
    <L^2>    = sum_k <Lk Lk>          (also = ladder form, cross-checked)
    ladder   = <Lz^2> + (<L+L-> + <L-L+>)/2,  L+/- = Lx +/- i Ly

Note: the CAS(2,3) span stays l-closed only because cc-pvdz carbon has a
single p shell — there are no foreign radials for CASSCF to rotate in.
The uranium relaxed spans are NOT l-closed (7p rotates in), which is why
their <L^2> will come out below the frozen-span value — reported as-is.
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring


def get_orbital_angular_momentum(mol, mo_coeff):
    """L = -i (r x grad) in the given MO basis; (3, nmo, nmo).
    Verified: single p shell (STO-3G C) lz eigenvalues exactly {-1, 0, +1}."""
    mol.set_common_origin([0.0, 0.0, 0.0])
    L_ao = -1j * mol.intor('int1e_cg_irxp')
    return np.einsum('pi,xpq,qj->xij', mo_coeff.conj(), L_ao, mo_coeff)


def expand_product_ci(ncore, ci_act, nelecas):
    """|core det (MOs 0..ncore-1 doubly occupied)> x |ci_act (ncas active
    MOs)> -> full-space CI vector over (ncore + ncas) MOs (pyscf
    direct_spin1 packing: ci[addr_alpha, addr_beta])."""
    ncas = ci_act.shape[0] and int(np.log2(ci_act.shape[0] * 2)) if False else None
    # number of active orbitals from the CI shape:
    ncas = int(np.log2(ci_act.shape[0])) if (ci_act.shape[0] & (ci_act.shape[0] - 1)) == 0 else None
    raise NotImplementedError


def main():
    pass


if __name__ == '__main__':
    main()
