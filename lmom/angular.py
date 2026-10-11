"""The shared primitive: angular-momentum matrices in an MO basis.

l never mixes l-shells (it is angular-only, radial-preserving), which is
the foundation of every theorem in this package. Verified: lz eigenvalues
{-1, 0, +1} on a single p shell; AO SO(3) algebra [Lx,S^-1,Ly] = iLz to
machine precision.
"""
import numpy as np


def get_L(mol, mo_coeff):
    """L = -i (r x grad) in the given MO basis; (3, nmo, nmo)."""
    mol.set_common_origin([0.0, 0.0, 0.0])
    L_ao = -1j * mol.intor('int1e_cg_irxp')
    return np.einsum('pi,xpq,qj->xij', mo_coeff.conj(), L_ao, mo_coeff)


def l2_op(L):
    """The one-electron l^2 operator, complete intermediates:
    sum_k L[k] @ L[k] over ALL MO columns (never over a restricted slice —
    radial mixtures leak into virtual same-l shells and slice products
    lose that leakage)."""
    return sum(L[k] @ L[k] for k in range(3))
