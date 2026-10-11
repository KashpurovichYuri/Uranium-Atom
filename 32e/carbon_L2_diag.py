#!/usr/bin/env python
"""
carbon_L2.py — validation benchmark for the full-space <L^2>/<Lz> evaluator.

Carbon 3P: ROHF -> CAS(2,3) on the (l-pure) ROHF 2p MOs: exact 3P, so the
exact <L^2> = 2.000000, <Lz> = 0 (real state). The product state
(closed-shell core det x active CI) is expanded into the (core+active)
determinant space; its exact spin-orbital RDMs come from fci.make_rdm12s
(trusted). Contraction with the spin-orbital L matrices gives the exact
<L^2> (brute-force reference).

The block-assembled version (for scaling to U) is validated against the
brute force: the spin-orbital 2-RDM blocks
  G[AAAA] = casdm2aa_emb + core det (per spin: P_ik P_jl - P_il P_jk)
  G[BBBB] = same with d2bb
  G[ABBA] = -P_c[i,k] casdm1b[j,l]     (same-spin core-active exchange)
  G[BAAB] = -P_c[i,k] casdm1a[j,l]
  G[ABAB] = +P_c[i,k] casdm1b[j,l]     (opposite-spin core-active direct)
  G[BABA] = +P_c[i,k] casdm1a[j,l]
  G[CCCC] = closed-shell core det: 2(P_ik P_jl - P_il P_jk)... pinned
are compared against the brute-force Gso block by block; the printed diff
drives the final sign/layout fixes.
"""
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring


def get_orbital_angular_momentum(mol, mo_coeff):
    """L = -i (r x grad) in the given MO basis; (3, nmo, nmo). Verified:
    single p shell (STO-3G C) lz eigenvalues exactly {-1, 0, +1}."""
    mol.set_common_origin([0.0, 0.0, 0.0])
    L_ao = -1j * mol.intor('int1e_cg_irxp')
    return np.einsum('pi,xpq,qj->xij', mo_coeff.conj(), L_ao, mo_coeff)


def expand_product_ci(ncore, ncas, ci_act, nelecas):
    """|core det> x |ci_act> -> full-space CI vector over (ncore+ncas) MOs."""
    n = ncore + ncas
    na, nb = ncore + nelecas[0], ncore + nelecas[1]
    core = (1 << ncore) - 1
    ci_full = np.zeros((cistring.num_strings(n, na),
                        cistring.num_strings(n, nb)))
    for ia in range(ci_act.shape[0]):
        sa = int(cistring.addr2str(ncas, nelecas[0], ia))
        for ib in range(ci_act.shape[1]):
            sb = int(cistring.addr2str(ncas, nelecas[1], ib))
            x = core | (sa << ncore)
            y = core | (sb << ncore)
            ci_full[cistring.str2addr(n, na, x),
                    cistring.str2addr(n, nb, y)] = ci_act[ia, ib]
    return ci_full


def main():
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    ncore, ncas = 2, 3
    n = ncore + ncas
    nso = 2 * n

    # CAS(2,3) on the frozen ROHF 2p MOs: exact 3P
    mci = mcscf.CASCI(mf, ncas, (2, 0))
    mci.kernel(mci.sort_mo([3, 4, 5]))
    mo = mci.mo_coeff[:, :n]
    ci_full = expand_product_ci(ncore, ncas, mci.ci, (2, 0))

    # L matrices in spin-orbital form over (C∪A) spin-orbitals
    L_sp = get_orbital_angular_momentum(mol, mo)          # (3, n, n)
    L_so = np.zeros((3, nso, nso))
    for s in range(2):
        L_so[:, s::2, s::2] = L_sp

    # ---------- brute force: exact Gso + exact <L^2> ----------
    dm1so, Gso = fci.direct_spin1.make_rdm12s(ci_full, nso, (6, 0))
    # spin-resolved: d1a, d1b; d2aa, d2ab, d2bb  (pyscf layout a+_i a+_j a_l a_k)
    d1a, d1b = dm1so
    d2aa, d2ab, d2bb = Gso
    Gso_abba = -d2ab.transpose(0, 1, 3, 2)   # a+_ia a+_jb a_kb a_la (swap annih)
    Gso_baab = -d2ab.transpose(1, 0, 2, 1)   # placeholder fix below if needed
    Gso_baba = d2ab.transpose(2, 1, 0, 3) * 0  # unused placeholder

    L2_exact = 0.0
    for k in range(3):
        Lk = L_so[k]
        L2_exact += np.einsum('ij,kl,ikjl->', Lk, Lk, d2aa).real
        L2_exact += np.einsum('ij,kl,ikjl->', Lk, Lk, d2bb).real
        L2_exact += 2 * np.einsum('ij,kl,ikjl->', Lk, Lk, d2ab).real
    print(f'brute-force <L^2> (active RDM contraction) = {L2_exact:.8f}')

    # ---------- Gso blocks from the brute force, for the diff ----------
    print('\nGso block diagnostics (alpha-alpha block, P_core projector):')
    P_c = np.zeros((n, n)); P_c[:ncore, :ncore] = np.eye(ncore)
    gA = P_c + np.zeros((n, n)); gA[ncore:, ncore:] = d1a
    gB = P_c + np.zeros((n, n)); gB[ncore:, ncore:] = d1b
    # the exact A-block 2-RDM: print a few elements vs the P_c det formula
    det_aa = np.einsum('ik,jl->ijkl', P_c, P_c) - np.einsum('il,jk->ijkl', P_c, P_c)
    diff = d2aa - det_aa
    print('  d2aa - P_c-det (max abs):', abs(diff).max())
    # core-active block: <a+_iα a_yβ a_wβ a_kα>: = -P_c[i,k] casdm1b[w,y]?? print
    blk = d2ab[:ncore, ncore:, ncore:, :ncore]   # i in C, j,l in A, k in C
    print('  d2ab core-active diag-ish (i,k,j=l) elements:')
    for ii in range(ncore):
        for jj in range(ncas):
            print(f'   [{ii},{ncore+jj},{ncore+jj},{ii}] = {blk[ii, ncore+jj, ncore+jj, ii]:+.6f}')


if __name__ == '__main__':
    main()
