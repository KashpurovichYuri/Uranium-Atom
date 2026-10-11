#!/usr/bin/env python
"""tests/battery_lib.py — shared helpers for the verification rows.

- n_expectation: the NUMBER OPERATOR route through the same E1
  machinery (lmat = 1): <N> and <N^2> on a sector vector; exact
  values are N = na+nb, N^2 = N^2 (fixed-N states).
- n_checks_dm: pyscf dm-route per-root particle number (Tr dm1a/b).
- energy_of_density: the total energy of a stored MO/occ pair
  (one Fock build; for SCF-converged MOs reproduces e_tot).
- casci_run: uniform CASCI wrapper.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import scf, mcscf, fci
from pyscf.fci import cistring
from lmom.moments import e1_matrix


def n_expectation(mol, mo, cols, na, nb, psi):
    """<N> and <N^2> for a sector vector via the number-operator
    matrices (E1 of the identity). Returns (n1, n2, n1_exact, n2_exact)."""
    nact = len(cols)
    nsa = cistring.num_strings(nact, na)
    nsb = cistring.num_strings(nact, nb)
    # N = sum_p a+_p a_p  ->  the coefficient matrix is the IDENTITY
    # (ones would give sum_pq a+_p a_q, which is NOT N)
    A = e1_matrix(np.eye(nact), nact, na)
    B = e1_matrix(np.eye(nact), nact, nb) if nb > 0 \
        else np.zeros((1, 1))

    def apply_N(V):
        out = A @ V
        if nb > 0:
            out = out + V @ B.T
        return out

    V = np.asarray(psi).reshape(nsa, nsb)
    W = apply_N(V)
    n1 = float(np.real(np.vdot(V, W)))
    # <N^2> = <psi| N (N psi) >  -- NOT <N psi| N |N psi> (= N^3)
    n2 = float(np.real(np.vdot(V, apply_N(W))))
    return n1, n2, na + nb, (na + nb) ** 2


def n_checks_dm(ci, nelecas):
    """dm-route particle numbers: returns (Tr dm1a, Tr dm1b, N)."""
    d1a, d1b = fci.direct_spin1.make_rdm1s(np.asarray(ci),
                                           _ncas_of(ci, nelecas), nelecas)
    return float(np.trace(d1a).real), float(np.trace(d1b).real), \
        nelecas[0] + nelecas[1]


def _ncas_of(ci, nelecas):
    return np.asarray(ci).shape[0]


def energy_of_density(mol, mo, occ, hf_like='ROHF'):
    """The total energy of the density built from (mo, occ) — for an
    SCF-converged pair this reproduces mf.e_tot; for post-processed
    MOs (e.g. spherical_reference output) it measures the energy cost
    of the rotation."""
    mf = scf.ROHF(mol)
    mo = np.asarray(mo)
    occ = np.asarray(occ)
    dm = (mo * occ) @ mo.T          # the explicit total density
    e_elec = mf.energy_elec(dm=dm, h1e=mf.get_hcore(),
                            vhf=mf.get_veff(mol, dm))
    return float(e_elec[0] + mol.energy_nuc())


def casci_run(mol, mf, act, nelecas, ncore, nroots, conv_tol=None,
              mo=None):
    mci = mcscf.CASCI(mf, len(act), nelecas)
    mci.ncore = ncore
    mci.nroots = nroots
    mci.fcisolver.nroots = nroots
    if conv_tol:
        mci.fcisolver.conv_tol = conv_tol
        mci.fcisolver.max_cycle = 300
    base = mf.mo_coeff if mo is None else mo
    mci.kernel(mci.sort_mo([c + 1 for c in act]))
    cis = mci.ci if isinstance(mci.ci, list) else [np.asarray(mci.ci)]
    return [np.asarray(c) for c in cis], np.atleast_1d(mci.e_tot)


def mo_passport(mol, mo, occ=None, act=None, core=None, nelec=None,
                expect_l=None, label=''):
    """Verify and print MO bookkeeping at a stage hand-off.
      (a) orthonormality:  max |mo^T S mo - I|
      (b) occupation sum:  sum(occ) vs nelec (if both given)
      (c) active-column l^2-diagonals vs the expected l values
          (expect_l: list of l ints aligned with act)
      (d) core occupancy consistency (|occ - 2| on core columns)
    Returns the passport dict. Prints PASS/FAIL per check."""
    from lmom.angular import get_L
    mo = np.asarray(mo)
    S = mol.intor('int1e_ovlp')
    G = mo.T @ S @ mo
    orth = float(np.abs(G - np.eye(G.shape[0])).max())
    p = dict(orthonorm=orth)
    msgs = [f'orth={orth:.1e}']
    ok = orth < 1e-8
    if occ is not None and nelec is not None:
        s = float(np.sum(occ))
        p['occ_sum'] = s
        msgs.append(f'sum(occ)={s:.3f} (expect {nelec})')
        ok = ok and abs(s - nelec) < 1e-6
    if occ is not None and core:
        cdev = float(max(abs(occ[c] - 2.0) for c in core))
        p['core_occ_dev'] = cdev
        msgs.append(f'core |occ-2|={cdev:.1e}')
        ok = ok and cdev < 1e-6
    if act and expect_l is not None:
        L = get_L(mol, mo)
        l2 = np.diag(sum(k @ k for k in L)).real
        targets = np.array([l * (l + 1) for l in range(7)])
        devs = [float(np.min(np.abs(targets - l2[c]))) for c in act]
        p['act_l_dev_max'] = float(max(devs))
        msgs.append(f'act l-dev max={max(devs):.1e}')
        ok = ok and max(devs) < 1e-6
    print(f'  [passport {label}] {" ".join(msgs)} -> '
          f'{"PASS" if ok else "FAIL"}')
    p['ok'] = ok
    return p


def no_audit(mol, mo, ci, nelecas, ncore, ncas, label='', lo=0.02,
             hi=1.98):
    """Full-space natural-occupancy audit of a CI solution.
    Builds the full-MO 1-RDM, diagonalizes it (the natural occupations)
    and reports:
      n_imp      — # occupations in (lo, hi): the 'important' NOs
      top_below  — the largest occupation below lo (how dead are the
                   dead columns?)
      min_above  — the smallest occupation above hi (core drain?)
    vs the space size (ncore + ncas). The missing-orbital question is
    answered by n_imp vs the space: n_imp approaching the space size
    means the CAS is saturated with important NOs."""
    from pyscf import fci
    mo = np.asarray(mo)
    ci = np.asarray(ci)
    nmo = mo.shape[1]
    S = mol.intor('int1e_ovlp')
    d1a, d1b = fci.direct_spin1.make_rdm1s(ci, ncas, nelecas)
    D = np.zeros(nmo)
    D[:ncore] = 2.0
    D[ncore:ncore + ncas] = np.diag(d1a).real + np.diag(d1b).real
    dm = (mo * D) @ mo.T
    # natural occupations = eigenvalues of the density OPERATOR in an
    # orthonormal basis. The AO-basis dm is a contravariant matrix:
    # its operator matrix in the Lowdin OAO basis is C^-1 dm C^-T
    # (NOT the congruence C^T dm C, which is not metric-correct when
    # the MOs are only S-orthonormal).
    s, U = np.linalg.eigh(S)
    Cinv = U @ np.diag(np.sqrt(s)) @ U.T          # S^{+1/2}
    dm_o = Cinv @ dm @ Cinv.T
    eig = np.linalg.eigvalsh(0.5 * (dm_o + dm_o.T))[::-1]
    n_imp = int(((eig > lo) & (eig < hi)).sum())
    below = eig[eig <= lo]
    above = eig[eig >= hi]
    top_below = float(below.max()) if below.size else 0.0
    min_above = float(above.min()) if above.size else 2.0
    print(f'  [no-audit {label}] space = {ncore}+{ncas}; important NOs '
          f'= {n_imp}; top below-band = {top_below:.2e}; min above-band '
          f'= {min_above:.4f}; spectrum head = '
          f'{np.array2string(eig[:min(12, nmo)], precision=3)}')
    return dict(n_imp=n_imp, space=ncore + ncas,
                top_below=top_below, min_above=min_above,
                spectrum=[float(x) for x in eig])


def lz_check(mol, mo, cols, label=''):
    """<Lz> per column: must be integers (the m-quantum numbers) after
    the m-separability step. Returns (max integer deviation, m values)."""
    from lmom.angular import get_L
    L = get_L(mol, mo)
    lz = L[2]
    mvals = np.real(np.diag(lz[np.ix_(cols, cols)]))
    dev = float(np.abs(mvals - np.round(mvals)).max())
    print(f'  [lz-check {label}] max |<Lz> - int| = {dev:.1e}; '
          f'm = {np.round(mvals, 3)}')
    return dev, mvals


def sz_expectation(mol, mo, cols, na, nb, psi):
    """<Sz> via the spin-resolved number operators (E1 of the identity
    per spin): Sz = (N_a - N_b)/2. Exact: (na - nb)/2."""
    nact = len(cols)
    nsa = cistring.num_strings(nact, na)
    nsb = cistring.num_strings(nact, nb)
    A = e1_matrix(np.eye(nact), nact, na)
    B = e1_matrix(np.eye(nact), nact, nb) if nb > 0 \
        else np.zeros((1, 1))
    V = np.asarray(psi).reshape(nsa, nsb)
    out = 0.5 * (A @ V) - 0.5 * (V @ B.T)
    sz = float(np.real(np.vdot(V, out)))
    return sz, (na - nb) / 2.0
