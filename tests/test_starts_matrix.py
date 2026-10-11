#!/usr/bin/env python
"""test_starts_matrix.py — CASSCF/CASCI on DIFFERENT INITIAL GUESSES
(U-14e): how the start affects energy, occupancies, and term validity.

Starts (all converged, from 14e/ or lmom/data/):
  A sph-ROHF            the stored spherical reference (green route)
  B raw-ROHF            Hund-broken canonical (negative control)
  C UKS-PBE free-occ    relaxes to 5f^4 (wrong-config control)
  D UKS-PBE-fixocc-CHAR character-locked 7s2 6d1 5f3 (strongest start)
  E UKS-PBE-fixocc-IDX  index-pinned
  F UKS-PBE-fixocc-MOM  MOM-held
Excluded as circular (the initial guess was changed): *_cas*, *rewarm*.

Treatments per start: DIRECT (as-is; UKS -> spin-summed natural
orbitals for the common-orbital requirement) and SYMMETRIZED (one-shot
m-average + re-diagonalization, placement-(ii) semantics: a START
treatment only — the optimizer remains unconstrained).

Methods: CASCI(6,13)(5,1) 13 roots; ss-CASSCF; SA-13 CASSCF
(ss/SA only for B-F; A is the reference CASCI).

Per cell: E, dE vs A, span invar/l_closed, purity, ladder dev,
<L^2>(root0), natural-occupancy config label, passport (orth, sum occ,
act l-dev, min overlap-match).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.reference import spherical_reference
from lmom.diag import purity_devs, lowdin_shell_pops
from lmom.moments import moment_mats, moment_matrix, moment_expectation
from battery_lib import mo_passport, casci_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4
REF_E = -50.16436519


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def uks_to_nos(mol, mo, occ):
    """UKS (2, nao, nmo) + per-spin occ -> common natural orbitals."""
    mo, occ = np.asarray(mo), np.asarray(occ)
    dma = (mo[0] * occ[0]) @ mo[0].T
    dmb = (mo[1] * occ[1]) @ mo[1].T
    dm = dma + dmb
    S = mol.intor('int1e_ovlp')
    # natural occupations = the OPERATOR eigenvalues in the OAO basis:
    # C^{-1} dm C^{-T} (NOT the congruence C^T dm C — same metric
    # lesson as in battery_lib.no_audit)
    s, U = np.linalg.eigh(S)
    C = U @ np.diag(1.0 / np.sqrt(s)) @ U.T       # S^{-1/2}
    Cinv = U @ np.diag(np.sqrt(s)) @ U.T          # S^{+1/2}
    dm_o = Cinv @ dm @ Cinv.T
    w, V = np.linalg.eigh(0.5 * (dm_o + dm_o.T))
    idx = np.argsort(w)[::-1]
    return C @ V[:, idx], w[idx]


def load_start(mol, sid):
    """Returns (mo, occ, tag)."""
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    if sid == 'A':
        return np.asarray(d['mo_coeff']), np.asarray(d['mo_occ']), 'direct'
    fn = {'B': 'rohf.chk',
          'C': 'uks_pbe.chk',
          'D': 'uks_pbe_fixocc_char.chk',
          'E': 'uks_pbe_fixocc_idx.chk',
          'F': 'uks_pbe_fixocc_mom.chk'}[sid]
    mol_chk, rec = scf.chkfile.load_scf(os.path.join(ROOT, '14e', fn))
    mo, occ = rec['mo_coeff'], rec['mo_occ']
    if np.asarray(mo).ndim == 3:          # UKS: -> natural orbitals
        mo, occ = uks_to_nos(mol, mo, occ)
        return np.asarray(mo), np.asarray(occ), 'NOs'
    return np.asarray(mo), np.asarray(occ), 'direct'


def match_to_ref(mol, mo, mo_ref, act_ref, core_ref, min_ov=0.3):
    S = mol.intor('int1e_ovlp')
    Ov = np.abs(np.asarray(mo).conj().T @ S @ np.asarray(mo_ref))
    used, core, act, ovs = set(), [], [], {}
    for tgt, refs in ((core, core_ref), (act, act_ref)):
        for r in refs:
            cand = [(Ov[j, r], j) for j in range(Ov.shape[0])
                    if j not in used]
            o, j = max(cand)
            used.add(j)
            tgt.append(j)
            ovs[j] = float(o)
    return core, act, min(ovs.values())


def main():
    t00 = time.time()
    mol = build_mol()
    d_sph = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                    allow_pickle=True)
    mo_ref = d_sph['mo_coeff']
    act_ref = [int(c) for c in d_sph['act_cols']]
    core_ref = [int(c) for c in d_sph['core_cols']]
    out = {}
    mf0 = scf.ROHF(mol)
    mf0.e_tot = 0.0
    mf0.converged = True

    rows = []
    for sid, sname in (('A', 'sph-ROHF'), ('B', 'raw-ROHF'),
                       ('C', 'UKS-5f4'), ('D', 'UKS-CHAR'),
                       ('E', 'UKS-IDX'), ('F', 'UKS-MOM')):
        mo, occ, tagk = load_start(mol, sid)
        if sid == 'A':
            # the stored occ vector sums to 22 (old-session artifact,
            # never used by any analysis); rebuild the m-averaged pattern
            occ = np.zeros(len(occ))
            occ[:NCORE] = 2.0
            for c in act_ref:
                occ[c] = 2.0 if c == 4 else \
                    (3.0 / 7 if 22 <= c <= 28 else 1.0 / 5)
        for treat in ('direct', 'symm'):
            if treat == 'direct':
                mo_t, occ_t = mo, occ
            else:
                mfd = scf.ROHF(mol)
                mfd.mo_coeff = mo
                mfd.mo_occ = occ
                mfd.e_tot = 0.0
                mfd.converged = True
                mo_t, _ = spherical_reference(mfd)
                occ_t = occ
            key = f'{sid} {sname} {treat}'
            try:
                pp = mo_passport(mol, mo_t, occ_t, None, None, 14,
                                 label=key)
                core_m, act_m, minov = match_to_ref(
                    mol, mo_t, mo_ref, act_ref, core_ref)
                nact = len(act_m)
                nsa = cistring.num_strings(nact, NA)
                nsb = cistring.num_strings(nact, NB)
                mfc = scf.ROHF(mol)
                mfc.mo_coeff = mo_t
                mfc.mo_occ = occ_t
                mfc.e_tot = 0.0
                mfc.converged = True
                cis, e = casci_run(mol, mfc, act_m, (NA, NB), NCORE, 13,
                                   conv_tol=1e-11)
                cis = [np.asarray(c) for c in cis]
                L = get_L(mol, mo_t)
                sc = span_certificate(L, act_m)
                pdev = purity_devs(L, act_m)
                mats = moment_mats(mol, mo_t, act_m, NA, NB, 'L2')
                MM = moment_matrix(mats, nsa, nsb, cis)
                lad = np.sort(np.linalg.eigvalsh(MM))
                dev = np.abs(lad - np.round(lad)).max()
                q0 = moment_expectation(mats, nsa, nsb, cis[0])
                pops = lowdin_shell_pops(mol, mo_t, act_m)
                p_tot = float(sum(p[1] for p in pops))
                occs = np.diag(fci.direct_spin1.make_rdm1s(
                    cis[0], nact, (NA, NB))[0]).real + \
                    np.diag(fci.direct_spin1.make_rdm1s(
                        cis[0], nact, (NA, NB))[1]).real
                row = dict(start=key, e=float(e[0]),
                           dE_mHa=(float(e[0]) - REF_E) * 1000,
                           l_closed=bool(sc['l_closed']),
                           invar=float(sc['invar_err']),
                           purity=float(pdev.max()),
                           ladder_dev=float(dev),
                           l2_root0=float(q0),
                           p_pop=p_tot, min_overlap=float(minov),
                           act_occs=[float(x) for x in occs],
                           passport=pp)
                rows.append(row)
                print(f'{key:26s}: E = {e[0]:.6f} ({(e[0]-REF_E)*1000:+8.2f}'
                      f' mHa) span {sc["l_closed"]} (invar '
                      f'{sc["invar_err"]:.1e}) purity {pdev.max():.1e} '
                      f'ladder {dev:.1e} <L2>r0 {q0:.3f} p-pop {p_tot:.2f} '
                      f'minov {minov:.2f}')
                out[key] = row
            except Exception as exc:
                import traceback
                tb = traceback.format_exc().strip().split('\n')[-1]
                print(f'{key:26s}: FAILED {tb}')
                out[key] = dict(error=tb)
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/starts_matrix_casci.meta.json'), 'w'), indent=1)
    print(f'\nWave 1 stored ({time.time() - t00:.0f} s): '
          f'starts_matrix_casci.meta.json')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
