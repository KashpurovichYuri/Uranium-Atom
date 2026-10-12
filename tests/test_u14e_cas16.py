#!/usr/bin/env python
"""test_u14e_cas16.py — CAS(6,16) physics track: extend the validated
13-column span (7s + 6d5 + 5f7) by the virtual 7p^3 (the population
analysis showed the CASSCF drift pulls exactly this p manifold in —
SA-13 carries ~3 electrons of p; the core 6p overlap ~ 0.00).

Span: 16 columns, l in {0,1,2,3} all complete -> l-closed.
Sector (5,1): C(16,5) x C(16,1) = 4368 x 16 = 69 888 dets.

Checks: span certificate; trace identity; full-sector Lanczos
integrality; 100-root ladder; per-energy-cluster term analysis with
(2L+1) completeness; Lz^2 composition for complete clusters.
Goal: physics quality — correlation, the 5I-vs-5L ordering, the
predicted duplicate 5I/5H clusters.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.reference import spherical_reference
from lmom.moments import (moment_mats, moment_matrix, trace_moment,
                           sector_eigenvalues)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4
NROOTS = 100


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, spin=4,
                 verbose=0)   # spin explicit: unset spin converges the singlet


def main():
    t00 = time.time()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo = d['mo_coeff']
    act13 = [int(c) for c in d['act_cols']]
    mol = build_mol()

    # pick the 7p^3: l=1 columns beyond the core (6p = cols 1,2,3),
    # lowest-energy first (energies from the m-averaged Fock)
    mf0 = scf.ROHF(mol)
    mf0.mo_coeff = mo
    mf0.mo_occ = d['mo_occ']
    mf0.e_tot = 0.0
    mf0.converged = True
    _, mo_e = spherical_reference(mf0)
    L = get_L(mol, mo)
    l2 = np.diag(sum(k @ k for k in L)).real
    targets = np.array([l * (l + 1) for l in range(7)])
    l_of = np.array([int(np.argmin(np.abs(targets - v))) for v in l2])
    p_cols = [i for i in range(len(l_of)) if l_of[i] == 1]
    p_cols.sort(key=lambda i: mo_e[i])
    p7 = [c for c in p_cols if c not in (1, 2, 3)][:3]
    act = sorted(act13 + p7)
    print(f'7p^3 picked: {p7}; CAS(6,16) act = {act}')

    sc = span_certificate(L, act)
    print(f'span: l_closed = {sc["l_closed"]}, invar = '
          f'{sc["invar_err"]:.2e}, groups = '
          f'{ {k: len(v) for k, v in sc["groups"].items()} }')

    nact = len(act)
    nsa = cistring.num_strings(nact, NA)
    nsb = cistring.num_strings(nact, NB)
    print(f'sector ({NA},{NB}): {nsa} x {nsb} = {nsa * nsb} dets')

    # CASCI 100 roots
    mci = mcscf.CASCI(mf0, nact, (NA, NB))
    mci.ncore = NCORE
    mci.nroots = NROOTS
    mci.fcisolver.nroots = NROOTS
    mci.fcisolver.conv_tol = 1e-12
    mci.fcisolver.max_cycle = 300
    t0 = time.time()
    mci.kernel(mci.sort_mo([c + 1 for c in act]))
    cis = mci.ci if isinstance(mci.ci, list) else [np.asarray(mci.ci)]
    cis = [np.asarray(c) for c in cis]
    e = np.atleast_1d(mci.e_tot)
    print(f'CASCI {len(cis)} roots done ({time.time() - t0:.0f} s); '
          f'E[0] = {e[0]:.6f} (13-col reference: -50.164365)')

    # the moment battery
    mats = moment_mats(mol, mo, act, NA, NB, 'L2')
    tr = trace_moment(mats, nsa, nsb)
    print(f'Tr W (analytic) = {tr:.4f}')

    t0 = time.time()
    k = 40
    ev = sector_eigenvalues(mats, nsa, nsb, k=k)
    devs = np.abs(ev - np.round(ev))
    print(f'full-sector Lanczos (the lowest {k}): max int dev = '
          f'{devs.max():.2e}  '
          f'{"PASS" if devs.max() < 1e-6 else "FAIL"} '
          f'({time.time() - t0:.0f} s)')

    t0 = time.time()
    MM = moment_matrix(mats, nsa, nsb, cis)
    ew, V = np.linalg.eigh(MM)
    devs_r = np.abs(ew - np.round(ew))
    print(f'{len(cis)}-root L^2-ladder: max int dev = '
          f'{devs_r.max():.2e}  '
          f'{"PASS" if devs_r.max() < 1e-4 else "FAIL"} '
          f'({time.time() - t0:.0f} s)')

    # the energy clusters
    tol = 1e-4
    clusters = [[0]]
    for i in range(1, len(cis)):
        if e[i] - e[i - 1] > tol:
            clusters.append([])
        clusters[-1].append(i)
    matsZ = moment_mats(mol, mo, act, NA, NB, 'Lz2')
    print(f'\nenergy clusters ({len(clusters)}):')
    cl_meta = []
    for ic, cl in enumerate(clusters):
        MMc = MM[np.ix_(cl, cl)]
        lad, Vc = np.linalg.eigh(MMc)
        dev = np.abs(lad - np.round(lad)).max()
        ls = {}
        for v in np.round(lad).astype(int):
            Ln = int(round((np.sqrt(max(v, 0) + 1) - 1)))
            ls[Ln] = ls.get(Ln, 0) + 1
        compl = all(n == 2 * Ln + 1 for Ln, n in ls.items())
        line = (f'  cluster {ic}: roots {cl[0]}..{cl[-1]} ({len(cl)}), '
                f'spread = {e[cl[-1]] - e[cl[0]]:.2e}, L = {ls}, '
                f'int dev = {dev:.1e}, complete: {compl}')
        r = dict(roots=cl, L=ls, dev=float(dev), complete=bool(compl))
        if compl and len(ls) == 1:
            Ln = next(iter(ls))
            MZc = Vc.conj().T @ moment_matrix(matsZ, nsa, nsb,
                                              [cis[i] for i in cl]) @ Vc
            MZc = 0.5 * (MZc + MZc.conj().T)
            zc = np.sort(np.real(np.linalg.eigvalsh(MZc)))
            exp = sorted([0] + [m * m for m in range(1, Ln + 1)
                                for _ in range(2)])
            okz = np.allclose(zc, exp, atol=1e-3)
            line += f', Lz^2 {"PASS" if okz else "FAIL"}'
            r['lz2_ok'] = bool(okz)
        print(line)
        cl_meta.append(r)

    np.savez_compressed(os.path.join(ROOT,
                        'lmom/data/u14e_cas16.npz'),
                        mo_coeff=mo, act_cols=np.array(act),
                        core_cols=np.arange(NCORE), e_roots=e,
                        cis=np.array(cis), ladder=ew)
    meta = dict(system='u14e_cas16', cas='CAS(6,16) = 7s + 6d5 + 5f7 '
                '+ 7p3 (virtual p, from the population evidence)',
                sector_dets=int(nsa * nsb),
                e_root0=float(e[0]),
                trace=float(tr),
                lanczos=dict(k=k, max_int_dev=float(devs.max())),
                ladder_dev=float(devs_r.max()),
                clusters=[dict(roots=c['roots'], L=c['L'],
                               dev=c['dev'], complete=c['complete'],
                               lz2_ok=c.get('lz2_ok'))
                          for c in cl_meta],
                soc_nist_note='LS terms; compare to NIST centers of '
                              'gravity only')
    json.dump(meta, open(os.path.join(ROOT,
              'lmom/data/u14e_cas16.meta.json'), 'w'), indent=1)
    print(f'\nstored: lmom/data/u14e_cas16.npz + meta '
          f'({time.time() - t00:.0f} s)')
    ok = devs.max() < 1e-6 and devs_r.max() < 1e-4
    print(f'==> CAS(6,16): {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
