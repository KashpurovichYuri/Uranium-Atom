#!/usr/bin/env python
"""test_u14e_roots.py — U-14e EXTENDED-ROOT study (nroots = 60) on the
stored spherical ROHF MOs (lmom/data/u14e_rohf_sph.npz).

Physics: the (5,1) sector (M_S = 2) of CAS(6,13) 7s 6d5 5f7 contains
ONLY quintet states (M_S=2 forces S>=2; max S for 5f^3 6d^1 is 2).
With enough roots the full low-lying quintet term ladder must appear:
  5L (L=8 -> 72), 5K (7 -> 56), 5I (6 -> 42) x2 distinct terms,
  5H (5 -> 30) x2/3, ... — DISTINCT energy clusters may share the same
  L^2 value (e.g. two different 5I terms from 4G- and 4I-f^3 coupling).

Checks:
  1. the full-sector L^2 spectrum (Lanczos k=40): integers;
  2. solver cross-checks: matvec-vs-dense consistency, eigsh stability,
     dense eigvalsh lowest-k agreement;
  3. the 60-root L^2-ladder: all integers;
  4. per-energy-cluster analysis: the L^2-matrix eigenvalues per
     cluster = integers; complete multiplets have exactly (2L+1) roots;
  5. Lz^2 composition for COMPLETE clusters (m^2 = {0, 1,1, 4,4, ...}).
Results stored to lmom/data/u14e_roots60.npz + meta.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.moments import (moment_mats, moment_matrix, moment_expectation,
                           sector_eigenvalues, apply_moment)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NROOTS = 60


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def main():
    t00 = time.time()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo = d['mo_coeff']
    act = [int(c) for c in d['act_cols']]
    nact = len(act)
    na, nb = 5, 1
    nsa = cistring.num_strings(nact, na)
    nsb = cistring.num_strings(nact, nb)
    print(f'U-14e roots-{NROOTS} study: act = {act}')
    print(f'sector ({na},{nb}): {nsa} x {nsb} = {nsa * nsb} dets')

    mol = build_mol()
    mf = scf.ROHF(mol)
    mf.mo_coeff = mo
    mf.mo_occ = d['mo_occ']
    mf.e_tot = 0.0
    mf.converged = True

    # the span certificate (re-check on the stored MOs)
    Lm = get_L(mol, mo)
    sc = span_certificate(Lm, act)
    print(f'span: l_closed = {sc["l_closed"]}, invar = '
          f'{sc["invar_err"]:.2e}, closure = {sc["closure_err"]:.2e}')

    # CASCI with 60 roots
    mc = mcscf.CASCI(mf, nact, (na, nb))
    mc.ncore = 4
    mc.nroots = NROOTS
    mc.fcisolver.nroots = NROOTS
    mc.fcisolver.conv_tol = 1e-13   # the sharp roots -> the sharp ladder
    mc.fcisolver.max_cycle = 300
    mc.kernel(mc.sort_mo([c + 1 for c in act]))
    cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    cis = [np.asarray(c) for c in cis]
    e = np.atleast_1d(mc.e_tot)
    print(f'CASCI {len(cis)} roots done ({time.time() - t00:.0f} s); '
          f'E[0] = {e[0]:.6f}, E[-1] = {e[-1]:.6f}')

    # the moment machinery
    t0 = time.time()
    mats = moment_mats(mol, mo, act, na, nb, 'L2')
    matsZ = moment_mats(mol, mo, act, na, nb, 'Lz2')
    print(f'one-string matrices built ({time.time() - t0:.1f} s)')

    # ---- 1+2: the full-sector spectrum with the solver cross-checks ----
    # (a) the matvec-vs-dense consistency on a random vector
    rng = np.random.default_rng(0)
    v = rng.standard_normal(nsa * nsb)
    w_fast = apply_moment(mats, nsa, nsb, v)
    W = None
    t0 = time.time()
    from lmom.moments import sector_matrix
    W = sector_matrix(mats, nsa, nsb)     # 16731^2 dense (~4.5 GB)
    print(f'dense sector matrix built ({time.time() - t0:.0f} s, '
          f'shape {W.shape})')
    w_dense = (W @ v.reshape(nsa, nsb).ravel().astype(W.dtype))
    dev_mv = np.abs(w_fast - np.asarray(w_dense).ravel()).max()
    print(f'matvec consistency |fast - dense| = {dev_mv:.2e}')

    # (b) the dense eigvalsh vs the Lanczos
    t0 = time.time()
    Wsym = 0.5 * (W + W.conj().T)
    ev_dense = np.linalg.eigvalsh(Wsym)
    print(f'dense eigvalsh done ({time.time() - t0:.0f} s)')
    t0 = time.time()
    ev_lan = sector_eigenvalues(mats, nsa, nsb, k=40)
    print(f'Lanczos k=40 done ({time.time() - t0:.1f} s)')
    dev_dd = np.abs(np.sort(ev_dense)[:40] - ev_lan).max()
    devs_full = np.abs(ev_dense - np.round(ev_dense)).max()
    print(f'FULL-sector spectrum ({len(ev_dense)} eigenvalues, dense): '
          f'max int dev = {devs_full:.2e}  '
          f'{"PASS" if devs_full < 1e-6 else "FAIL"}')
    print(f'Lanczos vs dense (the lowest 40): max dev = {dev_dd:.2e}  '
          f'{"PASS" if dev_dd < 1e-6 else "FAIL"}')
    del W, Wsym

    # ---- 3: the 60-root L^2-ladder ----
    MM = moment_matrix(mats, nsa, nsb, cis)
    ew, V = np.linalg.eigh(MM)
    devs_r = np.abs(ew - np.round(ew))
    print(f'\n{NROOTS}-root L^2-ladder: max int dev = {devs_r.max():.2e}  '
          f'{"PASS" if devs_r.max() < 1e-4 else "FAIL"}')
    print('  ladder:', np.round(ew, 4))

    # ---- 4: per-energy-cluster analysis ----
    gaps = np.diff(e)
    tol = 1e-4
    clusters = [[0]]
    for i in range(1, NROOTS):
        if e[i] - e[i - 1] > tol:
            clusters.append([])
        clusters[-1].append(i)
    print(f'\nenergy clusters (gap tol = {tol} Ha): {len(clusters)}')
    results = []
    all_ok = True
    for ic, cl in enumerate(clusters):
        ecs = e[cl]
        MMc = MM[np.ix_(cl, cl)]
        lad, Vc = np.linalg.eigh(MMc)
        dev = np.abs(lad - np.round(lad)).max()
        ls = {}
        for v in np.round(lad).astype(int):
            L = int(round((np.sqrt(max(v, 0) + 1) - 1)))
            ls[L] = ls.get(L, 0) + 1
        # the completeness check: every L block == 2L+1?
        compl = all(n == 2 * L + 1 for L, n in ls.items())
        print(f'  cluster {ic}: roots {cl[0]}..{cl[-1]} ({len(cl)}), '
              f'E spread = {ecs[-1] - ecs[0]:.2e}, L^2 = '
              f'{np.round(lad, 4)}, L = {ls}, '
              f'int dev = {dev:.1e}, complete multiplets: {compl}')
        r = dict(cluster=cl, L=ls, dev=float(dev), complete=compl)
        # the Lz^2 composition only for a COMPLETE single-L cluster
        if compl and len(ls) == 1:
            L = next(iter(ls))
            Vp = Vc.real if np.isclose(Vc.imag, 0).all() else Vc
            MZc = Vp.conj().T @ moment_matrix(matsZ, nsa, nsb,
                                              [cis[i] for i in cl]) @ Vp
            MZc = 0.5 * (MZc + MZc.conj().T)
            zc = np.sort(np.real(np.linalg.eigvalsh(MZc)))
            exp = sorted([0] + [m * m for m in range(1, L + 1)
                                for _ in range(2)])
            okz = np.allclose(zc, exp, atol=1e-3)
            print(f'    Lz^2 in the L={L} eigenspace: '
                  f'{np.round(zc, 3)}  expected {exp}  '
                  f'{"PASS" if okz else "FAIL"}')
            r['lz2'] = zc.tolist()
            r['lz2_ok'] = bool(okz)
        results.append(r)
        all_ok = all_ok and dev < 1e-4

    print(f'\nclusters with all-integer ladders: '
          f'{"PASS" if all_ok else "FAIL"}')
    print(f'total time: {time.time() - t00:.0f} s')

    # ---- store ----
    np.savez_compressed(os.path.join(ROOT, 'lmom/data/u14e_roots60.npz'),
                        mo_coeff=mo, mo_occ=d['mo_occ'],
                        act_cols=np.array(act), core_cols=np.arange(4),
                        e_roots=e,
                        cis=np.array([c for c in cis]),
                        ladder=ew)
    meta = dict(system='u14e_roots60', nroots=NROOTS,
                cas='CAS(6,13) (5,1) on spherical ROHF MOs',
                span=dict(l_closed=bool(sc['l_closed']),
                          invar=float(sc['invar_err'])),
                full_sector=dict(max_int_dev=float(devs_full),
                                 lanczos_vs_dense=float(dev_dd)),
                ladder=ew.tolist(),
                clusters=[dict(roots=r['cluster'], L=r['L'],
                               dev=r['dev'], complete=r['complete'],
                               lz2=r.get('lz2'),
                               lz2_ok=r.get('lz2_ok'))
                          for r in results])
    json.dump(meta, open(os.path.join(ROOT,
              'lmom/data/u14e_roots60.meta.json'), 'w'), indent=1)
    print('stored: lmom/data/u14e_roots60.npz + meta')
    return all_ok and devs_full < 1e-6 and dev_dd < 1e-6


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
