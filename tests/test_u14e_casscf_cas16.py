#!/usr/bin/env python
"""test_u14e_casscf_cas16.py — MECHANISM TEST (item 2): does putting the
7p^3 INSIDE the CI space remove the span-breaking drive?

Channel-2 hypothesis: the CASSCF drift on U-14e is polarization-driven
span breaking — the optimizer steals virtual 7p into the active columns
(population evidence: ss carries ~1.0 e, SA-13 ~3.0 e of p; core-6p
overlap ~0.00) because the 13-column CAS has no p functions for the CI.
Prediction: in CAS(6,16) = 7s + 6d5 + 5f7 + 7p3 the stealing loses its
purpose -> the span drift collapses.

Measured per variant (ss, SA-13), 13-col vs 16-col:
  E, converged, |orb grad|, span invar, purity max, ladder int dev,
  ss <L^2> (83.74 baseline at 13 cols), p-population of the active
  columns, natural occupancies (config label) of the ground root.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.moments import moment_mats, moment_matrix, moment_expectation
from lmom.diag import lowdin_shell_pops, pops_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4
P7 = [5, 6, 7]


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def analyze(tag, mol, mo, act, cis, t0):
    L = get_L(mol, mo)
    sc = span_certificate(L, act)
    nact = len(act)
    nsa = cistring.num_strings(nact, NA)
    nsb = cistring.num_strings(nact, NB)
    mats = moment_mats(mol, mo, act, NA, NB, 'L2')
    MM = moment_matrix(mats, nsa, nsb, [np.asarray(c) for c in cis])
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = np.abs(lad - np.round(lad)).max()
    q0 = moment_expectation(mats, nsa, nsb, cis[0])
    pops = lowdin_shell_pops(mol, mo, act)
    p_tot = float(sum(p[1] for p in pops))
    # natural occupancies of the ground root (config label)
    d1a, d1b = fci.direct_spin1.make_rdm1s(np.asarray(cis[0]), nact,
                                           (NA, NB))
    occ = np.sort(np.diag(d1a).real + np.diag(d1b).real)[::-1]
    print(f'    [{tag}] converged E/cis: span invar = '
          f'{sc["invar_err"]:.2e} (l_closed {sc["l_closed"]}), purity '
          f'p-pop = {p_tot:.2f} e')
    print(f'    [{tag}] ladder dev = {dev:.2e}; <L^2>(root0) = '
          f'{q0:.4f}; nroots = {len(cis)}')
    print(f'    [{tag}] nat occ (root0): '
          f'{np.array2string(occ, precision=2, max_line_width=200)}')
    return dict(invar=float(sc['invar_err']),
                l_closed=bool(sc['l_closed']),
                ladder_dev=float(dev), l2_root0=float(q0),
                p_pop=p_tot,
                nat_occ=[float(x) for x in occ])


def casscf_run(mf, nact, nroots, weights):
    mc = mcscf.CASSCF(mf, nact, (NA, NB))
    mc.ncore = NCORE
    mc.conv = 1e-7
    mc.max_cycle = 100
    mc.nroots = nroots
    mc.fcisolver.nroots = nroots
    if weights is not None:
        mc = mcscf.state_average_(mc, weights)
    mc.kernel(mf.mo_coeff)
    cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    g = None
    try:
        g = float(np.linalg.norm(np.asarray(mc.get_grad()).ravel()))
    except Exception:
        pass
    return mc, [np.asarray(c) for c in cis], g


def main():
    t00 = time.time()
    mol = build_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    act13 = [int(c) for c in d['act_cols']]
    act16 = sorted(act13 + P7)
    mf0 = scf.ROHF(mol)
    mf0.mo_coeff = d['mo_coeff']
    mf0.mo_occ = d['mo_occ']
    mf0.e_tot = 0.0
    mf0.converged = True

    baseline13 = dict(ss=dict(invar=1.21, l2=83.739, p_pop=1.027,
                              ladder_dev=2.60e-1, e=-50.454807),
                      sa13=dict(invar=1.73, p_pop=2.998,
                                ladder_dev=3.43e-1, e=-50.427438))
    out = dict(baseline13=baseline13, cas16={})

    for tag, nroots in (('ss', 1), ('SA-13', 13)):
        t0 = time.time()
        w = [1.0 / nroots] * nroots if nroots > 1 else None
        mc, cis, gnorm = casscf_run(mf0, len(act16), nroots, w)
        e = float(np.atleast_1d(mc.e_tot).ravel()[0])
        print(f'\n=== {tag} in CAS(6,16): converged = {mc.converged}, '
              f'E = {e:.6f} (13-col baseline '
              f'{baseline13["ss" if nroots == 1 else "sa13"]["e"]:.6f}), '
              f'|grad| = {gnorm} ({time.time() - t0:.0f} s)')
        r = analyze(tag, mol, mc.mo_coeff, act16, cis, t0)
        r.update(e=e, converged=bool(mc.converged), grad_norm=gnorm)
        out['cas16'][tag] = r

    print('\n=============== MECHANISM TEST SUMMARY ===============')
    b = baseline13
    for tag in ('ss', 'SA-13'):
        r = out['cas16'][tag]
        bl = b['ss'] if tag == 'ss' else b['sa13']
        print(f'  {tag:6s}: invar {bl["invar"]:.2f} -> '
              f'{r["invar"]:.2e} | p-pop {bl["p_pop"]:.2f} -> '
              f'{r["p_pop"]:.2f} | ladder dev {bl["ladder_dev"]:.1e} -> '
              f'{r["ladder_dev"]:.1e}'
              + (f' | <L2> {bl["l2"]:.2f} -> {r["l2_root0"]:.2f}'
                 if tag == 'ss' else ''))
    print('=======================================================')
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/u14e_casscf_cas16.meta.json'), 'w'), indent=1)
    print(f'stored: u14e_casscf_cas16.meta.json '
          f'({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
