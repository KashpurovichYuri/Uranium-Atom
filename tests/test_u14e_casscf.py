#!/usr/bin/env python
"""test_u14e_casscf.py — U-14e orbital-source comparison: which reference
produces term-pure CASCI/CASSCF states? Five variants, one table:

  1. sph-ROHF : Aufbau ROHF -> force_config (m-averaged 5f^3 6d^1 7s^2)
                -> spherical_reference        [the validated recipe]
  2. raw-ROHF : Aufbau ROHF as-is (Hund-broken canonical MOs)
  3. ss-CASSCF: state-specific CASSCF(6,13) (1 root) from the sph start
                — a degenerate multiplet optimized root-wise: expect
                symmetry drift
  4. SA-13    : state-averaged CASSCF over the 5I multiplet (13 roots)
  5. SA-60    : state-averaged CASSCF over 60 roots (multi-multiplet)

Per variant: the span certificate, the L^2-ladder over the roots,
per-root <L^2>, and the l^2-PURITY TABLE (per-MO l^2-diagonal deviation
from the nearest l(l+1)) — visualizing HOW each source breaks (or
keeps) the reference.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.moments import moment_mats, moment_matrix, moment_expectation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB = 5, 1
NCORE = 4


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def force_config(mol, mf, l_occs, core_n):
    """(see test_atoms_moments.py) 2-pass ROHF with index-pinned
    m-averaged occupations: get_occ returns the fixed vector at every
    SCF iteration, so occupations cannot drift during SCF."""
    L = get_L(mol, mf.mo_coeff)
    l2 = np.diag(sum(k @ k for k in L)).real
    targets = np.array([l * (l + 1) for l in range(7)])
    l_of = np.array([int(np.argmin(np.abs(targets - v))) for v in l2])
    occ = np.zeros(len(l_of))
    occ[:core_n] = 2.0
    for l, o in l_occs.items():
        cols = [i for i in range(core_n, len(l_of)) if l_of[i] == l]
        nsh = min(2 * l + 1, len(cols))
        for i in cols[:nsh]:
            occ[i] = o / nsh
    mf2 = scf.ROHF(mol)
    mf2.get_occ = lambda mo_e=None, mo=None: occ
    mf2.kernel()
    if not mf2.converged:
        mf2 = mf2.newton()
        mf2.get_occ = lambda mo_e=None, mo=None: occ
        mf2.kernel()
    mf2.mo_occ = occ
    return mf2


def purity_table(L, cols, label):
    """Per-MO l^2-diagonal deviation from the nearest l(l+1)."""
    l2 = np.diag(sum(k @ k for k in L)).real
    targets = np.array([l * (l + 1) for l in range(7)])
    devs = np.array([np.min(np.abs(targets - l2[c])) for c in cols])
    print(f'    purity[{label}]: max dev = {devs.max():.3e}, '
          f'mean = {devs.mean():.3e}, per-MO = '
          f'{np.array2string(devs, precision=3, max_line_width=200)}')
    return devs


def analyze(name, mol, mo, act, cis, t0, e_ground=None, converged=None):
    """The span certificate + the L^2-ladder + per-root <L^2>."""
    nact = len(act)
    nsa = cistring.num_strings(nact, NA)
    nsb = cistring.num_strings(nact, NB)
    L = get_L(mol, mo)
    sc = span_certificate(L, act)
    devs = purity_table(L, act, name)
    mats = moment_mats(mol, mo, act, NA, NB, 'L2')
    MM = moment_matrix(mats, nsa, nsb, [np.asarray(c) for c in cis])
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev_l = np.abs(lad - np.round(lad)).max()
    q = [moment_expectation(mats, nsa, nsb, c) for c in cis]
    print(f'    [{name}] span: l_closed = {sc["l_closed"]}, invar = '
          f'{sc["invar_err"]:.2e}')
    print(f'    [{name}] ladder int dev = {dev_l:.2e} '
          f'({"INTEGER" if dev_l < 1e-3 else "MIXED"}); '
          f'ladder[:8] = {np.round(lad[:8], 2)}')
    print(f'    [{name}] per-root <L^2> (the first 6): '
          f'{np.round(q[:6], 3)}')
    print(f'    [{name}] done ({time.time() - t0:.0f} s)')
    return dict(name=name, l_closed=bool(sc['l_closed']),
                invar=float(sc['invar_err']),
                purity_max=float(devs.max()),
                purity_mean=float(devs.mean()),
                ladder_dev=float(dev_l),
                ladder=lad.tolist(),
                l2_per_root=[float(x) for x in q],
                e_ground=(float(e_ground) if e_ground is not None
                          else None),
                casscf_converged=(None if converged is None
                                  else bool(converged)))


def casci_on(mf, act, nroots=20, conv_tol=None):
    mc = mcscf.CASCI(mf, len(act), (NA, NB))
    mc.ncore = NCORE
    mc.nroots = nroots
    mc.fcisolver.nroots = nroots
    if conv_tol:
        mc.fcisolver.conv_tol = conv_tol
        mc.fcisolver.max_cycle = 300
    mc.kernel(mc.sort_mo([c + 1 for c in act]))
    cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    return [np.asarray(c) for c in cis], np.atleast_1d(mc.e_tot)


def main():
    t00 = time.time()
    mol = build_mol()
    results = {}

    # BOTH references from the STORED validated data (the fresh Aufbau
    # ROHF ordering is not reproducible run-to-run; the stored MOs are
    # the canonical reference set)
    d_sph = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                    allow_pickle=True)
    d_raw = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_raw.npz'),
                    allow_pickle=True)
    act_sph = [int(c) for c in d_sph['act_cols']]     # 13: 7s + 6d5 + 5f7
    act_raw = [int(c) for c in d_raw['act_cols']]
    print(f'sph-ROHF active: {act_sph}')
    print(f'raw-ROHF active: {act_raw}')

    mf_sph = scf.ROHF(mol)
    mf_sph.mo_coeff = d_sph['mo_coeff']
    mf_sph.mo_occ = d_sph['mo_occ']
    mf_sph.e_tot = 0.0
    mf_sph.converged = True
    mf0 = scf.ROHF(mol)
    mf0.mo_coeff = d_raw['mo_coeff']
    mf0.mo_occ = d_raw['mo_occ']
    mf0.e_tot = 0.0
    mf0.converged = True
    mo_sph = d_sph['mo_coeff']

    # 1. sph-ROHF + CASCI (the reference case)
    t0 = time.time()
    cis, e = casci_on(mf_sph, act_sph, 20, conv_tol=1e-12)
    results['sph-ROHF CASCI'] = analyze('sph-ROHF CASCI', mol, mo_sph,
                                        act_sph, cis, t0)

    # 2. raw-ROHF + CASCI (the negative control, freshly computed)
    t0 = time.time()
    cis, e = casci_on(mf0, act_raw, 20, conv_tol=1e-12)
    results['raw-ROHF CASCI'] = analyze('raw-ROHF CASCI', mol,
                                        mf0.mo_coeff, act_raw, cis, t0)

    # 3. ss-CASSCF from the sph start (state-specific: drift expected)
    t0 = time.time()
    mc = mcscf.CASSCF(mf_sph, len(act_sph), (NA, NB))
    mc.ncore = NCORE
    mc.conv = 1e-7
    mc.max_cycle = 100
    mc.kernel(mf_sph.mo_coeff)
    mo_ss = mc.mo_coeff
    act_ss = list(range(NCORE, NCORE + len(act_sph)))
    print(f'    ss-CASSCF: converged = {mc.converged}, '
          f'E = {mc.e_tot:.6f}')
    results['ss-CASSCF'] = analyze('ss-CASSCF', mol, mo_ss, act_ss,
                                   [np.asarray(mc.ci)], t0,
                                   e_ground=mc.e_tot,
                                   converged=mc.converged)

    # 4. SA-13 CASSCF (the 5I multiplet)
    t0 = time.time()
    mc = mcscf.CASSCF(mf_sph, len(act_sph), (NA, NB))
    mc.ncore = NCORE
    mc.conv = 1e-7
    mc.max_cycle = 100
    n13 = 13
    mc.fcisolver.nroots = n13
    mc = mcscf.state_average_(mc, [1.0 / n13] * n13)
    mc.kernel(mf_sph.mo_coeff)
    mo_sa13 = mc.mo_coeff
    cis13 = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    print(f'    SA-13: converged = {mc.converged}, '
          f'E = {mc.e_tot:.6f}')
    results['SA-13 CASSCF'] = analyze('SA-13 CASSCF', mol, mo_sa13,
                                      act_ss,
                                      [np.asarray(c) for c in cis13], t0,
                                      e_ground=mc.e_tot,
                                      converged=mc.converged)

    # 5. SA-60 CASSCF (multi-multiplet)
    t0 = time.time()
    mc = mcscf.CASSCF(mf_sph, len(act_sph), (NA, NB))
    mc.ncore = NCORE
    mc.conv = 1e-7
    mc.max_cycle = 100
    n60 = 60
    mc.fcisolver.nroots = n60
    mc = mcscf.state_average_(mc, [1.0 / n60] * n60)
    mc.kernel(mf_sph.mo_coeff)
    mo_sa60 = mc.mo_coeff
    cis60 = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    print(f'    SA-60: converged = {mc.converged}, '
          f'E = {mc.e_tot:.6f}')
    results['SA-60 CASSCF'] = analyze('SA-60 CASSCF', mol, mo_sa60,
                                      act_ss,
                                      [np.asarray(c) for c in cis60], t0,
                                      e_ground=mc.e_tot,
                                      converged=mc.converged)

    # the summary table
    print('\n================ ORBITAL-SOURCE COMPARISON ================')
    print(f'{"variant":18s} {"l-closed":9s} {"invar":>9s} '
          f'{"purity_max":>11s} {"ladder_dev":>11s} '
          f'{"conv":>6s} {"E":>12s} verdict')
    for k, r in results.items():
        good = r['l_closed'] and r['ladder_dev'] < 1e-3
        ec = r['e_ground']
        print(f'{k:18s} {str(r["l_closed"]):9s} {r["invar"]:9.2e} '
              f'{r["purity_max"]:11.3e} {r["ladder_dev"]:11.3e} '
              f'{str(r["casscf_converged"]):>6s} '
              f'{(f"{ec:12.6f}" if ec is not None else "-"):>12s} '
              f'{"INTEGER" if good else "flagged/mixed"}')
    print('===========================================================')
    print(f'total time: {time.time() - t00:.0f} s')

    json.dump(results, open(os.path.join(ROOT,
              'lmom/data/u14e_casscf_comparison.meta.json'), 'w'),
              indent=1)
    print('stored: lmom/data/u14e_casscf_comparison.meta.json')
    ok = (results['sph-ROHF CASCI']['ladder_dev'] < 1e-3
          and results['raw-ROHF CASCI']['ladder_dev'] > 1e-2
          and results['SA-13 CASSCF']['ladder_dev'] < 1e-3)
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
