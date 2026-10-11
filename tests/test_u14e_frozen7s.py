#!/usr/bin/env python
"""test_u14e_frozen7s.py — A5: exclude the doubly-occupied 7s from the
CAS and measure (a) the energy deviation vs having it active,
(b) whether the CASSCF drift changes when the doubly-occupied active
orbital (a zero-mode source) is removed.

  CAS(6,13)(5,1) 7s+6d5+5f7   : the reference (E = -50.164365)
  CAS(4,12)(3,1) 6d5+5f7      : 7s^2 frozen into the core (ncore = 5)
  CAS(4,15)(3,1) +7p3         : frozen 7s + the polarization p
  ss/SA-13 CASSCF in CAS(4,12): the drift question

Sector (3,1): M_S = 1... NOTE: the 7s pair contributed S=0, so the
5f^3 6d^1 quintets keep M_S = 2 in the full space but with 7s frozen
the EXPLICIT electrons are (3,1) -> the f^3 d^1 spin coupling S = 2
requires 4 parallel electrons: (4,0)! The (3,1) sector instead holds
the f^3 d^1 states with M_S = 1 (triplets/quintet M-component) — the
L^2 ladder is unchanged (L is spin-independent); only the M_S sector
differs. We run (4,0) to keep the quintet M_S = 2 physics.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.moments import (moment_mats, moment_matrix, trace_moment,
                           sector_eigenvalues, moment_expectation)
from lmom.diag import purity_devs, lowdin_shell_pops
from battery_lib import mo_passport, casci_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P7 = [5, 6, 7]


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def ladder_block(mol, mo, act, nelecas, cis, e, tol=1e-4):
    mats = moment_mats(mol, mo, act, nelecas[0], nelecas[1], 'L2')
    nsa = cistring.num_strings(len(act), nelecas[0])
    nsb = cistring.num_strings(len(act), nelecas[1])
    tr = trace_moment(mats, nsa, nsb)
    MM = moment_matrix(mats, nsa, nsb, [np.asarray(c) for c in cis])
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = np.abs(lad - np.round(lad)).max()
    clusters = [[0]]
    for i in range(1, len(cis)):
        if e[i] - e[i - 1] > tol:
            clusters.append([])
        clusters[-1].append(i)
    cl = []
    for c in clusters:
        ladc = lad[c[0]:c[-1] + 1]
        devc = np.abs(ladc - np.round(ladc)).max()
        ls = {}
        for v in np.round(ladc).astype(int):
            Ln = int(round((np.sqrt(max(v, 0) + 1) - 1)))
            ls[Ln] = ls.get(Ln, 0) + 1
        cl.append(dict(roots=c, L=ls, dev=float(devc)))
    return dict(trace=float(tr), ladder_dev=float(dev),
                ladder=[float(x) for x in lad],
                max_int_dev=float(np.abs(
                    np.sort(sector_eigenvalues(mats, nsa, nsb, k=24))
                    - np.round(np.sort(
                        sector_eigenvalues(mats, nsa, nsb, k=24)))
                ).max()), clusters=cl)


def main():
    t00 = time.time()
    mol = build_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo = d['mo_coeff']
    act13 = [int(c) for c in d['act_cols']]
    s7 = 4
    act12 = [c for c in act13 if c != s7]
    act15 = sorted(act12 + P7)
    occ = np.asarray(d['mo_occ'])
    mf0 = scf.ROHF(mol)
    mf0.mo_coeff = mo
    mf0.e_tot = 0.0
    mf0.converged = True
    out = {}

    # occ vectors: core 4 cols 2.0; 7s 2.0; f 3/7; d 1/5 (from the
    # stored pattern, rebuilt explicitly to be self-consistent)
    occA = np.zeros(len(occ))          # for (6,13): 7s active
    occA[:4] = 2.0
    for c in act13:
        occA[c] = 2.0 if c == s7 else \
            (3.0 / 7 if c in range(22, 29) else 1.0 / 5)
    occB = occA.copy()                 # for (4,12): 7s frozen
    occB[s7] = 2.0

    print('=== CAS(4,12)(4,0): 6d5 + 5f7, 7s^2 frozen ===')
    mo_p = mo_passport(mol, mo, occB, act12, list(range(5)), 14,
                       expect_l=[2] * 5 + [3] * 7, label='CAS(4,12)')
    mf_a = scf.ROHF(mol)
    mf_a.mo_coeff = mo
    mf_a.mo_occ = occB
    mf_a.e_tot = 0.0
    mf_a.converged = True
    cis, e = casci_run(mol, mf_a, act12, (4, 0), 5, 20, conv_tol=1e-12)
    res = ladder_block(mol, mo, act12, (4, 0), cis, e)
    res['e_root0'] = float(e[0])
    print(f'  E = {e[0]:.6f} (CAS(6,13) ref -50.164365; '
          f'dE(frozen 7s) = {(e[0] + 50.164365) * 1000:+.2f} mHa)')
    print(f'  ladder dev = {res["ladder_dev"]:.1e}; trace = '
          f'{res["trace"]:.2f}; clusters:')
    for c in res['clusters'][:6]:
        print(f'    roots {c["roots"][0]}..{c["roots"][-1]}: '
              f'L = {c["L"]}, dev = {c["dev"]:.1e}')
    out['cas_4_12'] = res

    print('\n=== CAS(4,15)(4,0): + 7p3 ===')
    mo_p15 = mo_passport(mol, mo, occB, act15, list(range(5)), 14,
                         expect_l=[2] * 5 + [3] * 7 + [1] * 3,
                         label='CAS(4,15)')
    cis, e = casci_run(mol, mf_a, act15, (4, 0), 5, 20, conv_tol=1e-12)
    res15 = ladder_block(mol, mo, act15, (4, 0), cis, e)
    res15['e_root0'] = float(e[0])
    print(f'  E = {e[0]:.6f} (CAS(6,16) ref -50.168752; dE = '
          f'{(e[0] + 50.168752) * 1000:+.2f} mHa); ladder dev = '
          f'{res15["ladder_dev"]:.1e}')
    out['cas_4_15'] = res15

    print('\n=== CASSCF drift question: ss + SA-13 in CAS(4,12)(4,0) ===')
    for tag, nroots, w in (('ss', 1, None), ('SA-13', 13,
                                             [1 / 13] * 13)):
        t0 = time.time()
        mc = mcscf.CASSCF(mf_a, len(act12), (4, 0))
        mc.ncore = 5
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.nroots = nroots
        mc.fcisolver.nroots = nroots
        if w:
            mc = mcscf.state_average_(mc, w)
        mc.kernel(mo)
        cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
        mo_c = mc.mo_coeff
        act_c = list(range(5, 5 + len(act12)))
        mo_passport(mol, mo_c, None, act_c, None, None,
                    expect_l=[2] * 5 + [3] * 7, label=f'{tag}')
        L = get_L(mol, mo_c)
        sc = span_certificate(L, act_c)
        pdev = purity_devs(L, act_c)
        pops = lowdin_shell_pops(mol, mo_c, act_c)
        p_tot = float(sum(p[1] for p in pops))
        mats = moment_mats(mol, mo_c, act_c, 4, 0, 'L2')
        nsa = cistring.num_strings(len(act_c), 4)
        MM = moment_matrix(mats, nsa, 1, [np.asarray(c) for c in cis])
        lad = np.sort(np.linalg.eigvalsh(MM))
        dev = np.abs(lad - np.round(lad)).max()
        q0 = moment_expectation(mats, nsa, 1, cis[0])
        print(f'  [{tag}] converged = {mc.converged}, E = '
              f'{float(np.atleast_1d(mc.e_tot)[0]):.6f}; invar = '
              f'{sc["invar_err"]:.2e} (l_closed {sc["l_closed"]}); '
              f'purity {pdev.max():.1e}; p-pop {p_tot:.2f}; '
              f'ladder dev {dev:.1e}; <L^2>(r0) = {q0:.3f} '
              f'({time.time() - t0:.0f} s)')
        out[f'casscf_{tag}'] = dict(
            e=float(np.atleast_1d(mc.e_tot)[0]),
            invar=float(sc['invar_err']),
            l_closed=bool(sc['l_closed']),
            purity=float(pdev.max()), p_pop=p_tot,
            ladder_dev=float(dev), l2_root0=float(q0))

    print('\n=== A5 summary ===')
    print(f"  freezing 7s^2 costs "
          f"{(out['cas_4_12']['e_root0'] + 50.164365) * 1000:+.2f} mHa "
          f"(CAS 4,12 vs 6,13)")
    print(f"  freezing 7s^2 with 7p3: "
          f"{(out['cas_4_15']['e_root0'] + 50.168752) * 1000:+.2f} mHa "
          f"(CAS 4,15 vs 6,16)")
    for tag in ('ss', 'SA-13'):
        r = out[f'casscf_{tag}']
        print(f"  CASSCF[{tag}] in CAS(4,12): l_closed = "
              f"{r['l_closed']} (invar {r['invar']:.2e}), p-pop "
              f"{r['p_pop']:.2f}, ladder dev {r['ladder_dev']:.1e}")
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/u14e_frozen7s.meta.json'), 'w'), indent=1)
    print(f'stored: u14e_frozen7s.meta.json ({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
