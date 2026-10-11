#!/usr/bin/env python
"""test_starts_casscf.py — Wave 2: ss + SA-13 CASSCF from the
informative starts (B raw-direct, C UKS-5f4 direct, D UKS-CHAR direct,
F UKS-MOM direct). A-direct and B-symm are the already-measured
references. Battery per run: E, converged, span, purity, ladder,
<L^2>(r0), occupancy label.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.moments import moment_mats, moment_matrix, moment_expectation
from lmom.diag import purity_devs, lowdin_shell_pops
from battery_lib import mo_passport
from tests.test_starts_matrix import build_mol, load_start, match_to_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4
STARTS = [('B raw-direct', 'B', 'direct'),
          ('C UKS5f4-direct', 'C', 'direct'),
          ('D CHAR-direct', 'D', 'direct'),
          ('F MOM-direct', 'F', 'direct')]


def main():
    t00 = time.time()
    mol = build_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mo_ref = d['mo_coeff']
    act_ref = [int(c) for c in d['act_cols']]
    core_ref = [int(c) for c in d['core_cols']]
    out = {}
    for sname, sid, treat in STARTS:
        mo, occ, _ = load_start(mol, sid)
        core_m, act_m, minov = match_to_ref(mol, mo, mo_ref, act_ref,
                                            core_ref)
        nact = len(act_m)
        mfs = scf.ROHF(mol)
        mfs.mo_coeff = mo
        mfs.mo_occ = occ
        mfs.e_tot = 0.0
        mfs.converged = True
        for tag, nroots, w in (('ss', 1, None),
                               ('SA-13', 13, [1 / 13] * 13)):
            t0 = time.time()
            mc = mcscf.CASSCF(mfs, nact, (NA, NB))
            mc.ncore = NCORE
            mc.conv = 1e-7
            mc.max_cycle = 100
            mc.nroots = nroots
            mc.fcisolver.nroots = nroots
            if w:
                mc = mcscf.state_average_(mc, w)
            mc.kernel(mfs.mo_coeff)
            cis = mc.ci if isinstance(mc.ci, list) else \
                [np.asarray(mc.ci)]
            act_c = list(range(NCORE, NCORE + nact))
            L = get_L(mol, mc.mo_coeff)
            sc = span_certificate(L, act_c)
            pdev = purity_devs(L, act_c)
            mats = moment_mats(mol, mc.mo_coeff, act_c, NA, NB, 'L2')
            nsa = cistring.num_strings(nact, NA)
            nsb = cistring.num_strings(nact, NB)
            MM = moment_matrix(mats, nsa, nsb,
                               [np.asarray(c) for c in cis])
            lad = np.sort(np.linalg.eigvalsh(MM))
            dev = np.abs(lad - np.round(lad)).max()
            q0 = moment_expectation(mats, nsa, nsb, cis[0])
            pops = lowdin_shell_pops(mol, mc.mo_coeff, act_c)
            p_tot = float(sum(p[1] for p in pops))
            g = None
            try:
                g = float(np.linalg.norm(np.asarray(
                    mc.get_grad()).ravel()))
            except Exception:
                pass
            print(f'[{sname} {tag}] E = '
                  f'{float(np.atleast_1d(mc.e_tot)[0]):.6f}, conv = '
                  f'{mc.converged}, |grad| = {g}, invar = '
                  f'{sc["invar_err"]:.2e}, purity {pdev.max():.2f}, '
                  f'ladder {dev:.2e}, <L2>r0 = {q0:.3f}, p-pop '
                  f'{p_tot:.2f} ({time.time() - t0:.0f} s)')
            out[f'{sname} {tag}'] = dict(
                e=float(np.atleast_1d(mc.e_tot)[0]),
                converged=bool(mc.converged), grad=g,
                invar=float(sc['invar_err']),
                l_closed=bool(sc['l_closed']),
                purity=float(pdev.max()), ladder_dev=float(dev),
                l2_root0=float(q0), p_pop=p_tot,
                min_overlap_start=float(minov))
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/starts_casscf.meta.json'), 'w'), indent=1)
    print(f'\nstored: starts_casscf.meta.json ({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
