#!/usr/bin/env python
"""test_u14e_population.py — Lowdin-by-shell composition of the active
columns for each U-14e orbital source (sph ROHF, raw ROHF, ss-CASSCF,
SA-13 CASSCF) + the ORIGIN of any p character (overlap with the
reference core 6p vs the virtual 7p managagement). Decides the
CAS(6,16) extension shell.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from lmom.angular import get_L
from lmom.moments import moment_mats
from lmom.diag import lowdin_shell_pops, pops_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def shell_of_ref(mol, mo, cols):
    """l label per reference column from the l^2 diagonal."""
    L = get_L(mol, mo)
    l2 = np.diag(sum(k @ k for k in L)).real
    targets = np.array([l * (l + 1) for l in range(7)])
    return {c: int(np.argmin(np.abs(targets - l2[c]))) for c in cols}


def main():
    t00 = time.time()
    mol = build_mol()
    d_sph = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                    allow_pickle=True)
    d_raw = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_raw.npz'),
                    allow_pickle=True)
    results = {}

    def report(tag, mo, act):
        pops = lowdin_shell_pops(mol, mo, act)
        print(f'  [{tag}] active-column Lowdin shells:')
        print('   ', pops_summary(pops, tag))
        p_tot = float(sum(p[1] for p in pops))
        print(f'    total p population over the active columns: '
              f'{p_tot:.3f}')
        results[tag] = dict(pops=[p.tolist() for p in pops],
                            p_total=p_tot)
        return p_tot

    act_sph = [int(c) for c in d_sph['act_cols']]
    act_raw = [int(c) for c in d_raw['act_cols']]
    report('sph-ROHF', d_sph['mo_coeff'], act_sph)
    report('raw-ROHF', d_raw['mo_coeff'], act_raw)

    # the CASSCF variants (from the sph start)
    mf0 = scf.ROHF(mol)
    mf0.mo_coeff = d_sph['mo_coeff']
    mf0.mo_occ = d_sph['mo_occ']
    mf0.e_tot = 0.0
    mf0.converged = True

    for tag, nroots in (('ss-CASSCF', 1), ('SA-13', 13)):
        t0 = time.time()
        mc = mcscf.CASSCF(mf0, len(act_sph), (NA, NB))
        mc.ncore = NCORE
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.nroots = nroots
        mc.fcisolver.nroots = nroots
        if nroots > 1:
            mc = mcscf.state_average_(mc, [1.0 / nroots] * nroots)
        mc.kernel(mf0.mo_coeff)
        print(f'  [{tag}] converged = {mc.converged}, E = {mc.e_tot:.6f} '
              f'({time.time() - t0:.0f} s)')
        act_mc = list(range(NCORE, NCORE + len(act_sph)))
        p_tot = report(tag, mc.mo_coeff, act_mc)
        # the ORIGIN of the p character: overlap with the reference
        # core-6p (stored cols 1,2,3) and with the virtual p space
        S = mol.intor('int1e_ovlp')
        Ov = np.abs(mc.mo_coeff[:, act_mc].conj().T @ S
                    @ d_sph['mo_coeff'][:, [1, 2, 3]])
        ov_core_p = Ov.max(axis=1)
        print(f'    max |overlap| of each active column with the '
              f'reference core 6p: {np.round(ov_core_p, 3)}')
        results[tag]['max_ov_to_core_6p'] = [float(x) for x in
                                             ov_core_p]

    json.dump(results, open(os.path.join(ROOT,
              'lmom/data/u14e_population.meta.json'), 'w'), indent=1)
    print(f'\nstored: lmom/data/u14e_population.meta.json '
          f'({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
