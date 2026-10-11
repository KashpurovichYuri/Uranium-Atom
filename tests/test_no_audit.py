#!/usr/bin/env python
"""test_no_audit.py — A3: the full-space natural-occupancy audit over
every stored solution family:
  U-14e: sph-CASCI(6,13) 60 roots, CAS(6,16) 100 roots, the purified
         ss/SA-13 symmetrized solutions, and fresh 13-col ss/SA-13
  atoms: Si/Sc/Ti x (sto-3g, def2-tzvp): sph-CASCI root0, ss, SA
Question: does the CAS contain all important NOs (0.02..1.98), is the
core drained, how dead are the dead columns?
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.moments import moment_expectation
from battery_lib import no_audit, casci_run
from tests.test_u14e_energy_ncheck import build_mol as build_u

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4


def main():
    t00 = time.time()
    out = {}

    # ---------------- U-14e stored ----------------
    mol = build_u()
    for tag, fn, nact_e, nel, nc in (
            ('sph-CASCI(6,13)', 'u14e_roots60.npz', 13, (5, 1), 4),
            ('CAS(6,16)', 'u14e_cas16.npz', 16, (5, 1), 4)):
        d = np.load(os.path.join(ROOT, 'lmom/data', fn),
                    allow_pickle=True)
        act = [int(c) for c in d['act_cols']]
        cis = [np.asarray(c) for c in d['cis']]
        for k in (0,):
            print(f'U {tag} root {k}:')
            out[f'U {tag} r{k}'] = no_audit(mol, d['mo_coeff'], cis[k],
                                            nel, nc, nact_e,
                                            label=f'{tag} r{k}')
    for tag, fn in (('ss-symm', 'u14e_casscf_ss_symm.npz'),
                    ('SA13-symm', 'u14e_casscf_SA-13_symm.npz')):
        fp = os.path.join(ROOT, 'lmom/data', fn)
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            print(f'U {tag} (purified) root 0:')
            out[f'U {tag}'] = no_audit(mol, d['mo_purified'],
                                       np.asarray(d['cis'][0]), (5, 1),
                                       4, len(d['act_cols']),
                                       label=tag)

    # fresh 13-col ss/SA-13 (the broken references)
    d_sph = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                    allow_pickle=True)
    mf0 = scf.ROHF(mol)
    mf0.mo_coeff = d_sph['mo_coeff']
    mf0.mo_occ = d_sph['mo_occ']
    mf0.e_tot = 0.0
    mf0.converged = True
    act13 = [int(c) for c in d_sph['act_cols']]
    for tag, nroots, w in (('ss', 1, None), ('SA-13', 13, [1 / 13] * 13)):
        mc = mcscf.CASSCF(mf0, len(act13), (NA, NB))
        mc.ncore = NCORE
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.nroots = nroots
        mc.fcisolver.nroots = nroots
        if w:
            mc = mcscf.state_average_(mc, w)
        mc.kernel(mf0.mo_coeff)
        cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
        print(f'U 13-col {tag}:')
        out[f'U 13col {tag}'] = no_audit(mol, mc.mo_coeff,
                                         np.asarray(cis[0]), (NA, NB),
                                         NCORE, len(act13),
                                         label=f'13col {tag}')

    # ---------------- atoms ----------------
    from tests.test_atoms_table import SYSTEMS
    from lmom.reference import force_config, spherical_reference
    from lmom.angular import get_L
    for (name, atom, basis, spin, nelecas, exp_l, focc) in SYSTEMS:
        mol_a = gto.M(atom=f'{atom} 0 0 0', basis=basis, spin=spin,
                      verbose=0)
        mf_a = scf.ROHF(mol_a)
        mf_a.kernel()
        if not mf_a.converged:
            mf_a = mf_a.newton()
            mf_a.kernel()
        core_n = (mol_a.nelectron - (nelecas[0] + nelecas[1])) // 2
        mfr = force_config(mol_a, mf_a, focc, core_n) if focc else mf_a
        mo_sph, _ = spherical_reference(mfr)
        occ = mfr.mo_occ
        frac = [i for i in range(core_n, len(occ))
                if 1e-8 < occ[i] < 2 - 1e-8]
        Lg = get_L(mol_a, mo_sph)
        l2 = np.diag(sum(k @ k for k in Lg)).real
        targets = np.array([l * (l + 1) for l in range(7)])
        l_of = np.array([int(np.argmin(np.abs(targets - v)))
                         for v in l2])
        open_l = {int(l_of[i]) for i in frac}
        act = [i for i in range(core_n, len(occ))
               if int(l_of[i]) in open_l]
        nact = len(act)
        dim = cistring.num_strings(nact, nelecas[0]) * \
            cistring.num_strings(nact, nelecas[1])
        sa_n = min(dim, 60)
        mf_s = scf.ROHF(mol_a)
        mf_s.mo_coeff = mo_sph
        mf_s.mo_occ = occ
        mf_s.e_tot = 0.0
        mf_s.converged = True
        cis_s, _ = casci_run(mol_a, mf_s, act, nelecas, core_n, dim,
                             conv_tol=1e-12)
        print(f'{name} sph-CASCI root0:')
        out[f'{name} sph-CASCI'] = no_audit(mol_a, mo_sph, cis_s[0],
                                            nelecas, core_n, nact,
                                            label=f'{name} sph')
        mc = mcscf.CASSCF(mf_s, nact, nelecas)
        mc.ncore = core_n
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.kernel(mf_s.mo_coeff)
        print(f'{name} ss:')
        out[f'{name} ss'] = no_audit(mol_a, mc.mo_coeff,
                                     np.asarray(mc.ci), nelecas, core_n,
                                     nact, label=f'{name} ss')
        mc = mcscf.CASSCF(mf_s, nact, nelecas)
        mc.ncore = core_n
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.nroots = sa_n
        mc.fcisolver.nroots = sa_n
        mc = mcscf.state_average_(mc, [1.0 / sa_n] * sa_n)
        mc.kernel(mf_s.mo_coeff)
        cis_sa = mc.ci if isinstance(mc.ci, list) else \
            [np.asarray(mc.ci)]
        print(f'{name} SA:')
        out[f'{name} SA'] = no_audit(mol_a, mc.mo_coeff,
                                     np.asarray(cis_sa[0]), nelecas,
                                     core_n, nact, label=f'{name} SA')

    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/no_audit.meta.json'), 'w'), indent=1)
    print(f'\nstored: lmom/data/no_audit.meta.json '
          f'({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
