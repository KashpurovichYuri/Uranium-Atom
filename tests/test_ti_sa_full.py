#!/usr/bin/env python
"""test_ti_sa_full.py — A4: does BIGGER state-averaging restore the
symmetric solution? Ti def2-tzvp: the whole Sz sector is 190 roots;
the atom table capped SA at 60 (invar 0.97, broken). Here: SA-190
(the FULL sector). If the full-multiplet-set average still breaks the
span, averaging size is conclusively not the cure.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.reference import spherical_reference, force_config
from lmom.moments import moment_mats, moment_matrix
from lmom.diag import purity_devs, lowdin_shell_pops
from battery_lib import mo_passport

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    mol = gto.M(atom='Ti 0 0 0', basis='def2-tzvp', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    core_n = (mol.nelectron - 2) // 2      # 10
    mfr = force_config(mol, mf, {2: 2.0}, core_n)
    mo_sph, _ = spherical_reference(mfr)
    occ = mfr.mo_occ
    from lmom.diag import purity_devs as _pd
    Lg = get_L(mol, mo_sph)
    l2 = np.diag(sum(k @ k for k in Lg)).real
    targets = np.array([l * (l + 1) for l in range(7)])
    l_of = np.array([int(np.argmin(np.abs(targets - v))) for v in l2])
    frac = [i for i in range(core_n, len(occ))
            if 1e-8 < occ[i] < 2 - 1e-8]
    open_l = {int(l_of[i]) for i in frac}
    act = [i for i in range(core_n, len(occ))
           if int(l_of[i]) in open_l]
    nact = len(act)
    dim = cistring.num_strings(nact, 2)
    print(f'Ti def2: act {act} (n={nact}), sector dim = {dim}')
    mo_passport(mol, mo_sph, occ, act, list(range(core_n)), mol.nelectron,
                expect_l=[2] * nact, label='sph start')

    mf_s = scf.ROHF(mol)
    mf_s.mo_coeff = mo_sph
    mf_s.mo_occ = occ
    mf_s.e_tot = 0.0
    mf_s.converged = True

    t0 = time.time()
    mc = mcscf.CASSCF(mf_s, nact, (2, 0))
    mc.ncore = core_n
    mc.conv = 1e-7
    mc.max_cycle = 100
    mc.nroots = dim
    mc.fcisolver.nroots = dim
    mc = mcscf.state_average_(mc, [1.0 / dim] * dim)
    mc.kernel(mf_s.mo_coeff)
    cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    e = float(np.atleast_1d(mc.e_tot).ravel()[0])
    print(f'SA-full({dim}): converged = {mc.converged}, E = {e:.6f} '
          f'({time.time() - t0:.0f} s)')

    act_c = list(range(core_n, core_n + nact))
    mo_passport(mol, mc.mo_coeff, None, act_c, None, None,
                expect_l=[2] * nact, label='SA-full')
    L = get_L(mol, mc.mo_coeff)
    sc = span_certificate(L, act_c)
    pdev = purity_devs(L, act_c)
    pops = lowdin_shell_pops(mol, mc.mo_coeff, act_c)
    p_tot = float(sum(p[1] for p in pops))
    mats = moment_mats(mol, mc.mo_coeff, act_c, 2, 0, 'L2')
    nsa = cistring.num_strings(nact, 2)
    MM = moment_matrix(mats, nsa, 1, [np.asarray(c) for c in cis])
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = np.abs(lad - np.round(lad)).max()
    print(f'SA-full({dim}): span l_closed = {sc["l_closed"]} (invar '
          f'{sc["invar_err"]:.2e}), purity {pdev.max():.2e}, p-pop '
          f'{p_tot:.2f}, ladder dev {dev:.2e} '
          f'({"INTEGER" if dev < 1e-3 else "MIXED"})')
    print(f'  ladder: {np.round(lad, 3)}')

    out = dict(dim=dim, e=e, converged=bool(mc.converged),
               l_closed=bool(sc['l_closed']),
               invar=float(sc['invar_err']),
               purity=float(pdev.max()), p_pop=p_tot,
               ladder_dev=float(dev), ladder=[float(x) for x in lad])
    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/ti_sa_full.meta.json'), 'w'), indent=1)
    print('stored: ti_sa_full.meta.json')
    verdict = sc['l_closed'] and dev < 1e-3
    print(f'\n==> bigger-SA question: full-sector averaging '
          f'{"RESTORES" if verdict else "does NOT restore"} the '
          f'symmetric solution')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
