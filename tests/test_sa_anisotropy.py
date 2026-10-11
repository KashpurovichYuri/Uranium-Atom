#!/usr/bin/env python
"""test_sa_anisotropy.py — the mechanism-display experiment: monitor the
SA-CASSCF density anisotropy ||dm - dm_sph||_F / ||dm||_F per
optimization cycle.

Prediction from the analysis:
  - the quoted equal-weight SA protocol holds the density spherical only
    AT the exact symmetric stationary point;
  - along the path, inexact Davidson roots seed anisotropy, and where
    the anisotropic descent has fuel (flexible bases), the anisotropy
    GROWS monotonically into the broken minimum;
  - on starved bases (carbon) it stays ~0.

Systems: U-14e SA-13 (breaks), Ti def2 SA-60 (breaks), carbon SA-3
(green — the control).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from lmom.reference import spherical_reference, force_config, \
    m_average_density
from lmom.angular import get_L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_u_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def sa_anisotropy_run(mol, mf, nact, nelecas, ncore, nroots, label,
                      cycles_out):
    """SA-CASSCF with a per-cycle density-anisotropy probe."""
    state = dict(curves=[])

    def callback(envs):
        try:
            # the SA kernel exposes the WEIGHTED-AVERAGED CAS 1-RDM as
            # envs['casdm1'] (ncas x ncas, the SI/SA density)
            casdm1 = np.asarray(envs['casdm1'])
            mo = np.asarray(envs['casscf'].mo_coeff)
            nmo = mo.shape[1]
            D = np.zeros(nmo)
            D[:ncore] = 2.0
            D[ncore:ncore + nact] = np.diag(casdm1).real
            dm = (mo * D) @ mo.T
            dm_sph = m_average_density(mol, dm)
            a = float(np.linalg.norm(dm - dm_sph) /
                      np.linalg.norm(dm))
            cyc = envs.get('cycle', len(state['curves']))
            state['curves'].append((int(cyc), a))
        except Exception as e:
            state.setdefault('errors', []).append(repr(e))

    mc = mcscf.CASSCF(mf, nact, nelecas)
    mc.ncore = ncore
    mc.conv = 1e-7
    mc.max_cycle = 100
    mc.nroots = nroots
    mc.fcisolver.nroots = nroots
    mc = mcscf.state_average_(mc, [1.0 / nroots] * nroots)
    mc.kernel(mf.mo_coeff, callback=callback)
    cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    print(f'  [{label}] converged = {mc.converged}, E = '
          f'{float(np.atleast_1d(mc.e_tot).ravel()[0]):.6f}')
    aniso = [a for _, a in state['curves']]
    if aniso:
        print(f'  [{label}] anisotropy: cycle 0 = {aniso[0]:.2e}, '
              f'final = {aniso[-1]:.2e}, max = {max(aniso):.2e}')
    # the final-state verdict via the span
    L = get_L(mol, mc.mo_coeff)
    from lmom.expectation import span_certificate
    act = list(range(ncore, ncore + nact))
    sc = span_certificate(L, act)
    print(f'  [{label}] final span invar = {sc["invar_err"]:.2e} '
          f'(l_closed {sc["l_closed"]})')
    cycles_out[label] = dict(
        curves=[[int(c), float(a)] for c, a in state['curves']],
        e=float(np.atleast_1d(mc.e_tot).ravel()[0]),
        converged=bool(mc.converged),
        invar=float(sc['invar_err']),
        l_closed=bool(sc['l_closed']),
        errors=state.get('errors', [])[:3])
    return state['curves']


def main():
    t00 = time.time()
    out = {}

    # ---- the control: carbon SA-3 (the green case) ----
    mol = gto.M(atom='C 0 0 0', basis='cc-pvdz', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    print('carbon SA-3 (the green control):')
    cur = sa_anisotropy_run(mol, mf, 3, (2, 0), 2, 3, 'carbon SA-3', out)
    print('  curve:', [(c, f'{a:.1e}') for c, a in cur])

    # ---- U-14e SA-13 (the breaking case) ----
    mol_u = build_u_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mfu = scf.ROHF(mol_u)
    mfu.mo_coeff = d['mo_coeff']
    mfu.mo_occ = d['mo_occ']
    mfu.e_tot = 0.0
    mfu.converged = True
    print('\nU-14e SA-13 (the breaking case):')
    cur = sa_anisotropy_run(mol_u, mfu, 13, (5, 1), 4, 13,
                            'U SA-13', out)
    print('  curve:', [(c, f'{a:.1e}') for c, a in cur])

    # ---- Ti def2 SA-60 (the flexible-basis case) ----
    mol_ti = gto.M(atom='Ti 0 0 0', basis='def2-tzvp', spin=2,
                   verbose=0)
    mft = scf.ROHF(mol_ti)
    mft.kernel()
    if not mft.converged:
        mft = mft.newton()
        mft.kernel()
    core_n = (mol_ti.nelectron - 2) // 2
    mfr = force_config(mol_ti, mft, {2: 2.0}, core_n)
    mo_sph, _ = spherical_reference(mfr)
    occ = mfr.mo_occ
    from lmom.angular import get_L, l2_op
    l2 = np.diag(l2_op(get_L(mol_ti, mo_sph))).real
    targets = np.array([l * (l + 1) for l in range(7)])
    l_of = np.array([int(np.argmin(np.abs(targets - v))) for v in l2])
    frac = [i for i in range(core_n, len(occ))
            if 1e-8 < occ[i] < 2 - 1e-8]
    open_l = {int(l_of[i]) for i in frac}
    act = [i for i in range(core_n, len(occ))
           if int(l_of[i]) in open_l]
    mfs = scf.ROHF(mol_ti)
    mfs.mo_coeff = mo_sph
    mfs.mo_occ = occ
    mfs.e_tot = 0.0
    mfs.converged = True
    from pyscf.fci import cistring
    nact = len(act)
    dim = cistring.num_strings(nact, 2)
    sa_n = min(dim, 60)
    print(f'\nTi def2 SA-{sa_n} (the flexible-basis case):')
    cur = sa_anisotropy_run(mol_ti, mfs, nact, (2, 0), core_n, sa_n,
                            'Ti def2 SA', out)
    print('  curve:', [(c, f'{a:.1e}') for c, a in cur][:20])

    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/sa_anisotropy.meta.json'), 'w'), indent=1)
    print(f'\nstored: lmom/data/sa_anisotropy.meta.json '
          f'({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
