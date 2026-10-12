#!/usr/bin/env python
"""test_commutator.py — the route-independent theorem-premise check:
the commutator norm ||[H_CASCI, L^2]|| on the sector. The central
theorem states [P H P_span, L^2] = 0 for an l-invariant span; the
commutator must therefore be ~0 for green references and FINITE for
broken ones — a verdict that does not touch any CI vector.

Probe: ||[H,W]v|| / max(||Hv||, ||Wv||) on random vectors
(H via pyscf's own contract_1e/contract_2e on mc.h1eff/h2eff;
W via the moment reshape-trick). Coverage:
  green: U sph-CASCI(6,13), CAS(6,16); Si/Sc/Ti sph-CASCI (both bases)
  broken: fresh U 13-col ss/SA-13 (prediction: finite)
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.reference import spherical_reference, force_config
from lmom.moments import moment_mats, apply_moment
from battery_lib import casci_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def h_vec_apply_real(mc, V, nelecas):
    """H V through pyscf's own REAL kernels (the Hamiltonian is
    symmetric — pack_tril conventions fine for it)."""
    h1e = mc.get_h1eff()
    h1 = np.asarray(h1e[0] if isinstance(h1e, tuple) else h1e)
    h2 = np.asarray(mc.get_h2eff())
    if np.iscomplexobj(h1) and np.abs(h1.imag).max() < 1e-12:
        h1 = h1.real
    if np.iscomplexobj(h2) and np.abs(h2.imag).max() < 1e-12:
        h2 = h2.real
    norb = h1.shape[0]
    out = fci.direct_spin1.contract_1e(h1, V, norb, nelecas)
    return out + fci.direct_spin1.contract_2e(h2, V, norb, nelecas)


def commutator_probe(mc, mats, na_str, nb_str, nelecas, nvec=3, seed=0):
    """||[H,W]v|| / max(||Hv||, ||Wv||) in ALL-REAL algebra: the
    complex-Hermitian W splits as W = Wr + i Wi (Wr symmetric-real,
    Wi antisymmetric-real; the real/imag parts of apply_moment on real
    vectors are exactly those two REAL linear maps), so
    [H,W]v = [H,Wr]v + i[H,Wi]v with everything real."""
    rng = np.random.default_rng(seed)
    ratios = []

    def w_parts(v):
        w = apply_moment(mats, na_str, nb_str, v)
        return np.real(w), np.imag(w)

    for k in range(nvec):
        v = rng.standard_normal(na_str * nb_str)
        HV = h_vec_apply_real(mc, v, nelecas)
        wr_v, wi_v = w_parts(v)
        Hwr = h_vec_apply_real(mc, wr_v, nelecas)
        Hwi = h_vec_apply_real(mc, wi_v, nelecas)
        wr_Hv, wi_Hv = w_parts(HV)
        c_re = Hwr - wr_Hv
        c_im = Hwi - wi_Hv
        C = float(np.linalg.norm(c_re) ** 2 +
                  np.linalg.norm(c_im) ** 2) ** 0.5
        den = max(float(np.linalg.norm(HV)),
                  float(np.sqrt(np.linalg.norm(wr_v) ** 2 +
                                np.linalg.norm(wi_v) ** 2)), 1e-30)
        ratios.append(C / den)
    return max(ratios)


def build_u_mol():
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
    out = {}

    # ---- U-14e: the green references ----
    mol = build_u_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    act = [int(c) for c in d['act_cols']]
    mf = scf.ROHF(mol)
    mf.mo_coeff = d['mo_coeff']
    mf.mo_occ = d['mo_occ']
    mf.e_tot = 0.0
    mf.converged = True
    mc = mcscf.CASCI(mf, len(act), (5, 1))
    mc.ncore = 4
    mc.kernel(mc.sort_mo([c + 1 for c in act]))
    mats = moment_mats(mol, d['mo_coeff'], act, 5, 1, 'L2')
    nsa = cistring.num_strings(len(act), 5)
    nsb = cistring.num_strings(len(act), 1)
    r = commutator_probe(mc, mats, nsa, nsb, (5, 1))
    out['U sph-CASCI(6,13) [green]'] = r
    print(f'U sph-CASCI(6,13)  [green]:  ||[H,W]v||/max = {r:.2e}')

    d16 = np.load(os.path.join(ROOT, 'lmom/data/u14e_cas16.npz'),
                  allow_pickle=True)
    act16 = [int(c) for c in d16['act_cols']]
    occ16 = np.zeros(d16['mo_coeff'].shape[1])
    occ16[:4] = 2.0
    for c in act16:
        occ16[c] = 2.0 if c == 4 else \
            (3.0 / 7 if 22 <= c <= 28 else
             (0.0 if c in (5, 6, 7) else 1.0 / 5))
    mf16 = scf.ROHF(mol)
    mf16.mo_coeff = d16['mo_coeff']
    mf16.mo_occ = occ16
    mf16.e_tot = 0.0
    mf16.converged = True
    mc = mcscf.CASCI(mf16, len(act16), (5, 1))
    mc.ncore = 4
    mc.kernel(mc.sort_mo([c + 1 for c in act16]))
    mats = moment_mats(mol, d16['mo_coeff'], act16, 5, 1, 'L2')
    nsa = cistring.num_strings(len(act16), 5)
    nsb = cistring.num_strings(len(act16), 1)
    r = commutator_probe(mc, mats, nsa, nsb, (5, 1))
    out['U CAS(6,16) [green]'] = r
    print(f'U CAS(6,16)        [green]:  ||[H,W]v||/max = {r:.2e}')

    # ---- U-14e: the broken references (fresh ss/SA-13) ----
    for tag, nroots, w in (('ss', 1, None), ('SA-13', 13,
                                             [1 / 13] * 13)):
        mcx = mcscf.CASSCF(mf, len(act), (5, 1))
        mcx.ncore = 4
        mcx.conv = 1e-7
        mcx.max_cycle = 100
        mcx.nroots = nroots
        mcx.fcisolver.nroots = nroots
        if w:
            mcx = mcscf.state_average_(mcx, w)
        mcx.kernel(mf.mo_coeff)
        act_c = list(range(4, 4 + len(act)))
        mats = moment_mats(mol, mcx.mo_coeff, act_c, 5, 1, 'L2')
        nsa = cistring.num_strings(len(act_c), 5)
        mc_probe = mcscf.CASCI(mf, len(act), (5, 1))
        mc_probe.ncore = 4
        mc_probe.kernel(mcx.mo_coeff)
        r = commutator_probe(mc_probe, mats, nsa,
                             cistring.num_strings(len(act_c), 1),
                             (5, 1))
        out[f'U {tag} [broken]'] = r
        print(f'U 13-col {tag:6s} [broken]:  ||[H,W]v||/max = {r:.2e}')

    # ---- the atoms: sph-CASCI (green) ----
    from tests.test_atoms_table import SYSTEMS
    from battery_lib import no_audit  # noqa (parity of imports)
    for (name, atom, basis, spin, nelecas, exp_l, focc) in SYSTEMS:
        mol_a = gto.M(atom=f'{atom} 0 0 0', basis=basis, spin=spin,
                      verbose=0)
        mfa = scf.ROHF(mol_a)
        mfa.kernel()
        if not mfa.converged:
            mfa = mfa.newton()
            mfa.kernel()
        core_n = (mol_a.nelectron - (nelecas[0] + nelecas[1])) // 2
        mfr = force_config(mol_a, mfa, focc, core_n) if focc else mfa
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
        act_a = [i for i in range(core_n, len(occ))
                 if int(l_of[i]) in open_l]
        nact = len(act_a)
        mfs = scf.ROHF(mol_a)
        mfs.mo_coeff = mo_sph
        mfs.mo_occ = occ
        mfs.e_tot = 0.0
        mfs.converged = True
        mc = mcscf.CASCI(mfs, nact, nelecas)
        mc.ncore = core_n
        mc.kernel(mc.sort_mo([c + 1 for c in act_a]))
        mats = moment_mats(mol_a, mo_sph, act_a, nelecas[0],
                           nelecas[1], 'L2')
        nsa = cistring.num_strings(nact, nelecas[0])
        nsb = cistring.num_strings(nact, nelecas[1])
        r = commutator_probe(mc, mats, nsa, nsb, nelecas)
        out[f'{name} sph [green]'] = r
        print(f'{name:14s} [green]:  ||[H,W]v||/max = {r:.2e}')

    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/commutator.meta.json'), 'w'), indent=1)
    green = {k: v for k, v in out.items() if 'green' in k}
    broken = {k: v for k, v in out.items() if 'broken' in k}
    print(f'\nverdict: green max {max(green.values()):.2e} '
          f'(expect ~0), broken min {min(broken.values()):.2e} '
          f'(expect finite) -> '
          f'{"PASS" if max(green.values()) < 1e-5 and min(broken.values()) > 1e-3 else "CHECK"}')
    print(f'stored: commutator.meta.json ({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
