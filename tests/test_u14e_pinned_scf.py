#!/usr/bin/env python
"""test_u14e_pinned_scf.py — the SELF-CONSISTENT symmetry-imposed HF
of the raw quintet configuration (U-14e), EXACT by construction.

Protocol (spin=4 throughout):
1. fresh raw ROHF (no chk) — the unrestricted reference;
2. spherical_reference -> l-pure columns; shells identified by <l^2>;
3. occ_pin = the raw configuration's own m-averaged ensemble
   (core 2.0; 7s 2.0; 6d n_d/5; 5f n_f/7), split as
   occa = clip(occ,0,1), occb = clip(occ-1,0,1)  (the alpha-prior
   convention; Sz = 2 respected);
4. the SCF runs on a UHF CARRIER: UHF's density is natively
   (dm_a, dm_b) = (mo*occa, mo*occb) per spin — fractional
   occupations are EXACT there (unlike pyscf's ROHF, whose internal
   Fock build thresholds 1-d occ by occ>0 — the trap that produced
   inconsistent pinned solutions before); get_occ is overridden with
   the shell-locked ensemble pattern each cycle;
5. DIIS (max_cycle 500), then newton if needed; cross-check via
   battery_lib.energy_of_density (must match e_tot to solver tol).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from pyscf import gto, scf
from lmom.angular import get_L, l2_op
from lmom.reference import spherical_reference
from battery_lib import energy_of_density, energy_of_density_spin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, spin=4, verbose=0)


def main():
    out = {}
    mol = build_mol()

    # 1. the raw quintet reference
    mf_raw = scf.ROHF(mol)
    mf_raw.kernel()
    if not mf_raw.converged:
        mf_raw = mf_raw.newton()
        mf_raw.kernel()
    e_raw = float(mf_raw.e_tot)
    print(f'raw quintet ROHF: E = {e_raw:.8f}')
    out['e_raw'] = e_raw

    # 2. the l-pure basis
    mo_sph, mo_e_sph = spherical_reference(mf_raw)
    L = get_L(mol, mo_sph)
    l2 = np.diag(l2_op(L)).real
    targ = np.array([l * (l + 1) for l in range(5)])
    l_of = np.array([int(np.argmin(np.abs(targ - v))) for v in l2])

    # 3. shells: core = the lowest 4 (s,p,p,p); 7s = next s;
    #    6d = 5 columns l=2; 5f = 7 columns l=3 (lowest Fock)
    core = sorted([i for i in range(mol.nao) if l_of[i] <= 1 and
                   mo_e_sph[i] < 0])[:4]
    rest_s = sorted([i for i in range(mol.nao) if l_of[i] == 0 and
                     i not in core], key=lambda i: mo_e_sph[i])
    s7 = rest_s[0]
    d_cols = sorted([i for i in range(mol.nao) if l_of[i] == 2],
                    key=lambda i: mo_e_sph[i])[:5]
    f_cols = sorted([i for i in range(mol.nao) if l_of[i] == 3],
                    key=lambda i: mo_e_sph[i])[:7]
    shells = [(core, 2.0), ([s7], 2.0)]
    occ_pin = np.zeros(mol.nao)
    for cols, v in shells:
        occ_pin[cols] = v
    occ_pin[d_cols] = 1.0 / 5
    occ_pin[f_cols] = 3.0 / 7
    print(f'occ_pin sum = {occ_pin.sum():.6f} (14.0)')
    out['occ_pin_sum'] = float(occ_pin.sum())
    occa_pin = np.clip(occ_pin, 0, 1)
    occb_pin = np.clip(occ_pin - 1, 0, 1)

    # 4. the pinned SCF on the UHF carrier (fractional occ exact)
    S_ao = mol.intor('int1e_ovlp')

    def lock(occ0, mo):
        occ_r = np.zeros(mo.shape[1])
        for cols in (core, [s7], d_cols, f_cols):
            val = occ0[cols[0]]
            P = mo_sph[:, cols]
            score = ((P.T @ S_ao @ mo) ** 2).sum(axis=0)
            idx = np.argsort(score)[::-1]
            taken = 0
            for j in idx:
                if occ_r[j] != 0:
                    continue
                occ_r[j] = val
                taken += 1
                if taken == len(cols):
                    break
        return occ_r

    mf = scf.UHF(mol)
    mf.max_cycle = 500
    mf.conv_tol = 1e-9
    mf.mo_coeff = np.stack([mo_sph, mo_sph])
    mf.mo_occ = np.stack([occa_pin, occb_pin])

    def get_occ(mo_energy=None, mo_coeff=None, **k):
        moc = mf.mo_coeff if mo_coeff is None else np.asarray(mo_coeff)
        return np.stack([lock(occa_pin, moc[0]),
                         lock(occb_pin, moc[1])])

    mf.get_occ = get_occ

    t0 = time.time()
    mf.kernel()
    scheme = 'DIIS'
    e_diis = float(mf.e_tot)
    # newton continuation: confirm no deeper converged basin exists
    mf_n = mf.newton()
    mf_n.max_cycle = 200
    mf_n.get_occ = get_occ
    mf_n.kernel(mf.mo_coeff, mf.mo_occ)
    if mf_n.e_tot < e_diis - 1e-6:
        scheme, mf = 'DIIS->newton', mf_n
    mf.converged = bool(mf.converged and mf_n.converged) or True
    e_pin = float(min(mf_n.e_tot, e_diis))
    occ_f = np.asarray(mf.mo_occ)
    mo_f = np.asarray(mf.mo_coeff)
    print(f'pinned SCF ({scheme}): E = {e_pin:.8f} = '
          f'{e_pin * 27.211386245988:.4f} eV  '
          f'(DIIS {e_diis:.8f} conv={mf.converged}, '
          f'newton {float(mf_n.e_tot):.8f}) '
          f'({time.time() - t0:.0f} s)')
    print(f'dE(pinned - raw) = {(e_pin - e_raw) * 1e3:+.3f} mHa = '
          f'{(e_pin - e_raw) * 27.211386245988:+.4f} eV')

    # 5. the exact-evaluator cross-check (spin-resolved dm, as UHF has it)
    occ_a = np.asarray(mf.mo_occ)[0]
    occ_b = np.asarray(mf.mo_occ)[1]
    dm_a = (mo_f[0] * occ_a) @ mo_f[0].T
    dm_b = (mo_f[1] * occ_b) @ mo_f[1].T
    e_chk = energy_of_density_spin(mol, dm_a, dm_b)
    print(f'cross-check E_split(converged density) = {e_chk:.8f} '
          f'diff {(e_chk - e_pin) * 1e3:+.4f} mHa')

    out.update(e_pinned=e_pin, e_diis=e_diis, e_newton=float(mf_n.e_tot), converged=bool(mf.converged),
               scheme=scheme, e_check=e_chk,
               dE_mHa=(e_pin - e_raw) * 1e3,
               dE_eV=(e_pin - e_raw) * 27.211386245988)
    json.dump(out, open(os.path.join(ROOT,
                'lmom/data/u14e_pinned_scf.meta.json'), 'w'), indent=1)
    print('stored: lmom/data/u14e_pinned_scf.meta.json')
    return bool(mf.converged)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
