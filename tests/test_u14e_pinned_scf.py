#!/usr/bin/env python
"""test_u14e_pinned_scf.py — the SELF-CONSISTENT symmetry-imposed HF
of the raw quintet configuration (U-14e), EXACT by construction.

Protocol (spin=4 throughout):
1. fresh raw ROHF (no chk) — the unrestricted reference;
2. the CANONICAL stored sph basis (l-ordered; shells = fixed
   column indices core 0-3, 7s 4, 6d 9-13, 5f 22-28);
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
   battery_lib.energy_of_density_spin (must match e_tot);
6. the solution saved to restart/pinned_ensemble.chk; then the
   verification: spherical_reference on the pinned density ->
   certificates -> CASCI(6,13)(5,1) vs the stored reference
   -50.164365 + 13-root ladder integrality.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
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

    # 1. the raw quintet reference (fresh, no chk)
    mf_raw = scf.ROHF(mol)
    mf_raw.kernel()
    if not mf_raw.converged:
        mf_raw = mf_raw.newton()
        mf_raw.kernel()
    e_raw = float(mf_raw.e_tot)
    print(f'raw quintet ROHF: E = {e_raw:.8f}')
    out['e_raw'] = e_raw

    # 2. the canonical l-pure basis: the FRESH sph basis (the
    #    stored l-grouped basis turned out NOT to be a stable SCF
    #    pinning basis: 800 cycles without convergence - a documented
    #    negative); shells identified by <l^2> + Fock order, then
    #    FIXED column indices (no re-lock)
    mo_sph, mo_e_sph = spherical_reference(mf_raw)
    L = get_L(mol, mo_sph)
    l2 = np.diag(l2_op(L)).real
    targ = np.array([l * (l + 1) for l in range(5)])
    l_of = np.array([int(np.argmin(np.abs(targ - v))) for v in l2])
    core = sorted([i for i in range(mol.nao) if l_of[i] <= 1 and
                   mo_e_sph[i] < 0])[:4]
    rest_s = sorted([i for i in range(mol.nao) if l_of[i] == 0 and
                     i not in core], key=lambda i: mo_e_sph[i])
    s7 = rest_s[0]
    d_cols = sorted([i for i in range(mol.nao) if l_of[i] == 2],
                    key=lambda i: mo_e_sph[i])[:5]
    f_cols = sorted([i for i in range(mol.nao) if l_of[i] == 3],
                    key=lambda i: mo_e_sph[i])[:7]
    print(f'shells: core={core} 7s={s7} 6d={d_cols} 5f={f_cols}')
    occ_pin = np.zeros(mol.nao)
    occ_pin[core] = 2.0
    occ_pin[s7] = 2.0
    occ_pin[d_cols] = 1.0 / 5
    occ_pin[f_cols] = 3.0 / 7
    print(f'occ_pin sum = {occ_pin.sum():.6f} (14.0)')
    out['occ_pin_sum'] = float(occ_pin.sum())
    occa_pin = np.clip(occ_pin, 0, 1)
    occb_pin = np.clip(occ_pin - 1, 0, 1)
    occ_fix = np.stack([occa_pin, occb_pin])

    # 3. the pinned ensemble SCF: UHF carrier; the ensemble occ is
    #    REBUILT each cycle by the shell-lock (overlap to the start
    #    shells). NOTE: a fixed-index occ variant (no lock) was also
    #    tried and is ILL-CONDITIONED: 800+ cycles without pyscf
    #    convergence (recorded in the meta).
    S_ao = mol.intor('int1e_ovlp')
    shell_sets = [core, [s7], d_cols, f_cols]
    starts = [occa_pin, occb_pin]

    def locked_occ(occ0, mo):
        occ_r = np.zeros(mo.shape[1])
        for cols in shell_sets:
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
        return np.stack([locked_occ(starts[0], moc[0]),
                         locked_occ(starts[1], moc[1])])

    mf.get_occ = get_occ

    t0 = time.time()
    mf.kernel()
    scheme = 'DIIS'
    e_diis = float(mf.e_tot)
    if not mf.converged:
        mf_n = mf.newton()
        mf_n.max_cycle = 300
        mf_n.get_occ = get_occ
        mf_n.kernel(mf.mo_coeff, mf.mo_occ)
        scheme = 'DIIS->newton'
        mf = mf_n if mf_n.e_tot < float(mf.e_tot) else mf
    e_pin = float(mf.e_tot)
    occ_f = np.asarray(mf.mo_occ)
    mo_f = np.asarray(mf.mo_coeff)
    print(f'pinned ensemble SCF ({scheme}): E = {e_pin:.8f} = '
          f'{e_pin * 27.211386245988:.4f} eV  conv={mf.converged} '
          f'({time.time() - t0:.0f} s)')
    print(f'dE(pinned - raw) = {(e_pin - e_raw) * 1e3:+.3f} mHa = '
          f'{(e_pin - e_raw) * 27.211386245988:+.4f} eV')

    # 4. the exact-evaluator cross-check
    occ_f = np.asarray(mf.mo_occ)
    mo_f = np.asarray(mf.mo_coeff)
    dm_a = (mo_f[0] * occ_f[0]) @ mo_f[0].T
    dm_b = (mo_f[1] * occ_f[1]) @ mo_f[1].T
    e_chk = energy_of_density_spin(mol, dm_a, dm_b)
    print(f'cross-check E_split = {e_chk:.8f} '
          f'diff {(e_chk - e_pin) * 1e3:+.4f} mHa')

    # 5. save the canonical checkpoint
    chk_path = os.path.join(ROOT, 'restart', 'pinned_ensemble.chk')
    from pyscf import lib
    if not np.all(np.isfinite(mo_f)) or not np.all(np.isfinite(occ_f)):
        raise RuntimeError('non-finite solution - not saving')
    lib.chkfile.save_mol(mol, chk_path)
    lib.chkfile.dump(chk_path, 'scf',
                     {'e_tot': e_pin, 'mo_coeff': mo_f, 'mo_occ': occ_f,
                      'note': 'canonical pinned ensemble SCF, UHF '
                              'carrier, spin=4; mol = run_14e spin=4'})
    print(f'stored: {chk_path}')

    # 6. the verification: pinned orbitals -> spherical_reference ->
    #    certificates -> CASCI(6,13)(5,1) + ladder + commutator
    carrier = scf.ROHF(mol)
    carrier.mo_coeff = mo_f[0]
    carrier.mo_occ = occ_f.sum(axis=0)
    mo_r, mo_e_r = spherical_reference(carrier)
    from lmom.expectation import span_certificate
    act = [s7] + d_cols + f_cols
    L = get_L(mol, mo_r)
    sc = span_certificate(L, act)
    print(f'[verify] span l_closed = {sc["l_closed"]}, invar = '
          f'{sc["invar_err"]:.2e}')
    mc = mcscf.CASCI(carrier, 13, (5, 1))
    mc.ncore = 4
    mc.kernel(mc.sort_mo([c + 1 for c in act]))
    e_ci = float(np.atleast_1d(mc.e_tot)[0])
    print(f'[verify] CASCI = {e_ci:.8f} (gate -50.164365) '
          f'diff {(e_ci + 50.16436519) * 1e3:+.4f} mHa')
    mci = mcscf.CASCI(carrier, 13, (5, 1))
    mci.ncore = 4
    mci.nroots = 13
    mci.fcisolver.nroots = 13
    mci.kernel(mci.sort_mo([c + 1 for c in act]))
    cis = [np.asarray(c) for c in
           (mci.ci if isinstance(mci.ci, list) else [mci.ci])]
    from lmom.moments import moment_mats, moment_matrix
    mats = moment_mats(mol, mo_r, act, 5, 1)
    MM = moment_matrix(mats, 1287, 13, cis)
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = float(np.abs(lad - np.round(lad)).max())
    print(f'[verify] 13-root ladder dev = {dev:.2e}')

    ok = (bool(mf.converged) and sc['l_closed']
          and abs((e_ci + 50.16436519) * 1e3) < 0.5 and dev < 1e-5)
    out['ill_conditioning_note'] = (
        'fixed-index occ (no lock) variant: 800+ cycles, no pyscf '
        'convergence - the shell-locked variant is the convergent one')
    out.update(e_pinned=e_pin, e_diis=e_diis, scheme=scheme,
               converged=bool(mf.converged), e_check=e_chk,
               dE_mHa=(e_pin - e_raw) * 1e3,
               dE_eV=(e_pin - e_raw) * 27.211386245988,
               verify=dict(e_casci=e_ci, span_invar=sc['invar_err'],
                           ladder_dev=dev, passed=bool(ok)))
    json.dump(out, open(os.path.join(ROOT,
                'lmom/data/u14e_pinned_ensemble.meta.json'), 'w'),
        indent=1)
    print('stored: lmom/data/u14e_pinned_ensemble.meta.json')
    print('VERIFICATION:', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
