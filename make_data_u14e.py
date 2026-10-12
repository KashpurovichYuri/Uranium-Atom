#!/usr/bin/env python
"""make_data_u14e.py — U-14e data generation for the lmom store.

Loads the converged checkpoints (no SCF re-run), runs CASCI(6,13) with 20
roots on (a) the ROHF MOs and (b) the CASSCF MOs, and stores everything
for the offline lmom analysis:
    lmom/data/u14e_rohf.npz, u14e_casscf.npz  (+ <name>.meta.json)

Cross-check: the CASCI ground energy must reproduce the pipeline's
recorded value (CAS cols [5,6,7,8,9,14,15,19,20,27,28,29,30] 1-based,
CASCI E = -50.3878399792029).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '14e'))

from pyscf import gto, scf, mcscf, lib
import uranium_pipeline as up
from run_14e import BASIS, STUTTGART_RLC_ECP78 as ECP
from lmom.angular import get_L, l2_op
from lmom.expectation import L2Lz_expectation_dm, span_certificate, core_certificate

ACT = [4, 5, 6, 7, 8, 13, 14, 18, 19, 26, 27, 28, 29]   # 0-based
NELECAS = (5, 1)
NROOTS = 20
REF_E = -50.3878399792029
OUT = os.path.join(HERE, 'lmom', 'data')

up.configure(ECP_KIND='std', ATOM='U 0 0 0', UNIT='B', CHARGE=0, SPIN=4,
             BASIS=BASIS, ECP=ECP, CORE_MAP={0: 2, 1: 6},
             NELECAS=NELECAS, NCAS=13,
             CAS_MULTIPLETS={'s': 1, 'd': 5, 'f': 7},
             TARGET_CFG={'s': 2.0, 'd': 1.0, 'f': 3.0})
mol = up.build_system().mol
nao = mol.nao
print(f'U-14e: nao = {nao}, spin = 4, ACT(0-based) = {ACT}')


def load_mf(chk):
    mf = scf.ROHF(mol)
    d = lib.chkfile.load(chk, 'scf')
    mf.__dict__.update(d)
    mf.converged = True
    occ = np.asarray(mf.mo_occ)
    if occ.ndim == 2:
        occ = occ.sum(axis=0)
    mf.mo_occ = occ
    print(f'  loaded {chk}: E = {mf.e_tot:.8f}')
    return mf


def casci_roots(mf, mo, tag, check_ref=True, act=None):
    act = ACT if act is None else act
    mo_save = mf.mo_coeff
    mf.mo_coeff = mo
    mci = mcscf.CASCI(mf, 13, NELECAS)
    mci.nroots = NROOTS
    mci.fcisolver.nroots = NROOTS
    mci.kernel(mci.sort_mo([i + 1 for i in act]))
    cis = mci.ci
    mf.mo_coeff = mo_save
    if not isinstance(cis, list):
        cis = [np.asarray(cis)] if np.asarray(cis).ndim == 2 \
            else [np.asarray(c) for c in cis]
    w = np.atleast_1d(mci.e_tot)
    print(f'  [{tag}] CASCI E(root 0) = {w[0]:.10f}  '
          f'(reference {REF_E:.10f}, diff {abs(w[0] - REF_E):.1e});  '
          f'{len(cis)} roots')
    if check_ref:
        assert abs(w[0] - REF_E) < 1e-6, 'CAS pick mismatch vs pipeline record'
    return w, cis


def analyze_and_store(name, mo, occ, w, cis, act=None, core=None):
    act = ACT if act is None else act
    core = [i for i in range(nao) if occ[i] == 2 and i not in act] \
        if core is None else core
    L = get_L(mol, mo)
    sc = span_certificate(L, act)
    cc = core_certificate(L, occ, core, act)
    print(f'  [{name}] span: l-closed = {sc["l_closed"]} '
          f'(closure {sc["closure_err"]:.1e}, invariance '
          f'{sc["invar_err"]:.1e});  core: purity = '
          f'{cc["purity_err"]:.1e}, invariance = {cc["core_inv"]:.1e}, '
          f'disjoint = {cc["disjoint"]:.1e}, closed = {cc["closed"]}  '
          f'-> {"OK" if cc["ok"] else "FLAGGED"}')
    rows = []
    for k in range(len(cis)):
        rk = L2Lz_expectation_dm(mol, mo, act, cis[k], NELECAS, core_cols=core,
                        mo_occ=occ)
        rows.append(rk)
        print(f'    root {k:2d}: E = {w[k]:+.8f}  <L^2> = {rk["L2"]:+.6f}  '
              f'<Lz> = {rk["Lz"]:+.6f}  <Lz^2> = {rk["Lz2"]:+.6f}  '
              f'L_eff = {rk["L_eff"]:.3f}  Term: {rk["term"]}  '
              f'closure = {rk["closure"]:.1e}')
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'{name}.npz'), mo_coeff=mo, mo_occ=occ,
             e_roots=w, cis=np.array(cis), act_cols=np.array(act),
             core_cols=np.array(core))
    meta = dict(system=name, atom='U', ecp='stuttgart-rlc-ecp78 (14e)',
                spin=4, cas='CAS(6,13) 7s1 6d5 5f7', act_cols=act,
                core_cols=core, nelecas=list(NELECAS), nroots=len(cis),
                premise6='spherical H assumed (spin-free ECP); verify '
                         'multiplet completeness in decompose_states_op',
                l2_per_root=[rk['L2'] for rk in rows],
                lz2_per_root=[rk['Lz2'] for rk in rows],
                span=dc(sc), core=dc(cc))
    json.dump(meta, open(os.path.join(OUT, f'{name}.meta.json'), 'w'),
              indent=1)
    print(f'  stored -> {OUT}/{name}.npz + .meta.json\n')


def dc(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dc(v)
            continue
        a = np.asarray(v)
        out[k] = a.tolist() if a.ndim else float(a)
    return out


def spherical_pick(mf, occ_ref, raw_mo):
    """Radial-set-pure pick on the spherical reference: among the
    complete radial multiplets (the 7-fold f and 5-fold d epsilon
    clusters, l-invariant by construction), select the one capturing the
    most weight of the PARTIALLY occupied orbitals (the open shell), plus
    the valence s (the highest-epsilon s column)."""
    L = get_L(mol, mf.mo_coeff)
    l2d = np.diag(l2_op(L)).real
    S = mol.intor('int1e_ovlp')

    def clusters(lt):
        cols = sorted((i for i in range(nao) if abs(l2d[i] - lt) < 0.6),
                      key=lambda i: mf.mo_energy[i])
        cl, run = [], []
        for i in cols:
            if run and mf.mo_energy[i] - mf.mo_energy[run[-1]] > 0.05:
                cl.append(run); run = []
            run.append(i)
        cl.append(run)
        return cl

    def best_weighted(clusters, size, occ_lo, occ_hi, lval):
        cand = [c for c in clusters if len(c) == size]
        # ALL partially occupied columns (their l^2 may be impure — the
        # Hund-broken open shell is exactly what we are placing)
        occ_sel = [i for i in range(nao) if occ_lo < occ_ref[i] < occ_hi]
        if not cand or not occ_sel:
            return None, -1.0, []
        wts = []
        for c in cand:
            C = mf.mo_coeff[:, c]
            w = 0.0
            for i in occ_sel:
                v = raw_mo[:, i]     # the RAW occupied MO columns
                w += float(np.sum((C.T @ S @ v) ** 2))
            wts.append(w)
        k = int(np.argmax(wts))
        return cand[k], wts[k], wts
    f_cl = clusters(12)
    f_pick, f_w, f_wts = best_weighted(f_cl, 7, 0, 2, 12)
    assert f_pick is not None, f'f clusters: {[len(c) for c in f_cl]}'
    d_cl = clusters(6)
    d_pick, d_w, d_wts = best_weighted(d_cl, 5, 0, 2, 6)
    assert d_pick is not None, f'd clusters: {[len(c) for c in d_cl]}'
    # the valence s: the same overlap-weight rule (the size-1 clusters =
    # the individual s columns); the highest-epsilon s overall would pick
    # a high virtual (~+200 Eh)
    s_cl = clusters(0)
    s_pick, s_w, s_wts = best_weighted(s_cl, 1, 0, 2, 0)
    assert s_pick is not None, 'no s cluster'
    s_val = s_pick[0]
    act = sorted(f_pick + d_pick + [s_val])
    print(f'  spherical pick: f cluster (weight {f_w:.3f}) = {f_pick}')
    print(f'  d clusters: {[(len(c), round(w, 3)) for c, w in zip(d_cl, d_wts)]}'
          f' -> {d_pick} (weight {d_w:.3f});  s = {s_val}')
    print(f'  s clusters (weight): '
          f'{[(c[0], round(w, 3)) for c, w in zip(s_cl, s_wts) if w > 0.01]}'
          f' -> s = {s_val} (weight {s_w:.3f})')
    print(f'  act = {act};  per-MO <l^2> = {np.round(l2d[act], 6)}')
    return act


def main():
    # (0) frozen CASCI on the RAW ROHF MOs with the pipeline pick —
    # the NEGATIVE CONTROL: the certificates must flag
    mf = load_mf(os.path.join(HERE, '14e', 'rohf.chk'))
    w, cis = casci_roots(mf, mf.mo_coeff, 'rohf-raw')
    analyze_and_store('u14e_rohf_raw', mf.mo_coeff, mf.mo_occ, w, cis)

    # (a) frozen CASCI on the SPHERICAL reference MOs with the
    # radial-group pick — the certified term analysis
    mf_s = scf.ROHF(mol)  # a carrier for the spherical MOs/occ
    from lmom.reference import spherical_reference
    mo_sph, mo_e_sph = spherical_reference(mf)
    mf_s.mo_coeff = mo_sph
    mf_s.mo_energy = mo_e_sph
    occ_s = np.zeros(nao)
    occ_s[0:4] = 2.0                      # 6s^2 6p^6 (the closed core)
    mf_s.mo_occ = occ_s
    act_s = spherical_pick(mf_s, occ_ref=mf.mo_occ,
                           raw_mo=mf.mo_coeff)
    # the physical open-shell occupations for the certificates:
    # 7s^2 (the valence s), 6d^1 + 5f^3 on the picked groups
    occ_s[act_s[-1]] = 2.0
    occ_s[act_s[0:5]] = 1.0
    occ_s[act_s[5:12]] = 1.0
    mf_s.mo_occ = occ_s
    w, cis = casci_roots(mf_s, mf_s.mo_coeff, 'rohf-sph', check_ref=False,
                         act=act_s)
    analyze_and_store('u14e_rohf_sph', mf_s.mo_coeff, occ_s, w, cis,
                      act=act_s)

    # (b) CASCI on the CASSCF MOs (the post-optimization picture; the
    # active columns tracked by overlap with the seed-row pick)
    mcc = lib.chkfile.load(os.path.join(HERE, '14e', 'casscf.chk'), 'mcscf')
    mo_c = mcc['mo_coeff']
    L = get_L(mol, mf.mo_coeff)
    ov = mf.mo_coeff[:, ACT].conj().T @ mol.intor('int1e_ovlp') @ mo_c
    tracked, used = [], set()
    for i in range(len(ACT)):
        j = int(np.argmax(np.abs(ov[i])))
        while j in used:
            ov[i, j] = 0.0
            j = int(np.argmax(np.abs(ov[i])))
        used.add(j)
        tracked.append(j)
    print(f'  CASSCF active columns (tracked): {tracked}')
    occ_c = np.zeros(nao)
    occ_c[[i for i in range(nao) if i not in tracked
           and mf.mo_occ[i] == 2]] = 2  # the CAS core: doubly occupied
    mf_c = scf.ROHF(mol)
    mf_c.mo_coeff = mo_c
    mf_c.mo_occ = occ_c
    w, cis = casci_roots(mf_c, mo_c, 'casscf', check_ref=False)
    analyze_and_store('u14e_casscf', mo_c, occ_c, w, cis)


if __name__ == '__main__':
    main()
