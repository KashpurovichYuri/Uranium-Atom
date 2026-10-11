#!/usr/bin/env python
"""make_data_u32e.py — U-32e data generation (GRECP32, 32 explicit e).

Loads the converged GRPP checkpoints, runs CASCI(6,13) with 10 roots on
(a) the ROHF MOs and (b) the CASSCF MOs, stores for the offline lmom
analysis. NOTE: the (core+active) determinant space here is ~1e14 — the
operator-application methods (referee/vector/spectrum/decompose) are
infeasible at this scale; only the dm-contraction route applies
(lmom.expectation.L2Lz_expectation_dm), plus the certificates.
CAS pick (1-based, from the pipeline run log):
    [14, 15, 16, 17, 18, 23, 24, 28, 29, 36, 37, 38, 39]
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '32e'))

from pyscf import scf, mcscf, lib
import libgrpp4pyscf
import uranium_pipeline as up
from lmom.angular import get_L
from lmom.expectation import L2Lz_expectation_dm, span_certificate, core_certificate

ACT = [13, 14, 15, 16, 17, 22, 23, 27, 28, 35, 36, 37, 38]   # 0-based
NELECAS = (5, 1)
NROOTS = 10
OUT = os.path.join(HERE, 'lmom', 'data')

import test_uranium_atom_32e as cfg32

up.configure(ECP_KIND='grpp', ATOM='U 0 0 0', UNIT='B', CHARGE=0, SPIN=4,
             BASIS=cfg32.BASIS, ECP=cfg32.ECP, CORE_MAP=cfg32.CORE_MAP,
             NELECAS=NELECAS, NCAS=13)
mol = up.build_system().mol
nao = mol.nao
print(f'U-32e: nao = {nao}, spin = 4, ACT(0-based) = {ACT}')


def load_mf(chk):
    mf = scf.ROHF(mol)
    d = lib.chkfile.load(chk, 'scf')
    mf.__dict__.update(d)
    mf.converged = True
    occ = np.asarray(mf.mo_occ)
    if occ.ndim == 2:
        occ = occ.sum(0)
    mf.mo_occ = occ
    print(f'  loaded {chk}: E = {mf.e_tot:.8f}')
    return mf


def casci_roots(mf, tag):
    mci = mcscf.CASCI(mf, 13, NELECAS)
    mci.nroots = NROOTS
    mci.fcisolver.nroots = NROOTS
    mci.kernel(mci.sort_mo([i + 1 for i in ACT]))
    cis = mci.ci
    if not isinstance(cis, list):
        cis = [np.asarray(cis)] if np.asarray(cis).ndim == 2 \
            else [np.asarray(c) for c in cis]
    w = np.atleast_1d(mci.e_tot)
    print(f'  [{tag}] CASCI E(root 0) = {w[0]:.10f};  {len(cis)} roots')
    return w, cis


def dc(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dc(v)
            continue
        a = np.asarray(v)
        out[k] = a.tolist() if a.ndim else float(a)
    return out


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
    meta = dict(system=name, atom='U', ecp='GRECP-32 (libgrpp)',
                spin=4, cas='CAS(6,13) 7s1 6d5 5f7', act_cols=act,
                core_cols=core, nelecas=list(NELECAS), nroots=len(cis),
                methods='dm-only (the (C+A) det space ~1e14: the '
                        'application methods infeasible)',
                premise6='spin-free GRECP/AREP assumed; verify the '
                         'multiplet completeness when possible',
                l2_per_root=[rk['L2'] for rk in rows],
                lz2_per_root=[rk['Lz2'] for rk in rows],
                span=dc(sc), core=dc(cc))
    json.dump(meta, open(os.path.join(OUT, f'{name}.meta.json'), 'w'),
              indent=1)
    print(f'  stored -> {OUT}/{name}.npz + .meta.json\n')


def main():
    # (a) the ROHF MOs (the reference convention)
    mf = load_mf(os.path.join(HERE, '32e', 'rohf.chk'))
    w, cis = casci_roots(mf, 'rohf')
    analyze_and_store('u32e_rohf', mf.mo_coeff, mf.mo_occ, w, cis)

    # (b) the CASSCF MOs (the tracked active columns)
    mcc = lib.chkfile.load(os.path.join(HERE, '32e', 'casscf.chk'), 'mcscf')
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
           and mf.mo_occ[i] == 2]] = 2
    mf_c = scf.ROHF(mol)
    mf_c.mo_coeff = mo_c
    mf_c.mo_occ = occ_c
    w, cis = casci_roots(mf_c, 'casscf')
    analyze_and_store('u32e_casscf', mo_c, occ_c, w, cis, act=tracked)


if __name__ == '__main__':
    main()
