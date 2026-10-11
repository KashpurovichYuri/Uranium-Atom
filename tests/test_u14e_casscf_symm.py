#!/usr/bin/env python
"""test_u14e_casscf_symm.py — PHASE 2: post-hoc symmetrization of
CASSCF solutions (the ROHF-style purification applied to the FINAL
orbitals), U-14e.

4a: run ss-CASSCF / SA-13 from the stored sph start -> take the final
    MOs -> spherical_reference (m-averaged density, pure-l by Schur)
    -> re-pick the shells in the purified basis (core = lowest s + 3
    lowest p; act = 2nd s + 5 d + 7 f) -> CASCI re-solve -> certificates
    + ladder + dE vs the unconstrained CASSCF energy.
4b: if the span still fails, outer-loop CASSCF->purify (up to 3 rounds).

Caveat (documented): the span-certificate failure of the raw CASSCF
proved the optimizer crossed the span boundary (core<->active<->virtual),
so purification of the final columns is NOT guaranteed to restore an
l-closed span — the certificates judge. The energy gives back the
symmetry-breaking lowering (acceptance: dE 'not too far').
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate
from lmom.reference import spherical_reference
from lmom.moments import moment_mats, moment_matrix
from lmom.diag import purity_devs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA, NB, NCORE = 5, 1, 4
SHELL_N = {0: 1, 2: 5, 3: 7}   # the valence shells to pick (7s, 6d5, 5f7)


def build_mol():
    src = open(os.path.join(ROOT, '14e', 'run_14e.py')).read()
    ns = {}
    exec(compile(src.split('if __name__')[0], 'run_14e.py', 'exec'),
         {'__name__': 'run14e_cfg',
          '__file__': os.path.join(ROOT, '14e', 'run_14e.py')}, ns)
    return gto.M(atom='U 0 0 0', basis=ns['BASIS'],
                 ecp={'U': ns['STUTTGART_RLC_ECP78']}, verbose=0)


def dummy_mf(mol, mo, occ):
    mf = scf.ROHF(mol)
    mf.mo_coeff = np.asarray(mo)
    mf.mo_occ = np.asarray(occ)
    mf.e_tot = 0.0
    mf.converged = True
    return mf


def match_to_reference(mol, mo_new, mo_ref, core_ref, act_ref,
                       min_ov=0.7):
    """Assign the re-canonicalized columns to the trusted reference
    shells by MAXIMUM OVERLAP (energy-based picking is NOT robust: the
    m-averaged Fock from a different density reorders the radial
    spectrum). Greedy: each reference column takes the unused new
    column with the largest |<new|ref>|; flag if < min_ov."""
    S = mol.intor('int1e_ovlp')
    Ov = np.asarray(mo_new).conj().T @ S @ np.asarray(mo_ref)
    Ov = np.abs(Ov)
    used, core, act, ovs = set(), [], [], {}
    for kind, refs, tgt in (('core', core_ref, core),
                            ('act', act_ref, act)):
        for r in refs:
            cand = [(Ov[j, r], j) for j in range(Ov.shape[0])
                    if j not in used]
            o, j = max(cand)
            used.add(j)
            tgt.append(j)
            ovs[j] = float(o)
    bad = {j: o for j, o in ovs.items() if o < min_ov}
    if bad:
        print(f'    !! overlap match below {min_ov}: '
              f'{ {j: round(o, 3) for j, o in bad.items()} }')
    return core, act, ovs


def battery(tag, mol, mo, core, act, cis):
    L = get_L(mol, mo)
    sc = span_certificate(L, act)
    pdev = purity_devs(L, act)
    nact = len(act)
    nsa = cistring.num_strings(nact, NA)
    nsb = cistring.num_strings(nact, NB)
    mats = moment_mats(mol, mo, act, NA, NB, 'L2')
    MM = moment_matrix(mats, nsa, nsb, [np.asarray(c) for c in cis])
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = np.abs(lad - np.round(lad)).max()
    print(f'    [{tag}] span l_closed = {sc["l_closed"]} (invar '
          f'{sc["invar_err"]:.2e}), purity max = {pdev.max():.2e}')
    print(f'    [{tag}] ladder dev = {dev:.2e} '
          f'({"INTEGER" if dev < 1e-3 else "MIXED"}); first 8: '
          f'{np.round(lad[:8], 3)}')
    return dict(span_l_closed=bool(sc['l_closed']),
                invar=float(sc['invar_err']),
                purity_max=float(pdev.max()),
                ladder_dev=float(dev), ladder=lad.tolist())


def casscf_run(mf, nact, nroots, weights=None, mo_start=None):
    mc = mcscf.CASSCF(mf, nact, (NA, NB))
    mc.ncore = NCORE
    mc.conv = 1e-7
    mc.max_cycle = 100
    mc.nroots = nroots
    mc.fcisolver.nroots = nroots
    if weights is not None:
        mc = mcscf.state_average_(mc, weights)
    start = mf.mo_coeff if mo_start is None else mo_start
    mc.kernel(start)
    cis = mc.ci if isinstance(mc.ci, list) else [np.asarray(mc.ci)]
    return mc, [np.asarray(c) for c in cis]


def cas_occ(nao, cis, nact):
    """Occupation vector aligned to the CASSCF column order:
    core = 2.0, active = the natural occupancies from the CI (averaged
    over roots). Sums to 2*NCORE + (NA+NB)."""
    from pyscf import fci
    occ = np.zeros(nao)
    occ[:NCORE] = 2.0
    d1 = np.zeros(nact)
    for c in cis:
        d1a, d1b = fci.direct_spin1.make_rdm1s(np.asarray(c), nact,
                                               (NA, NB))
        d1 += np.diag(d1a).real + np.diag(d1b).real
    d1 /= len(cis)
    occ[NCORE:NCORE + nact] = d1
    return occ


def main():
    t00 = time.time()
    mol = build_mol()
    d = np.load(os.path.join(ROOT, 'lmom/data/u14e_rohf_sph.npz'),
                allow_pickle=True)
    mf0 = dummy_mf(mol, d['mo_coeff'], d['mo_occ'])
    results = {}

    for name, nroots, w in (('ss', 1, None), ('SA-13', 13, None)):
        weights = [1.0 / nroots] * nroots if nroots > 1 else None
        t0 = time.time()
        print(f'\n=== {name}: CASSCF({nroots} root(s)) from sph start ===')
        mc, cis = casscf_run(mf0, len(d['act_cols']), nroots, weights)
        e_mc = float(np.atleast_1d(mc.e_tot).ravel()[0])
        print(f'  converged = {mc.converged}, E = {e_mc:.6f} '
              f'({time.time() - t0:.0f} s)')
        act_mc = list(range(NCORE, NCORE + len(d['act_cols'])))
        r0 = battery(f'{name} raw', mol, mc.mo_coeff, None, act_mc, cis)
        r0['e_casscf'] = e_mc
        r0['converged'] = bool(mc.converged)

        # ---- 4a: post-hoc purification + CASCI re-solve ----
        # occ aligned to the CASSCF columns (core=2, active=natural);
        # the m-averaging needs the POPULATION IN THE RIGHT SHELLS
        nao = mc.mo_coeff.shape[0]
        occ_v = cas_occ(nao, cis, len(d['act_cols']))
        mo_p, mo_e = spherical_reference(dummy_mf(mol, mc.mo_coeff,
                                                  occ_v))
        core_p, act_p, ovs = match_to_reference(
            mol, mo_p, d['mo_coeff'],
            [int(c) for c in d['core_cols']],
            [int(c) for c in d['act_cols']])
        print(f'  purified: core = {core_p}, act = {act_p}')
        print(f'  overlap match (min): {min(ovs.values()):.3f}')
        mci = mcscf.CASCI(dummy_mf(mol, mo_p, np.zeros(mo_p.shape[1])),
                          len(act_p), (NA, NB))
        mci.ncore = NCORE
        mci.nroots = 20
        mci.fcisolver.nroots = 20
        mci.fcisolver.conv_tol = 1e-12
        mci.fcisolver.max_cycle = 300
        mci.kernel(mci.sort_mo([c + 1 for c in act_p]))
        cis_p = mci.ci if isinstance(mci.ci, list) else \
            [np.asarray(mci.ci)]
        e_p = float(np.atleast_1d(mci.e_tot).ravel()[0])
        rp = battery(f'{name} purified+CASCI', mol, mo_p, core_p, act_p,
                     cis_p)
        rp['e_root0'] = e_p
        rp['dE_vs_casscf'] = e_p - e_mc
        print(f'    dE(purified CASCI - CASSCF) = {e_p - e_mc:.6f} Ha')

        # ---- 4b: outer loop if the span still fails ----
        round_no = 1
        while not rp['span_l_closed'] and round_no < 3:
            round_no += 1
            print(f'  --- {name} outer loop, round {round_no}: CASSCF '
                  f'from the purified MOs ---')
            t0 = time.time()
            mc2, cis2 = casscf_run(mf0, len(act_p), nroots, weights,
                                   mo_start=mo_p)
            e_mc2 = float(np.atleast_1d(mc2.e_tot).ravel()[0])
            print(f'  converged = {mc2.converged}, E = {e_mc2:.6f} '
                  f'({time.time() - t0:.0f} s)')
            occ_v2 = cas_occ(nao, cis2, len(act_p))
            mo_p2, mo_e2 = spherical_reference(
                dummy_mf(mol, mc2.mo_coeff, occ_v2))
            core_p2, act_p2, ovs2 = match_to_reference(
                mol, mo_p2, d['mo_coeff'],
                [int(c) for c in d['core_cols']],
                [int(c) for c in d['act_cols']])
            print(f'  overlap match (min): {min(ovs2.values()):.3f}')
            mci = mcscf.CASCI(
                dummy_mf(mol, mo_p2, np.zeros(mo_p2.shape[1])),
                len(act_p2), (NA, NB))
            mci.ncore = NCORE
            mci.nroots = 20
            mci.fcisolver.nroots = 20
            mci.fcisolver.conv_tol = 1e-12
            mci.fcisolver.max_cycle = 300
            mci.kernel(mci.sort_mo([c + 1 for c in act_p2]))
            cis_p = mci.ci if isinstance(mci.ci, list) else \
                [np.asarray(mci.ci)]
            e_p = float(np.atleast_1d(mci.e_tot).ravel()[0])
            rp = battery(f'{name} round{round_no}', mol, mo_p2, core_p2,
                         act_p2, cis_p)
            rp['e_root0'] = e_p
            rp['dE_vs_casscf'] = e_p - e_mc2
            print(f'    dE(round{round_no} CASCI - CASSCF) = '
                  f'{e_p - e_mc2:.6f} Ha')
            r0[f'round{round_no}'] = rp

        r0['purified'] = rp
        r0['outer_rounds'] = round_no
        results[name] = r0
        np.savez_compressed(
            os.path.join(ROOT, f'lmom/data/u14e_casscf_{name}_symm.npz'),
            mo_purified=mo_p, act_cols=np.array(act_p),
            core_cols=np.array(core_p), e_roots=np.atleast_1d(e_p),
            cis=np.array([np.asarray(c) for c in cis_p]))

    print('\n================ SYMMETRIZATION TRACK ================')
    for name, r in results.items():
        p = r['purified']
        print(f'  {name:6s}: raw CASSCF dev {r["ladder_dev"]:.1e} -> '
              f'purified dev {p["ladder_dev"]:.1e}, span '
              f'{p["span_l_closed"]}, dE = {p["dE_vs_casscf"]:+.6f} Ha '
              f'({p["dE_vs_casscf"] * 1000:+.1f} mHa)')
    print('=======================================================')
    json.dump(results, open(os.path.join(ROOT,
              'lmom/data/u14e_casscf_symm.meta.json'), 'w'), indent=1)
    print(f'stored: u14e_casscf_ss_symm.npz, u14e_casscf_SA-13_symm.npz'
          f' + meta ({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
