#!/usr/bin/env python
"""test_atoms_table.py — item 3: the ATOM TABLE. For Si, Sc, Ti in
sto-3g and def2-tzvp, the full pipeline per atom:

  variants: raw-ROHF+CASCI, sph-ROHF+CASCI, ss-CASSCF (from the sph
  start), SA-full (all Sz-sector roots; capped at 60).

Per variant gates: span + core certificates, purity max, ladder int dev
vs the known terms, dm-vs-new-route agreement, electron-number checks
(Tr dm1, <N>, <N^2> op-route), Lowdin p-population, natural-occupancy
config label, energies (mHa and / smallest term gap).

Purpose: the 'which channel opens when' map — does ss break everywhere?
does SA break only where the CAS lacks polarization functions (the
U-14e signature)?
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring
from lmom.angular import get_L
from lmom.expectation import span_certificate, L2Lz_expectation_dm
from lmom.reference import spherical_reference, force_config
from lmom.moments import (moment_mats, moment_matrix,
                           moment_expectation)
from lmom.diag import purity_devs, lowdin_shell_pops, pops_summary
from battery_lib import n_expectation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEMS = [
    # name, atom, basis, spin, nelecas, expected L(L+1) present,
    # force_config l_occs or None
    ('Si sto-3g', 'Si', 'sto-3g', 2, (2, 0), [2], None),
    ('Si def2-tzvp', 'Si', 'def2-tzvp', 2, (2, 0), [2], None),
    ('Sc sto-3g', 'Sc', 'sto-3g', 1, (1, 0), [4], {2: 1.0}),
    ('Sc def2-tzvp', 'Sc', 'def2-tzvp', 1, (1, 0), [4], None),
    ('Ti sto-3g', 'Ti', 'sto-3g', 2, (2, 0), [12, 2], None),
    ('Ti def2-tzvp', 'Ti', 'def2-tzvp', 2, (2, 0), [12, 2], None),
]


def config_label(ci, nelecas, ncas):
    d1a, d1b = fci.direct_spin1.make_rdm1s(np.asarray(ci), ncas,
                                           nelecas)
    occ = np.sort(np.diag(d1a).real + np.diag(d1b).real)[::-1]
    return np.array2string(occ, precision=2, max_line_width=200)


def battery(tag, mol, mf_ref, mo, act, cis, nelecas, core_n):
    nact = len(act)
    nsa = cistring.num_strings(nact, nelecas[0])
    nsb = cistring.num_strings(nact, nelecas[1])
    L = get_L(mol, mo)
    sc = span_certificate(L, act)
    pdev = purity_devs(L, act)
    mats = moment_mats(mol, mo, act, nelecas[0], nelecas[1], 'L2')
    cis = [np.asarray(c) for c in cis]
    MM = moment_matrix(mats, nsa, nsb, cis)
    lad = np.sort(np.linalg.eigvalsh(MM))
    dev = np.abs(lad - np.round(lad)).max()
    terms = {}
    for v in np.round(lad).astype(int):
        Ln = int(round((np.sqrt(max(v, 0) + 1) - 1)))
        terms[Ln] = terms.get(Ln, 0) + 1
    q_new = [moment_expectation(mats, nsa, nsb, c) for c in cis]
    q_dm = [float(L2Lz_expectation_dm(mol, mo, act, c, nelecas,
                                      core_cols=list(range(core_n)),
                                      mo_occ=mf_ref.mo_occ)['L2'])
            for c in cis]
    dev_dm = max(abs(a - b) for a, b in zip(q_new, q_dm))
    n1, n2, n1x, n2x = n_expectation(mol, mo, act, nelecas[0],
                                     nelecas[1], cis[0])
    pops = lowdin_shell_pops(mol, mo, act)
    p_tot = float(sum(p[1] for p in pops))
    print(f'    [{tag}] span {sc["l_closed"]} (invar '
          f'{sc["invar_err"]:.1e}), purity {pdev.max():.1e}, ladder '
          f'{dev:.1e} L={terms}, dm-vs-new {dev_dm:.1e}, N-dev '
          f'{max(abs(n1 - n1x), abs(n2 - n2x)):.1e}, p-pop {p_tot:.2f}')
    print(f'    [{tag}] config(root0): '
          f'{config_label(cis[0], nelecas, nact)}')
    return dict(l_closed=bool(sc['l_closed']),
                invar=float(sc['invar_err']),
                purity_max=float(pdev.max()),
                ladder_dev=float(dev), terms=terms,
                dm_vs_new=float(dev_dm),
                n_dev=float(max(abs(n1 - n1x), abs(n2 - n2x))),
                p_pop=p_tot,
                ladder=[float(x) for x in lad],
                config=config_label(cis[0], nelecas, nact))


def main():
    t00 = time.time()
    out = {}
    for (name, atom, basis, spin, nelecas, exp_l, focc) in SYSTEMS:
        print(f'\n===== {name} =====')
        mol = gto.M(atom=f'{atom} 0 0 0', basis=basis, spin=spin,
                    verbose=0)
        mf0 = scf.ROHF(mol)
        mf0.kernel()
        if not mf0.converged:
            mf0 = mf0.newton()
            mf0.kernel()
        core_n = (mol.nelectron - (nelecas[0] + nelecas[1])) // 2
        if focc:
            mf_sph_rohf = force_config(mol, mf0, focc, core_n)
        else:
            mf_sph_rohf = mf0
        mo_sph, _ = spherical_reference(mf_sph_rohf)
        occ = mf_sph_rohf.mo_occ
        frac = [i for i in range(core_n, len(occ))
                if 1e-8 < occ[i] < 2 - 1e-8]
        Lg = get_L(mol, mo_sph)
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
        print(f'  act {act} (n={nact}), core {core_n}, sector {dim}')

        sysr = {}
        # raw + CASCI
        mf_raw = scf.ROHF(mol)
        mf_raw.mo_coeff = mf0.mo_coeff
        mf_raw.mo_occ = mf0.mo_occ
        mf_raw.e_tot = 0.0
        mf_raw.converged = True
        act_raw = [i for i in range(core_n, len(mf0.mo_occ))
                   if 1e-8 < mf0.mo_occ[i] < 2 - 1e-8]
        from battery_lib import casci_run
        cis_r, e_r = casci_run(mol, mf_raw, act_raw, nelecas, core_n,
                               dim)
        sysr['raw+CASCI'] = battery('raw+CASCI', mol, mf_raw,
                                    mf0.mo_coeff, act_raw, cis_r,
                                    nelecas, core_n)
        sysr['raw+CASCI']['e_root0'] = float(e_r[0])
        # sph + CASCI
        mf_s = scf.ROHF(mol)
        mf_s.mo_coeff = mo_sph
        mf_s.mo_occ = occ
        mf_s.e_tot = 0.0
        mf_s.converged = True
        cis_s, e_s = casci_run(mol, mf_s, act, nelecas, core_n, dim,
                               conv_tol=1e-12)
        sysr['sph+CASCI'] = battery('sph+CASCI', mol, mf_s, mo_sph,
                                    act, cis_s, nelecas, core_n)
        sysr['sph+CASCI']['e_root0'] = float(e_s[0])
        gap = float(np.min(np.diff(np.atleast_1d(e_s)))) \
            if len(e_s) > 1 else float('nan')
        # ss-CASSCF from sph
        mc = mcscf.CASSCF(mf_s, nact, nelecas)
        mc.ncore = core_n
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.kernel(mf_s.mo_coeff)
        g_ss = None
        try:
            g_ss = float(np.linalg.norm(np.asarray(
                mc.get_grad()).ravel()))
        except Exception:
            pass
        sysr['ss-CASSCF'] = battery('ss-CASSCF', mol, mc, mc.mo_coeff,
                                    list(range(core_n,
                                               core_n + nact)),
                                    [np.asarray(mc.ci)], nelecas,
                                    core_n)
        sysr['ss-CASSCF'].update(e=float(mc.e_tot),
                                 converged=bool(mc.converged),
                                 grad=g_ss)
        # SA-full
        w = [1.0 / sa_n] * sa_n
        mc = mcscf.CASSCF(mf_s, nact, nelecas)
        mc.ncore = core_n
        mc.conv = 1e-7
        mc.max_cycle = 100
        mc.nroots = sa_n
        mc.fcisolver.nroots = sa_n
        mc = mcscf.state_average_(mc, w)
        mc.kernel(mf_s.mo_coeff)
        cis_sa = mc.ci if isinstance(mc.ci, list) else \
            [np.asarray(mc.ci)]
        g_sa = None
        try:
            g_sa = float(np.linalg.norm(np.asarray(
                mc.get_grad()).ravel()))
        except Exception:
            pass
        sysr['SA'] = battery('SA', mol, mc, mc.mo_coeff,
                             list(range(core_n, core_n + nact)),
                             [np.asarray(c) for c in cis_sa], nelecas,
                             core_n)
        sysr['SA'].update(e=float(np.atleast_1d(mc.e_tot).ravel()[0]),
                          converged=bool(mc.converged), grad=g_sa)
        sysr['meta'] = dict(gap_sph_clusters_mHa=gap * 1000,
                            e_raw_rohf=float(mf0.e_tot),
                            dE_sph_density=(
                                float(mf_sph_rohf.e_tot) -
                                float(mf0.e_tot)) * 1000
                            if focc is None else None,
                            dE_casci_mHa=(float(e_s[0]) -
                                          float(e_r[0])) * 1000)
        exp_ok = all(any(abs(v - L * (L + 1)) < 1e-3
                         for v in np.round(sysr['sph+CASCI']['ladder']))
                     for L in exp_l)
        print(f'  ==> {name}: sph-CASCI terms-exist '
              f'{"PASS" if exp_ok else "FAIL"}; '
              f'dE_CASCI(raw->sph) = '
              f'{sysr["meta"]["dE_casci_mHa"]:+.2f} mHa '
              f'({abs(sysr["meta"]["dE_casci_mHa"]) / (gap * 1000):.2f} '
              f'gaps)' if gap > 0 else '')
        out[name] = sysr

    json.dump(out, open(os.path.join(ROOT,
              'lmom/data/atom_table.meta.json'), 'w'), indent=1)
    print(f'\nstored: lmom/data/atom_table.meta.json '
          f'({time.time() - t00:.0f} s)')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
