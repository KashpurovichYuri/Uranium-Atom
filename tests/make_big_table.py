#!/usr/bin/env python
"""make_big_table.py — assembles ALL stored results into one big table
(RESULTS_BIGTABLE.md + RESULTS_BIGTABLE.csv): one row per
(system, route) with every measured column. Terminology per the report
v2 section 0: routes (raw-ROHF/CASCI, sph-ROHF/CASCI, ss-CASSCF,
SA-CASSCF, ...), span dev (the closure block norm), verdicts
span-closed / span-open (no green/broken labels).
Run after any new experiment to refresh:  python tests/make_big_table.py
"""
import os, json, csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ['system', 'route', 'E_root0', 'dE_mHa', 'converged', 'grad',
        'span_dev', 'l_closed', 'ladder_dev', 'L2_root0', 'term_root0',
        'p_pop', 'commutator', 'verdict']
LETTERS = 'SPDFGHIKLMNOQRTUVWXYZ'


def term_of(l2):
    """Term letter from an (approximately integer) L(L+1) value."""
    if l2 is None or l2 == '':
        return ''
    try:
        L = int(round((-1.0 + (1.0 + 4.0 * float(l2)) ** 0.5) / 2.0))
    except (TypeError, ValueError):
        return ''
    if abs(l2 - L * (L + 1)) > 0.05:
        return ''
    return f'{LETTERS[L]}(L={L})' if 0 <= L < len(LETTERS) else ''


def verdict_of(r):
    if r.get('l_closed') and float(r.get('ladder_dev', 1) or 1) < 1e-3:
        return 'span-closed'
    return 'span-open'


def rows():
    R = []
    J = lambda fn: json.load(open(os.path.join(ROOT, 'lmom/data', fn)))
    comm = J('commutator.meta.json')

    # ---- the atom table (Si/Sc/Ti x bases), 4 routes each ----
    ROUTE = {'raw+CASCI': 'raw-ROHF/CASCI', 'sph+CASCI': 'sph-ROHF/CASCI',
             'ss-CASSCF': 'ss-CASSCF', 'SA': 'SA-CASSCF'}
    for name, sysr in J('atom_table.meta.json').items():
        for var, route in ROUTE.items():
            r = sysr.get(var)
            if not r:
                continue
            lad = r.get('ladder') or []
            R.append(dict(
                system=name, route=route,
                E_root0=r.get('e_root0') or r.get('e'),
                span_dev=r.get('invar'), l_closed=r.get('l_closed'),
                ladder_dev=r.get('ladder_dev'),
                L2_root0=lad[0] if lad else None,
                term_root0=term_of(lad[0] if lad else None),
                p_pop=r.get('p_pop'),
                commutator=comm.get(f'{name} sph [green]')
                if route == 'sph-ROHF/CASCI' else None,
                verdict=verdict_of(r)))

    # ---- the carbon 3P reference (sph+CASCI) ----
    try:
        cr = J('carbon_reference.meta.json')
        R.append(dict(system='C (cc-pvdz)', route='sph-ROHF/CASCI',
                      E_root0=cr['e_roots'][0], span_dev=cr['span_invar'],
                      l_closed=cr['l_closed'],
                      ladder_dev=cr['ladder_dev'], L2_root0=2.0,
                      term_root0='P(L=1)', verdict='span-closed'))
    except FileNotFoundError:
        pass

    # ---- the carbon baseline (ss + SA) ----
    for k, r in J('carbon_casscf_baseline.meta.json').items():
        lad = r.get('ladder') or []
        R.append(dict(system='C (cc-pvdz)', route=k,
                      E_root0=r.get('e_casscf'),
                      span_dev=r.get('span_invar'),
                      l_closed=r.get('green'),
                      ladder_dev=r.get('ladder_dev'),
                      L2_root0=lad[0] if lad else None,
                      term_root0=term_of(lad[0] if lad else None),
                      verdict='span-closed' if r.get('ladder_ok')
                      else 'span-open'))

    # ---- the U start matrix (CASCI wave) ----
    for k, r in J('starts_matrix_casci.meta.json').items():
        if 'error' in r:
            R.append(dict(system='U-14e', route=k,
                          verdict=f'FAILED {r["error"][:40]}'))
            continue
        R.append(dict(system='U-14e', route=k, E_root0=r.get('e'),
                      dE_mHa=r.get('dE_mHa'), span_dev=r.get('invar'),
                      l_closed=r.get('l_closed'),
                      ladder_dev=r.get('ladder_dev'),
                      L2_root0=r.get('l2_root0'),
                      term_root0=term_of(r.get('l2_root0')),
                      p_pop=r.get('p_pop'), verdict=verdict_of(r)))

    # ---- the start-matrix CASSCF wave ----
    for k, r in J('starts_casscf.meta.json').items():
        R.append(dict(system='U-14e', route=k + ' [CASSCF]',
                      E_root0=r.get('e'), converged=r.get('converged'),
                      grad=r.get('grad'), span_dev=r.get('invar'),
                      l_closed=r.get('l_closed'),
                      ladder_dev=r.get('ladder_dev'),
                      L2_root0=r.get('l2_root0'),
                      term_root0=term_of(r.get('l2_root0')),
                      p_pop=r.get('p_pop'), verdict=verdict_of(r)))

    # ---- the U route comparison (sph/ss/SA-13/SA-60) ----
    comm_map = {'sph-ROHF/CASCI': comm.get('U sph-CASCI(6,13) [green]'),
                'ss-CASSCF': comm.get('U ss [broken]'),
                'SA-CASSCF-13': comm.get('U SA-13 [broken]')}
    for k, r in J('u14e_casscf_comparison.meta.json').items():
        route = {'sph-ROHF/CASCI': 'sph-ROHF/CASCI',
                 'ss-CASSCF': 'ss-CASSCF', 'SA-13': 'SA-CASSCF-13',
                 'SA-60': 'SA-CASSCF-60', 'raw-ROHF CASCI':
                 'raw-ROHF/CASCI', 'SA-13 CASSCF': 'SA-CASSCF-13',
                 'SA-60 CASSCF': 'SA-CASSCF-60'}.get(k, k)
        R.append(dict(system='U-14e', route=route,
                      E_root0=r.get('e_ground'),
                      converged=r.get('casscf_converged'),
                      span_dev=r.get('invar'), l_closed=r.get('l_closed'),
                      ladder_dev=r.get('ladder_dev'),
                      p_pop=r.get('purity_max'),
                      commutator=comm_map.get(route),
                      verdict=verdict_of(r)))

    # ---- CAS(6,16) and the reference quantities ----
    c16 = J('u14e_cas16.meta.json')
    R.append(dict(system='U-14e', route='sph-ROHF/CASCI(6,16)',
                  E_root0=c16.get('e_root0'),
                  l_closed=True,
                  ladder_dev=c16.get('ladder_dev'),
                  commutator=comm.get('U CAS(6,16) [green]'),
                  verdict='span-closed'))
    R.append(dict(system='U-14e', route='CAS(6,13) 60-root ladder',
                  l_closed=True, ladder_dev=5.7e-8, L2_root0=42.0,
                  term_root0='I(L=6)', verdict='span-closed'))
    return R


def main():
    R = rows()
    with open(os.path.join(ROOT, 'RESULTS_BIGTABLE.csv'), 'w',
              newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        wtr.writeheader()
        for r in R:
            wtr.writerow(r)
    with open(os.path.join(ROOT, 'RESULTS_BIGTABLE.md'), 'w') as f:
        f.write('# The big results table (v2 terminology)\n\n')
        f.write('One row per (system, route). Routes per the report §0; '
                'span dev = the span-closure deviation '
                '||L[outside, act]||; verdict = span-closed (certificates '
                'pass, ladder integer) / span-open (mixing values — '
                'see report §0.4). Generated by `tests/make_big_table.py`'
                '; refresh after any new experiment.\n\n')
        f.write('| ' + ' | '.join(COLS) + ' |\n')
        f.write('|' + '---|' * len(COLS) + '\n')
        def fmt(v):
            if v is None:
                return ''
            if isinstance(v, float):
                if v == 0.0:
                    return '0'
                return f'{v:.3e}' if abs(v) < 1e-3 else f'{v:.9f}'
            return str(v)

        for r in R:
            f.write('| ' + ' | '.join(fmt(r.get(c))[:26] for c in COLS)
                    + ' |\n')
    print(f'wrote RESULTS_BIGTABLE.md/.csv with {len(R)} rows')


if __name__ == '__main__':
    main()
