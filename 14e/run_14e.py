"""
run_14e.py — U atom with the Stuttgart RLC ECP (78 core electrons).

14 explicit electrons: 6s^2 6p^6 (closed subvalence) + 5f^3 6d^1 7s^2
(quintet valence). Plain PySCF ECP (no GRPP). Pipeline (8 rows):
  ROHF -> CASCI pre-flight -> CASSCF(6,13) -> ROHF<-CAS,
plus fixed-occupancy UKS-PBE rows holding 7s^2 6d^1 5f^3 (PBE relaxes
free-occupation UKS to 5f^4): before CAS seeded from ROHF MOs, after CAS
seeded from CASSCF NOs, each in MOM, index-pinned (IDX) and per-cycle
character-lock (CHAR) variants. CHAR rows carry a trailing DIIS stage:
SOSCF freezes the handed (index-based) occupations and can morph their
character — the final DIIS re-engages the composition lock. The rewarm
row re-seeds from the converged CAS-branch CHAR solution and certifies
it to full convergence in a few cycles.

Active space: subshell-completion pick -> 1x7s + 5x6d + 7x5f = CAS(6,13),
SO(3)-complete (required for exact L^2 spectroscopy once the full-space
L^2 implementation lands).
"""
import os, sys
import numpy as np
from pyscf import gto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uranium_pipeline as up

STUTTGART_RLC_ECP78 = '''
ECP
U nelec 78
U ul
2       1.000000000            0.000000000
U S
2       4.063653000          112.920103000
2       1.883995000           15.647500000
2       0.886567000           -3.689971000
U P
2       3.986181000          118.758016000
2       2.000160000           15.077228000
2       0.960841000            0.556720000
U D
2       4.147972000           60.855892000
2       2.234563000           29.280047000
2       0.913695000            4.998029000
U F
2       3.998938000           49.924035000
2       1.998840000          -24.674042000
2       0.995641000            1.389480000
U G
2       3.817422000          -36.040977000
2       0.262501000           -0.090511000
END
'''

BASIS = {'U': gto.parse('''
    U S
        3.09793695E+01  1.0
    U S
        1.93389862E+01  1.0
    U S
        1.19091789E+01  1.0
    U S
        7.28108692E+00  1.0
    U S
        4.52381555E+00  1.0
    U S
        2.69147870E+00  1.0
    U S
        1.62933419E+00  1.0
    U S
        1.02810670E+00  1.0
    U S
        6.05884393E-01  1.0
    U S
        3.55646976E-01  1.0
    U S
        2.01800187E-01  1.0
    U S
        1.24813560E-01  1.0
    U S
        6.38361554E-02  1.0
    U S
        3.29899773E-02  1.0
    U S
        1.68638039E-02  1.0
    U P
        4.17040704E+01  1.0
    U P
        2.47028799E+01  1.0
    U P
        1.44963977E+01  1.0
    U P
        8.57586995E+00  1.0
    U P
        4.98086081E+00  1.0
    U P
        2.84040828E+00  1.0
    U P
        1.59903715E+00  1.0
    U P
        8.64765972E-01  1.0
    U P
        4.57242055E-01  1.0
    U P
        2.38933519E-01  1.0
    U P
        1.18792999E-01  1.0
    U P
        5.70066185E-02  1.0
    U P
        2.69891792E-02  1.0
    U P
        1.27490309E-02  1.0
    U D
        1.38063993E+01  1.0
    U D
        8.06231666E+00  1.0
    U D
        4.66375400E+00  1.0
    U D
        2.63565039E+00  1.0
    U D
        1.45796002E+00  1.0
    U D
        7.74812159E-01  1.0
    U D
        3.72484724E-01  1.0
    U D
        1.71955342E-01  1.0
    U D
        7.61336090E-02  1.0
    U D
        3.20319168E-02  1.0
    U F
        3.68446566E+01  1.0
    U F
        1.98480492E+01  1.0
    U F
        1.07216853E+01  1.0
    U F
        5.70450122E+00  1.0
    U F
        2.92433437E+00  1.0
    U F
        1.43960404E+00  1.0
    U F
        6.73859301E-01  1.0
    U F
        2.94349701E-01  1.0
    U F
        1.13239850E-01  1.0
    U F
        5.45732520E-02  1.0
    ''')}

if __name__ == '__main__':
    up.configure(
        ECP_KIND='std',
        ATOM='U 0 0 0',
        UNIT='B', CHARGE=0, SPIN=4,
        BASIS=BASIS,
        ECP={'U': STUTTGART_RLC_ECP78},
        CORE_MAP={0: 2, 1: 6},           # closed 6s^2 6p^6
        NELECAS=(5, 1), NCAS=13,
        CAS_MULTIPLETS={'s': 1, 'd': 5, 'f': 7},
        TARGET_CFG={'s': 2.0, 'd': 1.0, 'f': 3.0},
        STAGE1=[('ROHF', 'ROHF', ('DIIS', 'SOSCF')),
                ('UKS-PBE', 'UKS', ('DIIS', 'SOSCF')),
                ('UKS-PBE-fixocc-MOM', 'UKS', ('MOM', 'DIIS', 'SOSCF'),
                 'rohf'),
                ('UKS-PBE-fixocc-IDX', 'UKS', ('FIXOCC', 'DIIS', 'SOSCF'),
                 'rohf'),
                ('UKS-PBE-fixocc-CHAR', 'UKS',
                 ('CHAR', 'DIIS', 'SOSCF', 'DIIS'), 'rohf')],
        STAGE2_SEED='rohf',
        STAGE3=[('ROHF<-CAS', 'ROHF', 'casscf', ('DIIS', 'SOSCF')),
                ('UKS-PBE<-CAS', 'UKS', 'casscf', ('DIIS', 'SOSCF')),
                ('UKS-PBE<-ROHF', 'UKS', 'rohf', ('DIIS', 'SOSCF')),
                ('UKS-PBE<-CAS-fixocc-MOM', 'UKS', 'casscf',
                 ('MOM', 'DIIS', 'SOSCF')),
                ('UKS-PBE<-CAS-fixocc-IDX', 'UKS', 'casscf',
                 ('FIXOCC', 'DIIS', 'SOSCF')),
                ('UKS-PBE<-CAS-fixocc-CHAR', 'UKS', 'casscf',
                 ('CHAR', 'DIIS', 'SOSCF', 'DIIS')),
                ('UKS-PBE-fixocc-CHAR-rewarm', 'UKS',
                 'uks_pbe_cas_fixocc_char', ('CHAR', 'DIIS', 'SOSCF', 'DIIS'))],
        ROOT_SECTORS=[((5, 1), 20), ((6, 0), 6)],
        SPECTROSCOPY=True,
        FORCE_COLD_CAS=False,
    )
    up.main()
