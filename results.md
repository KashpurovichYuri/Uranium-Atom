| Method | Conv | Iterations | Tolerance | E, Eh | ΔE, mEh | E, eV | ΔE, eV | ⟨S²⟩ | ⟨Sz⟩ | Configuration | Match to ground | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ROHF | ✔ | DIIS 1 cyc | g 2.4e-06 | -473.54496952 | +55.75 | -12885.8151 | +1.517 | 6.000 | 2.00 | s^1.98 d^1.02 f^3.00 | ✔ | warm verify; stationary: dE_vs_chk 4.5e-10 |
| ROHF-smear* | ✔ | SMEAR 27 cyc | dE 3.4e-10, g 1.4e-07 | -472.65491220 | +945.81 | -12861.5954 | +25.737 | 0.000 | n/a | s^2.01 p^0.17 d^1.90 f^1.92 | ✘ | gaussian smear sigma=0.02 |
| UHF | ✔ | DIIS 1 cyc | g 1.0e-06 | -473.48831714 | +112.41 | -12884.2735 | +3.059 | 6.006 | 2.00 | s^2.00 f^4.00 | ✘ | warm verify; stationary: dE_vs_chk 1.0e-12 |
| UHF-smear* | ✔ | SMEAR 24 cyc | dE 5.4e-10, g 5.4e-07 | -472.65491219 | +945.81 | -12861.5954 | +25.737 | -0.000 | 0.00 | s^2.01 p^0.17 d^1.90 f^1.92 | ✘ | gaussian smear sigma=0.02 |
| UKS-PBE | ✔ | DIIS 1 cyc | g 1.1e-05 | -475.21375587 | -1613.03 | -12931.2251 | -43.893 | 6.009 | 2.00 | s^2.00 d^0.36 f^3.64 | ✘ | warm verify; stationary: dE_vs_chk 7.6e-10 |
| UKS-PBE-smear* | ✔ | SMEAR 14 cyc | dE 8.5e-10, g 1.3e-07 | -475.08960639 | -1488.88 | -12927.8468 | -40.515 | -0.000 | 0.00 | s^1.64 p^0.12 d^1.04 f^3.20 | ✘ | gaussian smear sigma=0.02 |
| UKS-PBE-smear-fs* | ✔ | SMEARFS 14 cyc | dE 7.4e-12, g 1.8e-07 | -475.13228433 | -1531.56 | -12929.0081 | -41.676 | 0.005 | 2.00 | s^1.51 p^0.13 d^1.12 f^3.24 | ✘ | gaussian smear sigma=0.02, fix_spin |
| CASCI | ✔ | — | — | -473.55617357 | +44.55 | -12886.1199 | +1.212 | 6.000 | 2.00 | s^1.91 p^0.05 d^1.04 f^3.00 | ✔ | frozen seed-row MOs |
| CASSCF | ✔ | casscf 4 macro | dE 1.0e-09, g 1.7e-06 | -473.60072282 | +0.00 | -12887.3322 | +0.000 | 6.000 | 2.00 | s^1.84 p^0.13 d^1.04 f^2.99 | ✔ | CAS(6,13) from seed-row MOs; w_max=0.17; dom class s^1 d^1 f^3; relax gain +0.0445 |
| UHF<-CAS | ✔ | DIIS 1 cyc | g 1.2e-06 | -473.55049942 | +50.22 | -12885.9655 | +1.367 | 6.237 | 2.00 | s^1.88 d^1.12 f^3.00 | ✔ | warm verify; stationary: dE_vs_chk 4.8e-12; dE vs ROHF -5.53e-03 (polarized core if cfg matches) |
| UKS-PBE<-CAS | ✔ | DIIS 3 cyc + SOSCF 2 macro | dE 0.0e+00, g 8.4e-06 | -475.21374753 | -1613.02 | -12931.2248 | -43.893 | 6.009 | 2.00 | s^2.00 d^0.36 f^3.64 | ✘ | warm verify; stationary: dE_vs_chk 2.6e-09 |
| UKS-PBE<-CAS-SOSCF | ✔ | SOSCF 2 macro | dE 0.0e+00, g 6.4e-06 | -475.21372812 | -1613.01 | -12931.2243 | -43.892 | 6.010 | 2.00 | s^2.00 d^0.36 f^3.64 | ✘ | warm verify; stationary: dE_vs_chk 2.9e-11 |
| UKS-PBE<-UHF<-CAS | ✔ | DIIS 1 cyc | g 1.2e-05 | -475.21371591 | -1612.99 | -12931.2240 | -43.892 | 6.009 | 2.00 | s^2.00 d^0.36 f^3.65 | ✘ | warm verify; stationary: dE_vs_chk 1.3e-09 |
| UKS-PBE<-UHF<-CAS-SOSCF | ✔ | SOSCF 1 macro | dE 0.0e+00, g 1.9e-05 | -475.21371610 | -1612.99 | -12931.2240 | -43.892 | 6.009 | 2.00 | s^2.00 d^0.36 f^3.64 | ✘ | warm verify; stationary: dE_vs_chk 1.7e-10 |
