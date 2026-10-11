| Method | Conv | Iterations | Tolerance | E, Eh | ΔE, eV | ⟨S²⟩ | ⟨L²⟩ | Configuration | Match | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| ROHF | ✔ | DIIS 100 cyc + SOSCF 18 macro | dE 0.0e+00, g 6.7e-06 | -473.54496952 | +1.517 | 6.000 | n/a | s^1.98 d^1.02 f^3.00 | ✔ | DIIS->SOSCF |
| ROHF-smear* | ✔ | SMEAR 22 cyc | dE 8.6e-10, g 4.1e-07 | -472.65491219 | +25.737 | 0.000 | n/a | s^2.01 p^0.17 d^1.90 f^1.92 | ✘ | gaussian smear sigma=0.02 |
| UHF | ✔ | DIIS 93 cyc | dE 6.8e-10, g 3.1e-05 | -473.48704798 | +3.093 | 6.006 | n/a | s^2.00 f^4.00 | ✘ | DIIS->SOSCF |
| UHF-smear* | ✔ | SMEAR 21 cyc | dE 8.3e-10, g 1.1e-06 | -472.65491218 | +25.737 | -0.000 | n/a | s^2.01 p^0.17 d^1.90 f^1.92 | ✘ | gaussian smear sigma=0.02 |
| UKS-PBE | ✔ | DIIS 100 cyc + SOSCF 26 macro | dE 0.0e+00, g 9.1e-06 | -475.21375583 | -43.893 | 6.009 | n/a | s^2.00 d^0.36 f^3.64 | ✘ | DIIS->SOSCF |
| UKS-PBE-smear* | ✔ | SMEAR 14 cyc | dE 8.3e-10, g 1.3e-07 | -475.08960639 | -40.515 | -0.000 | n/a | s^1.64 p^0.12 d^1.04 f^3.20 | ✘ | gaussian smear sigma=0.02 |
| UKS-PBE-smear-fs* | ✔ | SMEARFS 14 cyc | dE 1.2e-10, g 1.9e-07 | -475.13228433 | -41.676 | 0.005 | n/a | s^1.51 p^0.13 d^1.12 f^3.24 | ✘ | gaussian smear sigma=0.02, fix_spin |
| CASCI | ✔ | — | — | -473.55158044 | +1.337 | 6.000 | n/a | s^1.93 d^1.06 f^3.00 | ✔ | frozen seed-row MOs |
| CASSCF | ✔ | casscf 26 macro | dE 8.0e-10, g 3.2e-06 | -473.60072282 | +0.000 | 6.000 | 49.9 | s^1.84 p^0.13 d^1.04 f^2.99 | ✔ | CAS(6,13); Term: ^5K |
| UHF<-CAS | ✔ | DIIS 91 cyc | dE 4.3e-10, g 2.5e-05 | -473.54641392 | +1.478 | 6.218 | n/a | s^1.89 d^1.11 f^3.00 | ✔ | DIIS->SOSCF |
| UKS-PBE<-CAS | ✔ | DIIS 100 cyc + SOSCF 32 macro | dE 0.0e+00, g 6.9e-06 | -475.21374494 | -43.893 | 6.009 | n/a | s^2.00 d^0.36 f^3.64 | ✘ | DIIS->SOSCF |
| UKS-PBE<-CAS-SOSCF | ✔ | SOSCF 29 macro | dE 0.0e+00, g 1.6e-05 | -475.21374494 | -43.893 | 6.009 | n/a | s^2.00 d^0.36 f^3.64 | ✘ | SOSCF |
| UKS-PBE<-UHF<-CAS | ✔ | DIIS 100 cyc + SOSCF 15 macro | dE 0.0e+00, g 2.2e-05 | -475.21371417 | -43.892 | 6.009 | n/a | s^2.00 d^0.36 f^3.64 | ✘ | DIIS->SOSCF |
| UKS-PBE<-UHF<-CAS-SOSCF | ✔ | SOSCF 15 macro | dE 0.0e+00, g 1.0e-05 | -475.21371417 | -43.892 | 6.009 | n/a | s^2.00 d^0.36 f^3.64 | ✘ | SOSCF |
