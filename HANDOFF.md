# HANDOFF — L²-moment project state (read this first in a fresh context)

## What this project is
Term identification (L, S) for the U-14e atom (Stuttgart RLC ECP78, 14
explicit electrons) via an exact L²/Lz² operator machinery, validated on
carbon first, then Si/Sc/Ti. Everything lives in
`/home/kashpur/pylibgrpp/test/uranium/` (conda env: `pyscf`; run anything
as `python tests/<script>.py`).

**Read `README.md` (v2) first** — it contains: §0 notation +
route names + the ⟨L²⟩ mixing semantics, §1 summary, §2 results
(carbon → atoms → U), §3 the verification chain and the span-closure
condition, §4 every route with steps/code/math, §5 the conceptual Q&A
with assumption flags, §6 energy tables, §7 the code map + traps, §8
open items. Data behind every number: `lmom/data/*.npz + *.meta.json`.
All solutions consolidated: `RESULTS_BIGTABLE.md` (54 rows; regenerate
via `python tests/make_big_table.py`).

## Current state (v2, post-refactor)
- `lmom/` — `angular.py`, `moments.py` (THE engine: exact sector L²/Lz²
  from one-string matrices; trace invariants; `_expand` — renamed from
  `moments2.py` in v2), `expectation.py` (dm route + span/core
  certificates), `reference.py` (spherical_reference, m_average_density,
  force_config, _m_separate default OFF), `diag.py`. `operator.py`
  deleted (historical).
- `tests/` — one script per experiment + `battery_lib.py` (mo_passport,
  n_expectation, n_checks_dm, no_audit, energy_of_density, casci_run,
  sz_expectation, lz_check).
- Terminology discipline (per the user): NO "green/broken" labels in the
  report/tables — use span-closed/span-open + measured numbers
  (energies, multiplicities, deviations). ⟨L²⟩ = 7–84 on non-closed
  solutions is a MIXING diagnostic (Σ wᵢLᵢ(Lᵢ+1)), never "garbage".
- The dm route vs the operator route (report §0.5): they agree IFF the
  span is closed; on open spans dm − operator = the leak (verified
  identity) — a closure diagnostic, and the operator route stays exact
  for CAS states. ⟨Lz⟩ = 0 on real columns is a realness theorem
  (pyscf's CI rejects complex Hamiltonians — NotImplementedError).

## Verified results (do not re-derive)
- CASCI(6,13) on the frozen sph-ROHF reference: E = −50.164365; ground
  ⁵I (L=6, 42.0, 13 roots), ⁵L (72, 17), {⁵S-like + ⁵F ×7}, ⁵K (56, 15);
  ladder 5.7e-8 (60 roots), full-sector 3.7e-12 (16 731 eigenvalues).
- CASCI(6,16) (+7p³): E = −50.168752 (+4.4 mHa); duplicate terms
  CONFIRMED: two ⁵G (9+9) and a second ⁵I (13) — distinct energies,
  identical exact L². Term results are CASCI (frozen orbitals); CASSCF
  yields NO terms anywhere (mixing values 7.2–83.7).
- Every relaxation route opens the span on flexible bases: ss-CASSCF
  (≥4 stationary points −50.4399…−50.4594, |grad| 2–6e-5), SA-CASSCF
  (any averaging size incl. SA-190), post-hoc symmetrization
  (integer-but-wrong-shell, +25 Ha), ensemble-SCF (radial-mixed
  columns), every CASSCF start (8/8 in wave 2). The span-open minima
  are the true variational minima; the symmetric manifold is a saddle
  (depth 0.26–0.29 Ha).
- Energy legitimacy: raw vs sph CASCI on U: 223.5 mHa = 16.8 term gaps
  (the purity premium); ≤3 mHa on light atoms. Freezing 7s²: +8.78 mHa.
- NO audit: all CAS spaces saturated (no missing orbitals, no core
  drain). Frozen-7s refutes the p-theft mechanism (drift persists with
  p-pop ≈ 0).
- Carbon-verified construction: gates B/M/Z exact, UDT 1.3e-15,
  e1-vs-fermionic-algebra 0.0, N-checks 3.6e-14, worked m-average
  example diag(1,1,0) → (2/3)I.
- The start matrix (§4.11): the symmetrized raw-ROHF start reproduces
  the reference EXACTLY (pipeline self-consistency); all UKS starts
  term-invalid (direct: l-smeared; symm: overlap-matching fails at
  0.13–0.32 → +31 Ha wrong spaces).

## The route-independent verdict tool
`tests/test_commutator.py`: ‖[H_CASCI, L²]‖ on the sector — ~1e-13 for
span-closed references, ~5e-2 for span-open ones, ten orders apart,
legitimate on ANY solution (measures span invariance, no CI vectors).
Run it FIRST on any new reference.

## Pending / open (report §8)
1. Constrained (in-loop projected) CASSCF — designed: get_grad = packed
   act×(core+virt) + core×virt rotations; zero the first
   ncas·(ncore+nvir) elements; project the STEP (the default AH
   optimizer leaks through the Hessian off-diagonals). Delivers
   E_constrained ∈ [−50.1644, −50.4548].
2. nroots → 200 in CAS(6,16) (complete the second ⁵L; hunt ⁵F/⁵D
   duplicates).
3. NIST centers-of-gravity comparison (no SOC in the model; average
   J-levels into term centers).
4. U-32e (libgrpp route) — engine ready; sector dims are the obstacle.
5. The m-resolved basis (report §4.2 stmt 2): a complex-capable CI
   route (pyscf's direct_spin1 REJECTS complex Hamiltonians — verified)
   or the completeness-checked multiplet picks / C[r,m]-profile
   diagnostics in the real AO basis.
6. C[r,m]-profile diagnostic on stored drifted solutions — upgrades the
   radial-differentiation mechanism (report §5.3b) to a measurement.
7. Data hygiene: `u14e_rohf.meta.json` and `u14e_rohf_raw.meta.json`
   are corrupt (old-session truncation; superseded by `u14e_rohf_sph`
   and `u14e_rohf_raw_control`).

## Environment facts
- PySCF 2.13.1, numpy 2.5.1, scipy; 128 cores; OMP_NUM_THREADS=64
  typical.
- pyscf traps (all pinned in code comments): ROHF 1-d fractional occ →
  `occa = (occ>0)`; `contract_1e` pack_tril symmetrization (cannot apply
  non-symmetric one-body operators); MOM integerizes occupations;
  Davidson mixes degenerate blocks; MO columns reorder at every
  eigendecomposition (use mo_passport); `mc.get_h1eff()` returns a
  tuple; no `make_ham` in 2.13; natural occupancies = eigenvalues of
  C⁻¹γC⁻ᵀ in the OAO metric, NOT CᵀγC; the stored sph occ vector sums
  to 22 (old-session artifact — never use it for densities).
