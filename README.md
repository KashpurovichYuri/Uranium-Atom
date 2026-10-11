# L²-Moment Analysis of Atomic Term Structure — Full Review (v2)

**Project**: `uranium` — term identification (L, S) for the U-14e atom
(Stuttgart RLC ECP78, 14 explicit electrons) with an exact L²/Lz² moment
machinery, validated first on carbon, then Si, Sc, Ti.
**Raw data**: every number below is stored in `lmom/data/*.meta.json` (the
pointer is given at each table); nothing here is recomputed by hand.
**Companion documents**: `HANDOFF.md` (context switch), `RESULTS_BIGTABLE.md`
(all solutions in one table).

---

## 0. Notation and preliminaries (read this first)

### 0.1 Objects and symbols

| symbol | meaning |
|---|---|
| dm, γ | the spin-summed one-particle density matrix; γ = Σ nᵢ nᵢnᵢᵀ over occupied spin-orbitals |
| F | the Fock operator, F = h + G[γ] (one-electron part + Coulomb–exchange functional of γ) |
| E_pq | = a†_p a_q, the spin-summed one-body excitation operator (E = E^α + E^β); the lift of a one-electron operator to Fock space, so "sum over electrons" is built in and N never appears explicitly |
| l, l_k | the one-electron angular momentum operator r̂ × ∇ and its Cartesian components k = x, y, z |
| sector | the Fock-space block with FIXED (n_α, n_β) — exactly the space CASCI solves in; a CI vector is a matrix Ψ[a-string, b-string] whose dimension is C(n,n_α)·C(n,n_β). Here C(n,k) = n!/(k!(n−k)!) is the binomial coefficient — the number of ways to place k electrons in n spin-orbitals, i.e. the number of distinct occupation strings of that spin; the sector dimension = (#α-strings) × (#β-strings). Example, U CAS(6,13): C(13,5)·C(13,1) = 1287·13 = 16 731 |
| W | the matrix of L² (or Lz²) inside one sector — "the sector L² matrix" |
| string | an occupation bitmask (bit p = orbital p occupied); `e1_matrix` builds the matrix of Σ l_k[p,q] E_pq acting on single strings |
| l-purity | an MO column is l-pure when ⟨l²⟩ = l(l+1) exactly |
| (radial × m) separability | an l-pure column whose coefficients C[r, m] over the shell block have rank 1 (one radial function × one spherical harmonic); rank ≥ 2 = an "averaged polynomial" |
| ladder | the multiset of L(L+1) eigenvalues over a set of CI roots; "exact ladder" = integer values with (2L+1) multiplicities |

**Slater–Condon rules** — the standard reduction of operator matrix elements
between Slater determinants: for a one-body operator, the element between
two determinants is a sum of one-electron integrals over the differing
orbital (equal determinants: the sum over all occupied orbitals).
`moments.e1_matrix` implements exactly this for Σ l_k[p,q] E_pq, sign
conventions included.

**PSD (positive semidefinite)** — a Hermitian matrix whose eigenvalues are
all ≥ 0. The sector L² matrix is a literal square of a Hermitian matrix
(the per-component E·E form), so PSD holds by construction; a negative
eigenvalue anywhere is an immediate construction bug (this is exactly the
historical diagnostic that exposed the early one-body-only route).

### 0.2 The routes (names used everywhere below)

1. **raw-ROHF/CASCI** — CASCI in the canonical MOs of a plain ROHF run (no
   symmetrization).
2. **sph-ROHF/CASCI** — ROHF, then the m-averaged-pinned configuration
   (`force_config`), then `spherical_reference` (m-average the density,
   re-diagonalize F → pure-l MOs), then CASCI with FROZEN orbitals. This is
   **the reference route** of the project.
3. **ss-CASSCF** — state-specific CASSCF (orbitals optimized for one root).
4. **SA-CASSCF** — state-averaged CASSCF over the first k roots with equal
   weights 1/k.
5. **post-hoc symmetrization** — take a finished CASSCF solution, project its
   MOs back to l-pure form, re-run CASCI (optionally re-start CASSCF).
6. **ensemble-SCF** — SCF with the fractional shell-m-averaged occupations
   (a self-consistent version of the pinned density).
7. **larger CAS** — CAS(6,16) = CAS(6,13) + the virtual 7p³.

**m-averaged configuration pinning vs spherical_reference** (two distinct
steps): pinning fixes the *occupations* (the m-averaged Hund pattern, e.g.
5f⁷ spread as 1.0 e over seven 5f columns) so the SCF target configuration is
unambiguous; `spherical_reference` fixes the *orbitals* (it m-averages the
resulting density within each shell and re-diagonalizes F), which restores
l-pure eigenvectors. Pinning without averaging leaves anisotropic orbitals;
averaging without pinning can target the wrong configuration.

### 0.3 The central caveat — read before any number

The machinery (`lmom/moments.py`) computes the sector L² matrix **on any
orbital span whatever**. The guarantee that the outcome is an exact integer
ladder L(L+1) with (2L+1) multiplicities holds **only when the span is
l-closed** (L maps the span into itself; see §3.2). On a non-closed span the
code still runs and returns exact expectation values — but those values are
**mixing diagnostics, not term labels** (§0.4). Every route in §4 is
therefore classified by one measured quantity: the span-closure deviation.

### 0.4 The semantics of ⟨L²⟩ (term vs mixing value)

For an eigenstate of L², ⟨L²⟩ = L(L+1) exactly and the term label follows
(L = 6 → ⁵I etc.). For a mixture |Ψ⟩ = Σᵢ wᵢ|Lᵢ⟩ (wᵢ ≥ 0, Σwᵢ = 1):

⟨L²⟩ = Σᵢ wᵢ Lᵢ(Lᵢ+1),

i.e. the weighted average of the participating terms' L(L+1). The value is
never "wrong" — it is a **composition readout** that constrains which terms
participate. Examples measured in this project: 48.97 = 42 + 0.498 × (56−42)
→ an ⁵I-dominant state with ≈50% ⁵K weight (a two-term reading); 83.7 > 72
→ requires components beyond the ⁵L channel (L ≥ 9); 33.3 sits between ⁵G
(20) and ⁵I (42). Non-integer values are the *expected output* on
non-closed spans, and their magnitude is informative.

### 0.5 The two evaluation routes (behind every results table)

1. **The operator route** (`moments.py`): build the sector matrix W from
   the one-string matrices of the ACTIVE columns and take ⟨Ψ|W|Ψ⟩ (or
   eigvalsh of the root-space matrix). Fully active-restricted — and for
   a CAS state that restriction is EXACT on any span: matrix elements
   connecting the active columns to core/virtual vanish identically
   (they create determinants orthogonal to the CAS basis).
2. **The dm route** (`expectation.py`): contract the active-space 1- and
   2-RDM with the l-matrices, using COMPLETE intermediates in the
   one-body piece (the contraction runs through ALL columns, then is
   restricted). Exact when the span is closed (then the two routes
   coincide); on an open span it exceeds the true value by the **leak**
   through the outside columns — verified as an identity on synthetic
   open spans: dm − operator = Σ_k ⟨(L_k L_k)[act,act] −
   (L_k[act]L_k[act])⟩ contractions, exactly the block the span
   certificate flags (Si raw: +2.0; Ti raw: +12).

So: the two routes compute the same physical ⟨L²⟩ and **agree iff the
span is closed**; their disagreement is itself a closure diagnostic, and
the operator route remains exact for CI states in either case.

**A realness identity used throughout**: every real orbital column has
⟨Lz⟩ = 0 exactly — ⟨Lz⟩ = i ΨᵀAΨ with A real antisymmetric. It is a
theorem about REAL columns, not physics information; the informative
partner is ⟨Lz²⟩ (§4.2, statement 2).

**Conventions and pinned warnings** (moved here from the methods
chapter; do not violate in the moment pipeline): the dm2 pairing
PAIRING='pqrs', SIGN=+1 follows from the standard Wick expansion of
⟨a†_p a_q a†_r a_s⟩ and is *validated* (not defined) by the 1e-13
agreement with the operator route on carbon — same logic as the
e1_matrix signs, checked against brute-force fermionic algebra (0.0).
pyscf's `contract_1e` packs its one-body integrals with `pack_tril` and
*silently symmetrizes* — it cannot apply the non-symmetric
complex-Hermitian l-matrices; `contract_2e` conventions were abandoned
in favor of the exact one-string construction.

---

## 1. Executive summary

**The instrument.** `lmom/moments.py` builds the sector L²/Lz² matrix
*exactly*: per spherical component k,
L² = Σ_k Σ_pqrs l_k[p,r] l_k[q,s] E_pq E_rs = Σ_k (E_k^α E_k^α + E_k^β E_k^β + 2 E_k^α E_k^β),
assembled from one-string Slater–Condon matrices, PSD by
construction, O(n_str³) per application. The headline results below became
available only after this construction replaced the earlier one-body
operator route (whose symptoms — the {4×9} anomaly, non-integer full-sector
spectra, eigenvalue −2 — are historical and appear nowhere in the current
numbers; §4.9).

**The headline physics (U-14e).**
- **CASCI(6,13)** = 7s¹6d⁵5f⁷ on the FROZEN sph-ROHF reference orbitals:
  E(ground) = **−50.164365**, ground term **⁵I (L=6, ⟨L²⟩ = 42.0, 13
  roots)**, then **⁵L (L=8, 72, 17 roots)**, then {L=0 ×1 + ⁵F ×7}, then
  **⁵K (L=7, 56, 15 roots)** — all complete multiplets, exact m²
  composition, exact (2L+1) degeneracies (`u14e_roots60.meta.json`).
- **CASCI(6,16)** = + virtual 7p³: E(ground) = **−50.168752** (the 7p
  correlation gains 4.4 mHa). The **predicted duplicate terms appeared**:
  two distinct ⁵G terms (9+9 roots) and a second ⁵I (13 roots) at different
  energies but identical, exactly integer L² — CASCI(6,16), again on the
  same frozen reference (`u14e_cas16.meta.json`).
- **No relaxation route yields terms.** ss-CASSCF and SA-CASSCF (any start,
  any averaging size up to the full 190-root sector) return roots with
  ⟨L²⟩ = 7–84 — *mixing values* (§0.4), so they are not terms. The
  duplicate ⁵L/⁵K structure of CASCI(6,13) has no CASSCF counterpart: with
  optimized orbitals the span is not l-closed and no term labels exist.

**Energy deviations (the price of purity, all measured).** CASCI on the raw
basis: −50.387840 — **223.5 mHa below** the spherical CASCI, i.e. 16.8
term gaps (⁵I→⁵L = 13.3 mHa) of variational advantage for giving up term
identity. The CASSCF minima descend further (−50.4548…−50.4594), i.e.
0.26–0.29 Ha below the spherical reference: **the span-open solutions are the
true variational minima; the symmetric manifold is a saddle**. Ensemble-SCF
lands at −49.4897 — above the spherical CASCI, because it enforces the
ensemble (fractional) density at SCF level while sacrificing radial purity.

**The route that works everywhere.** sph-ROHF/CASCI + certificates:
verified on U-14e, carbon, Si, Sc, Ti (sto-3g and def2-tzvp), with exact
integer ladders everywhere the span certificate passes.

---

## 2. Results (carbon first — the verification bed — then atoms, then U)

### 2.1 Carbon (cc-pvdz): the verification bed and its gates

Carbon is where every new construction and every convention is proven
before it touches uranium. The gates (`tests/test_carbon_verification.py`,
`tests/test_mini_gates.py`, data in `mini_gates.meta.json` and the carbon
baseline `carbon_casscf_baseline.meta.json`):

1. **Gate B (the term gate, ³P).** CASCI of carbon in its p² space on
   the spherical reference (3 roots); every root must give ⟨L²⟩ = 2
   (L = 1, ³P) through BOTH evaluation routes (§0.5). Measured:
   ⟨L²⟩ = 2.0 on all 3 roots; route agreement to 6+ digits
   (`carbon_reference.meta.json`: E(ground) = −37.655041, span 9.4e-16).
2. **Gate M (the multiplet gate).** The (n_α, n_β) = (1,1) sector of
   the CAS(2,3) p-block: 9 determinants; the exact L² spectrum of the
   ORBITAL coupling of two electrons in a p shell is {L(L+1)} = {0 ×1,
   2 ×3, 6 ×5}. CAUTION: the multiplicities 1, 3, 5 are **2L+1** (the
   M_L components of the orbital multiplets L = 0, 1, 2 — S, P, D),
   NOT spin degeneracies: L² is spin-independent, the sector has M_S = 0
   fixed, and no ⁵D spin state exists here (two electrons cannot carry
   S = 2). Measured: exact to 1e-11.
3. **Gate Z (the Lz² gate).** In a complete degenerate eigenspace the Lz²
   spectrum must be the multiset {m²} = {0 ×3, 1 ×4, 4 ×2}. Measured:
   exact.
4. **UDT (the full-sector spectrum test).** ALL eigenvalues of the sector
   L² matrix (dense diagonalization) must be integers; historically the
   test that exposed the early construction bugs. Now: max deviation
   **1.3e-15**.
5. **The mini-gates (the hand-checkable miniature).**
   `mini_gates.meta.json`:
   - *Worked m-average*: the M_L-selected determinant density in the
     p-block, dm_p = diag(1, 1, 0) (px, py occupied), m-averages to
     **(2/3)·I₃** with anisotropy 0.5774 = 1/√3 — the exact group average
     of an anisotropic shell density.
   - *Incomplete-shell gates*: BOTH CAS(2,2) spans — the two-column
     picks ({px, py} and {px, pz}) of the 3-column p-block — fail
     closure (‖L[outside, span]‖ = 1.0: L leaks into the missing
     column) and their "ladder" is [0]: the projected in-span value,
     NOT the physical ⟨L²⟩ = 2. **An incomplete span gives meaningless
     L² even when numerically integer** — the miniature of the whole
     project's lesson (contrast: an L = 0 value in a CERTIFIED-CLOSED
     span, e.g. the ⁵S-channel root of §2.3, is a genuine term).
6. **Carbon CASSCF behaviour** (the carbon ↔ uranium mechanism, §4.4):
   ss-CASSCF E = −37.682418 with ⟨L²⟩ = 1.9176 (dm route: 2.0000) — a
   4% deficit from the ground root mixing excited-L character;
   SA-CASSCF-3 and SA-9 keep span closure (2.2e-6, 3.0e-6) and give the
   exact ladder {2, 2, 2} (dev 1e-11), E = −37.682391 / −37.633970.
   Note the symmetry advantage exists already here: the mixed ss
   solution is 2.7e-5 Ha BELOW the SA-3 average — the same (small)
   anisotropic gain that becomes 0.26–0.29 Ha on uranium.

**Why the two routes agree (the cross-validation logic).** Both routes
are defined in §0.5; each follows from the SAME exact second-quantized
identity for L², derived independently: the operator route via the
one-string Slater–Condon algebra, the dm route via the Wick expansion of
⟨L²⟩ in terms of the 1- and 2-RDM (the PAIRING/SIGN conventions are the
latter's bookkeeping, fixed by theory and *validated* — not defined — by
the comparison). Their implementations are disjoint, so the agreement to
~1e-14 on closed spans validates the conventions and sign rules, not
merely each other; and the identity dm − operator = leak (§0.5) makes
their disagreement on open spans a *feature* — a closure diagnostic. A
built-in sanity gauge: the same E1 machinery applied to the identity
matrix reproduces ⟨N⟩ and ⟨N²⟩ exactly (dev 3.6e-14, §3.1 check 6).

### 2.2 The atom table: C, Si, Sc, Ti

(`tests/test_atoms_table.py`, `atom_table.meta.json`; carbon reference
`carbon_reference.meta.json`; full rows in `RESULTS_BIGTABLE.md`.)
Homogeneous table — E is root 0 for CASCI routes and the state-average
for SA-CASSCF; "span dev" = the span-closure deviation
‖L[outside, act]‖ (§3.2). Raw rows have a SINGLE CASCI root, so their
one ladder value can look integer while the span is open and the term
ladder is absent — marked "(1 root)". "(mixed)" = mixing values (§0.4).

| system | route | E / Ha | E / eV | span dev | ladder max dev | terms |
|---|---|---|---|---|---|---|
| C cc-pvdz | sph-ROHF/CASCI | −37.655041 | −1024.6459 | 9.4e-16 | exact | ³P (3 roots) |
| C cc-pvdz | ss-CASSCF | −37.682418 | −1025.3908 | 2.0e-1 | 8.2e-2 | mixed (³P ⊕ ~4% excited-L) |
| C cc-pvdz | SA-CASSCF-3 | −37.682391 | −1025.3901 | 2.2e-6 | 1e-11 | ³P (3 roots) |
| Si sto-3g | raw-ROHF/CASCI | −285.466211 | −7767.9313 | 9.6e-1 | 0 (1 root) | (mixed) |
| Si sto-3g | sph-ROHF/CASCI | −285.466144 | −7767.9295 | 5.4e-16 | 4.4e-16 | ³P (3 roots) |
| Si sto-3g | ss-CASSCF | −285.466211 | −7767.9313 | 2.6e-3 | 1.4e-5 | ³P ⊕ 1e-5 admixture |
| Si sto-3g | SA-CASSCF | −285.466187 | −7767.9307 | 5.3e-16 | 4.4e-16 | ³P (3 roots) |
| Si def2-tzvp | raw-ROHF/CASCI | −288.846568 | −7859.9155 | 9.7e-1 | 0 (1 root) | (mixed) |
| Si def2-tzvp | sph-ROHF/CASCI | −288.847919 | −7859.9523 | 1.9e-13 | 4.9e-14 | p-block: ¹S ⊕ ³P (L=0,1) |
| Si def2-tzvp | ss-CASSCF | −288.854613 | −7860.1344 | 3.4e-1 | 2.1e-3 | (mixed) |
| Si def2-tzvp | SA-CASSCF | −288.329153 | −7845.8359 | 3.8e-9 | 3.5e-13 | ladder exact (SA avg) |
| Sc sto-3g | raw-ROHF/CASCI | −752.001200 | −20462.9951 | 7.6e-1 | 0 (1 root) | (mixed) |
| Sc sto-3g | sph-ROHF/CASCI | −751.984318 | −20462.5357 | 1.7e-15 | 7.1e-15 | ²D + ²P |
| Sc sto-3g | ss-CASSCF | −752.001200 | −20462.9951 | 6.3e-3 | 3.7e-5 | (mixed, near-closed) |
| Sc sto-3g | SA-CASSCF | −751.994538 | −20462.8138 | 1.2e-10 | 1.4e-13 | ladder exact |
| Sc def2-tzvp | raw-ROHF/CASCI | −759.723147 | −20673.1200 | 9.9e-1 | 0 (1 root) | (mixed) |
| Sc def2-tzvp | sph-ROHF/CASCI | −759.721663 | −20673.0796 | 6.7e-14 | 3.8e-14 | ²D |
| Sc def2-tzvp | ss-CASSCF | −759.723146 | −20673.1200 | 2.1e+0 | 7.5e-2 | (mixed) |
| Sc def2-tzvp | SA-CASSCF | −759.434254 | −20665.2588 | 3.0e+0 | 1.1e-3* | (mixed)* |
| Ti sto-3g | raw-ROHF/CASCI | −839.562751 | −22845.6663 | 1.6e+0 | 0 (1 root) | (mixed)** |
| Ti sto-3g | sph-ROHF/CASCI | −839.562751 | −22845.6663 | 3.0e-14 | 1.2e-14 | ³F + ³P |
| Ti sto-3g | ss-CASSCF | −839.562751 | −22845.6663 | 3.0e-14 | 2.3e-10 | ³F + ³P (stays closed) |
| Ti sto-3g | SA-CASSCF | −839.547585 | −22845.2536 | 3.0e-14 | 2.3e-14 | ladder exact |
| Ti def2-tzvp | raw-ROHF/CASCI | −848.391432 | −23085.9069 | 1.0e+0 | 0 (1 root) | (mixed) |
| Ti def2-tzvp | sph-ROHF/CASCI | −848.394361 | −23085.9866 | 2.9e-14 | 1.3e-13 | complete d² ladder (190 roots, L=0…4) |
| Ti def2-tzvp | ss-CASSCF | −848.396814 | −23086.0534 | 1.7e+0 | 2.0e-1 | (mixed) |
| Ti def2-tzvp | SA-CASSCF | −848.118777 | −23078.4876 | 9.7e-1 | 4.8e-1 | (mixed) |

\* the Sc def2 SA footnotes: the dm and operator routes disagree by 12
on this solution (§0.5 — the leak diagnostic); its ladder dev is
solver-luck (6.4e-9 in the earlier recorded run, 1.1e-3 in the fresh
run) — integrality without the certificates is never a verdict.
\** the Ti sto raw row: this run's Aufbau ROHF landed in a different
basin than the earlier recorded run (the ROHF multi-solution issue,
§4.1), coincidentally at the spherical energy — the span is still open
(dev 1.6), so the (1-root) ladder [0] is again an in-span projection,
not a term.

Reading: **sph-ROHF/CASCI is term-exact in all seven systems**. CASSCF
stays closed on minimal bases (Si sto near-closed at 2.6e-3, Ti sto at
3.0e-14 — there the symmetric point IS the global minimum, so ss
converges onto it and shares its energy) and opens the span on flexible
bases (def2 rows; the mechanism — §4.4).

### 2.3 U-14e

**CASCI(6,13), 60 roots, solver conv 1e-13** (`tests/test_u14e_roots.py`,
`u14e_roots60.meta.json`). Reference E(ground) = **−50.164365**; the
M_L-degeneracy spread within every term cluster is ~1e-12 Ha, as spherical
symmetry demands.

| E-order | term | E (center) / Ha | ΔE / mHa | E / eV | ΔE / eV | roots | complete (2L+1) | m² composition |
|---|---|---|---|---|---|---|---|---|
| 1 | **⁵I (L=6, 42)** — ground | −50.164365 | 0 | −1365.0419 | 0.0000 | 13 | ✓ | {0,1,4,…,36} exact |
| 2 | ⁵L (L=8, 72) | −50.151030 | +13.33 | −1364.6790 | 0.3627 | 17 | ✓ | {0,1,…,64} exact |
| 3 | {⁵K (L=7, 56)} | −50.141520 | +22.85 | −1364.4203 | 0.6218 | 15 | ✓ | {0,1,…,49} exact |
| 4 | {L=0 ×1, ⁵F (12) ×7} | −50.139515 | +24.85 | −1364.3657 | 0.6762 | 8 | ✓ | — |
| 5 | ⁵G (L=4, 20) | −50.128587 | +35.78 | −1364.0683 | 0.9736 | 7 of 9 | window cut | — |

(E per cluster from the stored per-root energies `u14e_roots60.npz`;
intra-cluster M_L degeneracy spread ≤ 0.07 mHa — spherical symmetry.)
The 60-root ladder is integer to **5.7e-8**; the FULL sector spectrum
(all 16 731 eigenvalues, dense) is integer to **3.7e-12**
(`u14e_verify.meta.json`). No spin–orbit coupling in the model, so NIST
J-levels must be averaged into term centers of gravity before any
ordering comparison.

**CASCI(6,16) = +7p³** (`tests/test_u14e_cas16.py`,
`u14e_cas16.meta.json`; sector 69 888 determinants, 100 roots).
Motivated by the population analysis: the CASSCF solutions (ss and SA
alike) carry ~1–3.5 e of p-character, essentially all in the virtual 7p
(§4.5). E(ground) = **−50.168752** (the 7p correlation gains 4.4 mHa
over CAS(6,13)).

| E-order | term | E (center) / Ha | ΔE / mHa | E / eV | ΔE / eV | roots |  |
|---|---|---|---|---|---|---|---|
| 1 | ⁵I (L=6) — ground | −50.168752 | 0 | −1365.1613 | 0.0000 | 13 ✓ |  |
| 2 | ⁵L (L=8) | −50.152014 | +16.74 | −1364.7058 | 0.4555 | 17 ✓ |  |
| 3 | {L=0 ×1, ⁵F ×7} | −50.143695 | +25.06 | −1364.4795 | 0.6819 | 8 ✓ | excited configuration |
| 4 | ⁵K (L=7) | −50.143130 | +25.62 | −1364.4641 | 0.6972 | 15 ✓ |  |
| 5 | **⁵G (first)** | −50.130747 | +38.01 | −1364.1271 | 1.0343 | 9 ✓ |  |
| 6 | **⁵G (second term!)** | −50.130335 | +38.42 | −1364.1159 | 1.0455 | 9 ✓ | distinct energy, same L² |
| 7 | **⁵I (second term!)** | −50.127384 | +41.37 | −1364.0356 | 1.1257 | 13 ✓ | new f³ parentage |
| 8 | ⁵H (L=5) | −50.123188 | +45.56 | −1363.9214 | 1.2398 | 11 ✓ |  |
| 9 | ⁵L (again) | −50.120224 | +48.53 | −1363.8408 | 1.3206 | 5 of 17 | window cut |

**The duplicate-term prediction was confirmed**: two ⁵G terms and two ⁵I
terms at *distinct energies* with the *same, exactly integer* L². (The
second ⁵I carries a different 5f³ parentage than the ground ⁵I — the
same total coupling assembled from different subshell term schemes.)
Definitions of the two quality numbers: *cluster dev* = max |λ −
round(λ)| over the roots of one multiplet cluster (~1e-10 here);
*Lanczos-vs-dense* = the same W diagonalized iteratively vs by full dense
diagonalization, agreement 6.8e-14. This is the completeness of the
(2L+1) structure: nothing but CASCI(6,16) on the frozen spherical
reference was needed — and none of it survives orbital relaxation (§4.4:
CASSCF in the same space returns mixing values 7.2–48.8).

### 2.4 Carbon ↔ uranium (the same mechanism at two scales)

All energies in Hartree (Ha) throughout the report; the tables also
give eV (1 Ha = 27.2114 eV = 219 476 cm⁻¹). For scale, the ⁵I→⁵L gap =
13.33 mHa = 0.3627 eV ≈ 1 Ha/75.

| | carbon (cc-pvdz) | U-14e (ECP78) |
|---|---|---|
| sph-ROHF/CASCI | ³P exact | ⁵I, ⁵L, ⁵K, ⁵F, ⁵S-like exact (§2.3) |
| ss-CASSCF ⟨L²⟩ | 1.9176 (³P ⊕ ~4% excited-L) | 7.2–83.7 (mixing values, §0.4) |
| SA-CASSCF | stays closed (2e-6) | opens (1.5–2.3 span dev) |
| stationary-point depth | 2.7e-5 Ha | 0.26–0.29 Ha below |
| mechanism | Davidson's arbitrary basis inside the degenerate ³P mixes excited-L character into the ss root (§4.4) | the same, amplified: many near-degenerate valence shells give the anisotropic descent a deep, reachable minimum |

The anisotropy curves (§4.5) show the mechanism directly on both:
constant 7.55e-2 for carbon SA (a fixed symmetric stationary point)
versus a seed-and-descend transient 3.4e-4 → 0.20 for U SA-13. What
scales with the basis is not the m-anisotropy (already present at HF
level, removed by the m-average of §4.2) but the freedom of the
CASSCF columns to leave the l-structure: on def2-class bases the columns
can differentiate *radially* within a shell and steal polarization
character — the channels measured in §4.4 and interpreted in §5.3b.

---

## 3. The verification chain — what is checked, why, and when it can fail

### 3.1 The invariants

Two families. **Code invariants** validate the construction on any input
(solution-independent — if these fail, the *code* is wrong). **Physics
invariants** hold if and only if the span is l-closed (if these fail, the
*space* is wrong). The suite deliberately separates the two questions.

| # | check | type | why it must hold (math) | measured | data |
|---|---|---|---|---|---|
| 1 | Tr W analytic vs Σ λᵢ | code | Tr W_k = n_b·Tr(E^α²) + n_α·Tr(E^β²) + 2 Tr(E^α)Tr(E^β) per component — an independent closed-form route to the sum of eigenvalues; construction identity, holds always | dev 1.2e-10 (U, 16 731 eigenvalues) | `u14e_verify` |
| 2 | Tr W² (Frobenius) vs Σ λᵢ² | code | second moment of the same identity | 1.5e-8 | `u14e_verify` |
| 3 | full-sector integrality | physics | a complete-l-shell span carries the SO(3) representation theory: the sector decomposes into irreducible (2L+1)-multiplets, so the spectrum of W is exactly {L(L+1)} with (2L+1) multiplicities. Fails iff the span is open (mixing) | 3.7e-12 (U full sector); 1.3e-15 (carbon UDT) | `u14e_verify` |
| 4 | Lanczos vs dense eigenvalues | code | two eigen-solvers on the same matrix | 5.6e-13 | `u14e_roots60` |
| 5 | dm-route vs operator-route | code | the same exact ⟨L²⟩ derived twice (§0.5): they agree iff L[act,out] = 0; on open spans dm − operator = the leak (verified identity) | ~1e-14 (closed spans); +2.0/+12 (raw spans) | per-run metas |
| 6 | N-checks: Tr γ, ⟨N̂⟩, ⟨N̂²⟩ | code | the identity matrix through the same E1 machinery must give the exact particle number and its square | 3.6e-14 | `u14e_energy_ncheck` (n_checks_max_dev) |
| 7 | [H_CASCI, L²] commutator norm | physics | the theorem premise, made measurable (§3.3); finite iff the span is open | 1e-15–4.2e-13 closed; 4.5e-2–8.1e-2 open | `commutator` |
| 8 | multiplet completeness + degeneracies | physics | the (2L+1) multiplicity and the M_L E-degeneracy ~1e-12 Ha are exact consequences of spherical symmetry of H | exact on closed spans | `u14e_roots60`, `u14e_cas16` |
| 9 | e1_matrix vs explicit fermionic algebra | code | validates the IMPLEMENTATION of the (known-correct) Slater–Condon rules — sign conventions are where string-algebra code goes wrong; the math itself is not in question | 0.0 on random complex matrices | pinned at construction |

*Are they always satisfied?* Checks 1–2, 4–6, 9: yes, by construction, on
every input — they are what makes the engine trustworthy. Checks 3, 7, 8:
exactly when the span is l-closed; on open spans they fail in a
quantitative, interpretable way (mixing values, finite commutator). The
one number that decides which regime you are in is the span deviation —
hence §3.2.

**The operator construction's correctness record.** Demonstrated
carbon-first — the gates of §2.1 (B/M/Z/UDT), the e1-vs-fermionic-algebra
check 0.0, the N-checks — and by checks 1–2 and 4–6 on every system
since. The construction (per-component E·E, §1) is used nowhere with a
known discrepancy; the pyscf `contract_1e`/`contract_2e` conventions are
excluded from the pipeline for the reasons pinned in §0.5.

### 3.2 Span closure — what it is and why it is THE condition

**Definition.** Let P project onto the active span (the CAS columns). The
span is **l-closed** when L P = P L P, i.e. L maps the span into itself:
‖L[outside, act]‖ ≈ 0. The certificate
(`expectation.span_certificate`) measures exactly this block norm, plus
the per-column ⟨l²⟩ = l(l+1) purity.

**Why it is the condition.** For an l-closed span and a closed l-pure
core, φ, χ in the span:
⟨φ|[P H P, L²]|χ⟩ = ⟨φ|[H, L²]|χ⟩ = 0
(the P·φ = φ because P φ stays in the span; H is spherical so it commutes
with L²). H_CASCI and L² are then simultaneously diagonalizable — the
CASCI roots *are* term states with integer L(L+1). If the span is open,
P·φ ≠ φ, the first equality dies, and nothing forces integrality: the
eigenvalues become the mixing values of §0.4. Closure is the hypothesis
of the theorem; the certificates test the hypothesis directly, and the
commutator probe measures the conclusion's premise independently of any
CI vector.

**The refined criterion (m-separability).** L is *radially diagonal* and
mixes only the m index (§4.2, statement 1). Closure therefore requires
the span to contain complete m-sets for every radial function it
contains — column-wise l-purity is NOT sufficient. This is measured, not
assumed: the ensemble-SCF basis has l-pure columns (1.3e-5) yet fails
closure (§4.2, statement 2). The certificate detects this automatically because it
measures L-images, never purities alone.

```python
# expectation.py — the operative test (abridged)
outside = [i for i in range(nmo) if i not in act_cols]
for k in range(3):
    blk = L[k][np.ix_(outside, act_cols)]
    invar_max = max(invar_max, float(np.abs(blk).max()))   # <-- closure
```

### 3.3 The commutator probe (route-independent verdict)

‖[H_CASCI, L²]‖ on the sector, normalized by max(‖Hv‖, ‖Wv‖) on random
vectors. Legitimate for ANY solution — including CASSCF output: H is the
CI Hamiltonian *in whatever span the solution supplies* and W is L² in
the same span; the probe measures span invariance without touching CI
vectors. W genuinely is complex-Hermitian in general (in a real AO basis
each l_k = i × (real antisymmetric)), not just formally: the probe
therefore splits it into its symmetric and antisymmetric REAL parts and
never builds a complex vector. The cost is negligible BY CONSTRUCTION:
everything acts inside the fixed sector via three matrix-vector
products (Wv, Hv, and W(Hv)) on vectors of the sector dimension — no
matrix is ever assembled (§3.3 code):

```python
# test_commutator.py — all-real evaluation of [H,W]v
w = apply_moment(mats, na_str, nb_str, v)          # complex
c_re = H(w.real) - w_parts(H(v)).real              # [H, Wr]v
c_im = H(w.imag) - w_parts(H(v)).imag              # [H, Wi]v
ratio = norm(c_re, c_im) / max(norm(Hv), norm(Wv))
```

Measured (`commutator.meta.json`): span-closed references 1.0e-15 (Si
sto) … 4.2e-13 (U CAS(6,13)); U CAS(6,16) 3.2e-13; span-open solutions
8.1e-2 (U ss) and 4.5e-2 (U SA-13) — **ten orders of separation**. This
is the first check to run on any new reference.

### 3.4 The reading protocol (how to judge a solution)

1. **[H_CASCI, L²] commutator norm** — the theorem premise, measured
   directly (§3.3).
2. **Certificates** (span + core): is the span l-closed, is the core
   closed and disjoint? (`expectation.py`)
3. **L²-ladder integrality** — the drift detector; a non-integer ladder
   on a certified-closed span means degenerate-block mixing (resolvable,
   §4.2 step 6).
4. **dm-vs-operator agreement + trace identity** — construction sanity:
   on a closed span the two routes must coincide (§0.5); a nonzero
   dm − operator IS the leak through the outside columns.
5. **Energies + natural occupancies** — is it the intended physical
   configuration?

Integer ladders are necessary, never sufficient: the wrong-shell spaces
of §4.7 are integer *and* wrong; the Sc def2 SA row of §2.2 shows the
dm-vs-operator disagreement (12) that flags an open span no matter how
integer the ladder looks. The
protocol exists because the failure modes are orthogonal.

---

## 4. The routes, one by one

Every route below is described with the same protocol: what is done
(numbered steps), what comes out (numbers + data file), and where it
fails or holds. Carbon results precede uranium throughout.

### 4.1 raw-ROHF/CASCI (the negative control)

Steps:
1. Converge ROHF with the default Aufbau occupations (a single M_L-selected
   determinant: e.g. 5f⁷ occupies seven *specific* 5f columns).
2. Run CASCI in its canonical MOs, frozen.

What fails, and why: the single-determinental density is not spherical →
the Fock operator F = h + G[γ] is anisotropic → its eigenvectors are
l-smeared → the active span is not l-closed → [P·H·P, L²] ≠ 0 → the
ladder is a continuum of mixing values. Measured: span dev 0.83–2.6 on
every system (`u14e_rohf_raw_control`, `atom_table`, big-table rows).
Code invariants (checks 1–2, 4–6, 9 of §3.1) still pass — the code is
right, the space is not — while the dm and operator routes separate by
exactly the leak (§0.5: +2.0 on Si raw, +12 on Ti raw): the operator
route remains the exact ⟨L²⟩ of the CI root (e.g. 0.0 for the single Si
raw root — an accidental mixture value), the dm route picks up the
outside-column pathways the closed-span derivation excludes. A note on
basins: plain Aufbau ROHF has many near-degenerate solutions and the run
lands in one of them start-dependently (the multi-basin issue behind
the Ti sto row of §2.2). On U, the raw CASCI energy −50.387840 is
*lower* than the spherical −50.164365 by 223.5 mHa: the open span is a
richer variational space — term-blind, which is exactly the point of
§6.

### 4.2 sph-ROHF/CASCI — the reference route

The exact chain (`tests/test_u14e_roots.py`, `lmom/reference.py`):
1. Converge ROHF with the m-averaged pinned configuration
   (`force_config`): the occupations {7s¹, 6d⁵, 5f⁷} are spread equally
   over the (2l+1) m-columns of each shell, pinned at every SCF cycle.
   The pinning exists because the plain Aufbau ROHF of step 0 (the raw
   route, §4.1) can converge to an IMPROPER configuration outright
   (measured: Sc/sto-3g lands in 4p¹ instead of 3d¹); force_config
   detects the valence l-columns from that first run and re-runs ROHF
   with the target pattern.
2. M-average the converged density (`m_average_density`): every
   separately-outlined (2l+1) AO block → (tr/n)·I; same-l different-radial
   pairs keep their same-m coupling; cross-l blocks are zeroed (a group
   average cannot contain cross-l content).
3. Build F[dm_sph] and re-diagonalize (`spherical_reference`). On a
   spherical atom [F, l²] = 0, so by Schur every eigenvector can be
   chosen l-pure; complete shells are degenerate manifolds.
4. Pick the active span = complete l-shells (7s, 6d, 5f); freeze; run
   CASCI.
5. Certificates: span closure (‖L[out, act]‖ = 3.1e-12 on U) and core
   (purity 2.4e-15, disjoint 4.0e-14, occ = 2).
6. **L² over the root set** → eigvalsh (`moment_matrix`): the term
   ladder. This step also *resolves* degenerate-block mixing: the CI
   solver returns arbitrary bases inside exactly-degenerate levels, so
   per-root ⟨L²⟩ values scatter (the recorded frozen-7s example:
   one energy cluster containing {L=0 ×1, ⁵D ×6, ⁵I ×6} — per-root
   ⟨L²⟩ anywhere between 0 and 42 at the same energy); the eigenvalues
   of the k×k matrix ⟨vᵢ|W|vⱼ⟩ over the whole cluster are the exact
   ladder. The energies are untouched.
7. Cross-checks: dm-vs-operator, trace identities, commutator (§3).
8. Output: roots + clusters → `u14e_roots60.npz/.meta.json`,
   `u14e_cas16.*`.

```python
# reference.py — the group average, exact in shell blocks (abridged)
for a0, a1, la in ranges:
    for b0, b1, lb in ranges:
        if la != lb: continue                     # cross-l: zero
        n = a1 - a0
        dm_sph[...] = (np.trace(dm[...]) / n) * np.eye(n)   # m-average
```

```python
# moments.py — the sector application (reshape trick), EXACT, O(n_str^3)
V = psi.reshape(na_str, nb_str)
for Ea, Eb in mats:                                # one pair per component k
    out += Ea @ (Ea @ V)                           # alpha-alpha
    out += (V @ Eb.T) @ Eb.T                       # beta-beta
    out += 2.0 * (Ea @ V @ Eb.T)                   # the 2 l(1).l(2) pair
```

**The worked m-average example** (carbon p-block, `mini_gates.meta.json`):
dm_p = diag(1, 1, 0) → m-average → (2/3)·I₃, anisotropy 0.5774 = 1/√3.
The M_L-selected determinant IS anisotropic; its group average is the
spherical shell density. This three-line computation is the whole
mechanism of step 2.

**The incomplete-shell gates** (same file): both {px,py} and {px,pz}
spans fail closure (block norm 1.0) and project ⟨L²⟩ to [0] — not the
physical 2. A complete m-set per radial is what closure needs; partial
multiplet picks silently return in-span projections.

**The three mathematical statements** (verified in
`tests/test_mini_gates.py` and throughout):

1. **l is purely angular — the radial-overlap theorem.** ∇ = r̂ ∂_r +
   (1/r) ∇_S, and the r̂∂_r × r̂ term vanishes identically, so
   l = r̂ × ∇_S: no radial derivatives. Every l matrix element therefore
   factorizes as (angular factor on m) × ∫ R′(r) R(r) r² dr — the plain
   radial overlap. S-orthogonal radial functions ⇒ zero cross-radial
   l-elements ⇒ a span of one-radial complete-m-multiplets is *exactly*
   L-invariant. Closure can only break via (a) incomplete m-sets per
   radial (the gates above) or (b) l-impure / multi-radial columns.
   Check status: the theorem's consequence is what every span
   certificate measures (the invar 1e-16–1e-12 of the reference route);
   there is no standalone gate for the factorization itself — the gates
   B/M/Z verify its end results.
2. **The rank-1 separability criterion — and the Lz obstruction.** An
   l-pure column's coefficients over the shell block form C[r, m].
   Separability (one radial × one harmonic) is rank C = 1; generic
   converged orbitals have rank ≥ 2 — the "averaged polynomial" (a
   fixed radial combination weighted by m-dependent factors). No
   Hermitian REAL operator selects the separable form: l² is ∝ identity
   within the block (radially blind), and the operator that could
   resolve m — Lz — has only COMPLEX eigenstates (∝ Y_lm), while every
   real column has ⟨Lz⟩ = 0 identically (§0.5). This is the root cause
   of the m-separability obstruction: the pyscf SCF/CASCI/CASSCF
   pipeline is real-arithmetic by construction, so it can never produce
   m-eigenstates; the measured consequences are the ensemble-SCF paradox
   (l-pure columns 1.3e-5, span still open — §4.3) and the failed
   cluster-Lz fix (`reference._m_separate`, verdict FAIL in
   `u14e_ensscf_msep.meta.json`: invar 1.17 → 0.73, purity degraded,
   ⟨Lz⟩ = 0 on all 13 columns). Working with complex orbitals would be
   the direct cure — but pyscf's CI solver REJECTS complex Hamiltonians
   (`direct_spin1`: NotImplementedError, verified), so the complex
   route would need a complex-capable CI solver; the practical
   alternatives: (a) completeness-checked multiplet picks — cluster
   columns by Fock degeneracy and verify each cluster's L² block =
   l(l+1)·I before accepting (the certificates already compute exactly
   this); (b) the C[r,m] PROFILES are still readable in the real AO
   shell basis (functions there are contraction × real-Y_lm) — only the
   Lz-diagonalization needs complex m-eigenstates. Status: open (§8).
3. **Exchange is parallel-spin only**, and it favors the m-separated
   (Hund) pattern. The m-anisotropic states harvest it; the ensemble pays
   750 mHa for giving it up (measured, §6) — precisely BECAUSE the
   configuration is forced (the m-average destroys the exchange-
   favorable orbital assignment of the Hund determinant). Radial mixing
   neither creates parallel-spin pairs nor improves exchange at fixed
   occupations — hence no ROHF fuel for radial drift; the CASSCF fuel is
   static correlation instead (§4.4).

### 4.3 ensemble-SCF + CASCI (`tests/test_u14e_ensscf.py`, `u14e_ensscf.meta.json`)

**Why the name.** The pinned m-averaged occupations (7s¹, 6d⁵ spread as
1/5 over five columns, 5f⁷ as 3/7 over seven) are a *fractional-occupation
ensemble density*; the route makes SCF self-consistent with that ensemble
instead of a single determinant.

Steps and the three implementation traps:
1. Pin the fractional occ vector on the stored spherical basis.
2. **Trap 1 — index-pinning scrambles.** Each SCF cycle re-diagonalizes F
   and re-sorts columns by orbital energy; identities swap between
   cycles. Concretely (illustrative index list):

   | cycle | col 9 | col 13 |
   |---|---|---|
   | k | ε = −0.31 → 5f₁ | ε = −0.30 → 6d₁ |
   | k+1 | ε = −0.30 → 6d₁ | ε = −0.31 → 5f₁ |

   The pinned occ "column 9 ← 1/5" now lands on the *other* physical
   orbital; occupations leak across shells and the density re-anisotropizes.
3. **Trap 2 — MOM integerizes.** pyscf's maximum-overlap method
   (`scf_addons.mom_occ_`) assigns whole electrons to the columns that
   overlap most with the previous occupied set — the right *steering*
   tool (it drives SCF into excited configurations; used for the UKS-MOM
   start of §4.9) but unable to express an ensemble.
4. **Trap 3 — pyscf ROHF corrupts fractional occ.** `make_rdm1` maps the
   1-d occ by `occa = (occ > 0)`: a 0.43-occupied column becomes FULLY
   α-occupied and the electron count corrupts. Replacement:

```python
# test_u14e_ensscf.py — the true fractional density
def frac_rdm1(mo_coeff=None, mo_occ=None, **k):
    moc = mo0 if mo_coeff is None else np.asarray(mo_coeff)
    o = occ if mo_occ is None else np.asarray(mo_occ)
    return (moc * o) @ moc.T            # occ split explicitly, no threshold
mf.make_rdm1 = frac_rdm1
```

5. The occupation *lock* that survives the cycles: per cycle, match each
   reference shell to its best-overlap current columns (shell-based MOM)
   and re-assign the fractional pattern to THEM.
6. CASCI on the result; certificates; CASCI-on-ensemble-basis control.

Outcome (`u14e_ensscf.meta.json`): E(ensemble-SCF) = −49.489717 — exactly
between the one-shot averaged density (−49.401574, §6) and the fresh
Aufbau SCF (−50.151644); orbitals l-pure to 1.3e-5; **but** the span is
open (dev 1.23) because the energetically optimal columns are
*radial-mixed within l* (rank ≥ 2 — statement 2 of §4.2). Radial mixing
is detected indirectly by the span certificate; the DIRECT test — the
rank of C[r,m] per column — is not implemented, and its clean definition
needs the m-resolved basis (the profiles are readable in the real AO
shell basis, §4.2 statement 2; the Lz-eigenbasis needs complex columns).
CASCI inside it: E = −49.655936, ⟨L²⟩(root 0) = 17.3 — a mixing value.
Purifying afterwards makes closure worse (dev 2.23). Verdict:
self-consistency and radial purity fight each other; not a term-valid
reference. The attempted m-separation rescue is described with §4.2,
statement 2.

---

### 4.4 ss-CASSCF (`tests/test_casscf.py`, `test_u14e_casscf.py`,
`test_u14e_energy_ncheck.py`, `test_u14e_casscf_cas16.py`)

**Why a single root cannot be a term.** The lowest CASCI root of a
degenerate multiplet — returned by Davidson, the iterative CI
diagonalization of H_CASCI inside the sector — is an *arbitrary vector
inside the degenerate subspace*: an unconstrained M-mixture. Its density
is anisotropic; the CASSCF gradient inherits the anisotropy; the
optimizer follows it off the symmetric manifold. The orbitals relax into
l-open configurations and the roots' ⟨L²⟩ become mixing values (§0.4).

**"Why not just rotate the roots to the needed M values?"** Three
independent blockers: (i) in the real basis M-eigenstates do not exist
for the columns (⟨Lz⟩ ≡ 0, §0.5) — the rotation needs complex columns,
which pyscf's CI machinery rejects (§4.2, statement 2); (ii) even a
perfect M_L eigenstate would not help — a single |L, M⟩ density is
itself anisotropic (|Y_lm|² is not spherical; only the COMPLETE
multiplet average is), so the next CASSCF cycle re-ani­sotropizes;
(iii) decisively: once the optimizer has moved the orbitals, the SPAN is
gone and no CI-vector rotation can restore it — rotations act on the CI,
the damage is in the basis. The resolution is therefore span-level:
supply the closed span (§4.2) or constrain the optimizer (§4.8b).

**Carbon first** (`carbon_casscf_baseline.meta.json`): ss-CASSCF
E = −37.682418, ⟨L²⟩ = 1.9176 (dm-route 2.0000, dev 0.082; span dev
0.202) — a 4% excited-L admixture in the ground root: the mechanism,
miniature scale.

**U-14e** (`u14e_casscf_comparison`, `u14e_energy_ncheck`,
`u14e_casscf_cas16` metas): every ss run returns mixing values and an
open span. At least **four distinct stationary points**, all reproducible
(the same values recur across independent scripts: −50.439933 appears in
both `u14e_casscf_comparison` and `starts_casscf` B-ss to 3e-7;
convergence-insensitive: tightening the gradient 1e-7 → 1e-9 moves the
energy 2.9e-8 Ha):


| stationary point | E / Ha | E / eV | grad | ⟨L²⟩(root 0) | span dev | data |
|---|---|---|---|---|---|---|---|---|
| SP-1 | −50.439933 | −1372.5405 | 4.8e-5 | 27.07 | 1.75 | `starts_casscf` B-ss |
| SP-2 | −50.454807 | −1372.9452 | 3.3e-5 | 83.74 | 1.21 | `u14e_casscf_comparison` ss |
| SP-3 | −50.454894 | −1372.9476 | 2.9e-5 | — | — | `u14e_energy_ncheck` stationarity |
| SP-4 | −50.459443 | −1373.0714 | — | — | — | `u14e_energy_ncheck` known points |

(83.74 > 72 proves components beyond the ⁵L channel — L ≥ 9; §0.4.)
CAS(6,16) versions behave identically: ss E = −50.462617 (|grad| 3.6e-5,
⟨L²⟩ = 7.24, span dev 1.50, p-population 3.48 e), SA-13 E = −50.460731
(⟨L²⟩ = 17.09, dev 1.49). Freezing 7s² (CAS(4,12)) does not help:
E = −50.155587 (+8.78 mHa vs the (6,13) reference), span still open
(dev 2.16), ⟨L²⟩ = 48.97 — while the p-population drops to ~0
(`u14e_frozen7s`): **virtual-polarization theft is a symptom, not the
mechanism** (§4.5). The full-space NO audit (§5.5) refutes "missing
orbitals" for every one of these points.

**Verdict.** The ss minima are the true variational minima of the model;
the symmetric manifold is a saddle (their depth: 0.26–0.29 Ha, §6). Term
identity cannot be an emergent property of unconstrained optimization on
a flexible basis.

**What opens the span passing from sto-3g to def2 — the two channels.**
1. *Radial differentiation within an l-block* — the radial analogue of
   Hund breaking: with ≥ 2 radial functions per shell the optimizer can
   give different m-components different radial extents, reducing the
   direct Coulomb repulsion between the open-shell electrons at fixed
   exchange structure; the CI then re-optimizes correlation inside the
   modified subspace. sto-3g has one radial per l — the differentiation
   degree of freedom simply does not exist, hence the stable symmetric
   point there. This channel is what the frozen-7s² experiment isolates:
   drift persists (dev 2.16) with p-population ≈ 0 — a d/f-internal
   channel suffices.
2. *Polarization theft* — def2's polarization functions let the valence
   columns acquire foreign-l character (the measured p-theft, 0.7–6 e;
   purity_max up to 2.8 = strongly l-mixed columns). Physically
   intuitive; unambiguously measured.

Status of channel 1: **interpretation, consistent with all evidence**
(frozen-7s, NO audit §5.5, purity values), not yet a direct measurement
— the direct check (C[r,m] profiles of the drifted columns; the energy
decomposition Coulomb-vs-correlation) is feasible and listed as an open
item (§8).

### 4.5 SA-CASSCF, and the density-sphericity experiment
(`tests/test_sa_anisotropy.py`, `test_ti_sa_full.py`, `test_no_audit.py`)

**What SA changes.** State-averaging over a COMPLETE multiplet with equal
weights 1/(2L+1) makes the CI density exactly spherical *at a symmetric
stationary point* — removing the single-root anisotropy fuel of §4.4.
Whether the optimizer stays there is an empirical question, answered per
system in §2.2: carbon (SA-3, SA-9) and Si stay closed; Sc def2, Ti def2
and U open the span. Averaging size is not the cure: U SA-60 (span dev
1.38, E = −50.398083) and the **full 190-root sector** Ti def2 SA-190
(dim 325, `ti_sa_full.meta.json`: E = −847.035166, span dev 2.28,
ladder dev 2.1e-2, 6.0 e of p-population) still open it.

**The terms.** Span dev = the closure block norm of §3.2. Ladder dev =
max |λᵢ − round(λᵢ)| over the root ladder. p-theft (p-population) = the
Löwdin population accumulated in p-columns by the relaxed valence —
measured by `diag.lowdin_shell_pops` (`u14e_population.meta.json`: the
spherical reference has p-pop ~1e-24; the span-open solutions 1–6 e).

**The mechanism experiment** (`sa_anisotropy.meta.json`): track the
density anisotropy ‖γ − γ_sph‖/‖γ‖ per SCF/CASSCF cycle during
equal-weight SA:

| system | anisotropy over cycles | final span dev | ladder |
|---|---|---|---|
| carbon SA-3 | 7.55e-2, **constant** (all 8 cycles) | 1.8e-6 | exact |
| U SA-13 | 3.4e-4 → **2.0e-1** (cycle 27) → ~3e-2 | 1.5 | mixing |
| Ti def2 SA-60 | 1e-3 → 5.7e-3 → 1.2e-3 (≈ spherical) | 1.5 | mixing |

Findings:
1. U shows seed-and-descend: the start density is spherical (3.4e-4);
   inexact Davidson roots seed anisotropy; the optimizer takes the
   downhill anisotropic directions (peak 0.20) and settles at the open
   minimum (~3e-2).
2. **Density sphericity is neither necessary nor sufficient** for a
   closed span — measured in both directions: Ti def2 SA keeps the
   density spherical to 1.2e-3 yet the span opens; carbon SA keeps
   7.55e-2 anisotropy yet is term-exact. The operative condition is the
   span closure of §3.2 — radial mixing within l (statement 2 of §4.2)
   violates it while remaining invisible to any density-based test.
   Why SA sometimes holds: minimal/less flexible bases starve the
   anisotropic descent (nothing downhill within reach — §5.3); on
   flexible bases the span-open minima are deep enough to be reached from
   the seeded start. And the natural repair attempt — build the natural
   orbitals of the m-averaged density and take their eigenvectors — is
   EXACTLY the post-hoc symmetrization of §4.7: tried, measured, and it
   fails for the radial-reordering reason documented there.

### 4.6 CASSCF → CASCI in the same orbitals (the relation)

*Why this remark exists: it kills the tempting one-line fix "just re-run
CASCI on the CASSCF orbitals" with a measured no.*

Diagonalizing the model Hamiltonian in a span-open basis does NOT restore
symmetry: the CI coefficients are tied to the basis, and a CI "ground
state" in an open span is the model's minimum, not an eigenstate of L².
Measured everywhere: CASCI on ss-CASSCF orbitals returns the same mixing
values as the CASSCF root (e.g. 17.3 in the ensemble basis, §4.3;
48.97 in the frozen-7s ss basis, §4.4). The inverse relation is the one
that works: CASCI on the *spherical* reference (§4.2), where the span is
closed by construction. Code: `battery_lib.casci_run` (the uniform
wrapper used by every CASCI row in this report).

---

### 4.7 post-hoc symmetrization: CASSCF → purify → CASCI
(`tests/test_u14e_casscf_symm.py`, `u14e_casscf_symm.meta.json`)

Steps:
1. Take the converged ss (or SA) CASSCF orbitals.
2. Project them back to l-pure form: m-average the drifted density,
   re-diagonalize (the §4.2 machinery applied to the *drifted* density).
   Purity is restored to 1.8e-14.
3. Re-run CASCI in the purified columns.
4. Optionally restart CASSCF from them (outer loop).

What happens: purification restores l-*purity* but NOT the radial
ordering — **the pure-l columns no longer correspond to 7s/6d/5f**: the
m-averaged Fock built from the drifted density reorders the radial
spectrum (reference overlaps drop to 0.36–0.7), and the CAS picks the
WRONG radial set. The purified ss space: span closed (4.6e-13), ladder
[6 ×5, 12 ×7, 72 ×7] — integers! — at E(root 0) = −25.297857, i.e.
**+25 162 mHa above the CASSCF energy** (the wrong-shell space; similar
cases span +12 to +27 Ha). The outer loop (restart from the purified
orbitals) simply re-breaks: E descends further (−50.4548) and the span
reopens — 2 outer rounds, same end. **Integers ≠ correct**: the ladder
is a property of the space, and the space is wrong. The Sc def2 SA row
of §2.2 (span dev 3.0, routes disagree by 12) is the adjacent lesson arriving
uninvited: on an open span even the evaluation routes stop agreeing.

### 4.8 Constrained CASSCF — tested variant and designed protocol

**(a) Tested: the outer-loop constraint** (= steps of §4.7 with a CASSCF
restart). Result: re-breaks (§4.7 step 4). A constraint applied BETWEEN
full optimizations does not survive the optimization.

**(b) Designed, not yet executed: the in-loop gradient projection.** The
symmetric stationary point is the minimum *restricted to the symmetric
manifold*; the tool is a CASSCF whose orbital gradient is projected onto
that manifold every cycle. The pyscf format makes the constraint
explicit: `mc.get_grad` returns the packed rotation gradients
[act×(core+virt) | core×virt], so "freeze the active span" = zero the
first ncas·(ncore + nvir) elements and leave the core–virtual block.
Caveat, documented for the implementer: the default second-order (AH)
optimizer propagates the constraint through the Hessian off-diagonal
blocks — the projection must be applied to the STEP (or a first-order
projected optimizer written), not merely to the gradient. Expected
deliverable: E_constrained ∈ [−50.1644, −50.4548] — the CASCI energy
plus core/virtual relaxation; a new physical quantity (the cost of
symmetry at fixed span), which would close the "saddle depth" question
of §6 from above.

### 4.9 The start matrix: CASSCF/CASCI from every initial guess
(`tests/test_starts_matrix.py`, `test_starts_casscf.py`;
`starts_matrix_casci`, `starts_casscf` metas)

**The starts.** UKS starts are PBE Kohn–Sham solutions stored as
checkpoint files (`14e/uks_pbe*.chk`), converted to spin-summed natural
orbitals for the common-orbital requirement:

| start | definition |
|---|---|
| A sph-ROHF | the reference of §4.2 (m-averaged pinned ROHF + spherical_reference) |
| B raw-ROHF | plain Aufbau ROHF |
| C UKS-5f⁴ | UKS converged into the 5f⁴ configuration — a *wrong-configuration control* |
| D UKS-CHAR | character-locked: occupations pinned per shell during the UKS SCF (the strongest, most configuration-pure start) |
| E UKS-IDX | index-pinned occupations through the UKS cycles |
| F UKS-MOM | maximum-overlap occupation assignment through the UKS cycles (the §4.3 steering tool) |

**The treatments.** *direct* = CASCI/CASSCF in the start's natural
orbitals matched to the reference columns by overlap. *symm* = the
placement-(ii) start treatment: symmetrize the start orbitals FIRST
(m-average density + re-diagonalize, §4.2 step 2–3), THEN match and run.

```python
# test_starts_matrix.py — the column matching (greedy, overlap-based)
Ov = np.abs(mo.conj().T @ S @ mo_ref)
for r in refs:                       # each reference column
    o, j = max((Ov[j, r], j) for j in unused)
    minov = min(minov, o)            # if this drops low, the match is bogus
```

**CASCI results (13 roots each):**

| start (treatment) | E / Ha | ΔE vs A / mHa | E / eV | ΔE / eV | span dev | ⟨L²⟩₀ |
|---|---|---|---|---|---|---|
| A sph (direct) | −50.164365 | 0 | −1365.0419 | 0.0000 | 3.1e-12 | 42.0 |
| A sph (symm) | −38.162175 | +12 002 | −1038.4457 | 326.5911 | 1.8e-13 | 12.0 |
| B raw (direct) | −50.389564 | −225.2 | −1371.1699 | −6.1280 | 1.09 | 33.3 |
| **B raw (symm)** | −50.164365 | **0.00** | −1365.0419 | 0.0000 | **7.4e-12** | **42.0** |
| C UKS-5f⁴ (direct) | −50.295264 | −130.9 | −1368.6039 | −3.5620 | 1.08 | 6.80 |
| D UKS-CHAR (direct) | −50.357441 | −193.1 | −1370.2958 | −5.2545 | 0.83 | 17.7 |
| E UKS-IDX (direct) | −50.296933 | −132.6 | −1368.6493 | −3.6082 | 0.83 | 4.44 |
| F UKS-MOM (direct) | −50.300789 | −136.4 | −1368.7542 | −3.7116 | 0.93 | 4.27 |
| C–F (symm) | −19.06…−19.25 | **+30 900…+31 100** | −518.6…−523.8 | +840.8…+846.3 | 2.9–3.0 | 21.5–22.0 |

**Findings in plain language.**
1. *The pipeline is self-consistent*: symmetrizing the raw-ROHF start
   (B-symm) reproduces the reference EXACTLY — same E to 1.5e-6 Ha, same
   span closure, same ⟨L²⟩ = 42. The route of §4.2 does not depend on
   having been followed historically; its output is unique.
2. *B-direct is lower but term-blind* (−225 mHa, mixing value 33.3): the
   usual open-span story of §4.1.
3. *Every UKS start is term-invalid, in both treatments, for two
   different reasons.* Direct: the DFT potential under the m-selected
   density is anisotropic, so the UKS columns are l-smeared (span dev
   0.8–1.1, mixing values 4.3–17.7). Symm: the symmetrizer needs to find
   each reference column's partner in the start; the UKS natural
   orbitals are so far from the ROHF valence that the best overlaps are
   only 0.13–0.32 — the matcher pairs WRONG columns and builds a space
   ≈ +31 Ha too high. (A-symm fails differently: A is already symmetric,
   so the "symm" pass just re-picks degenerate columns — +12 Ha; note
   its ladder is still exact-integer [12.0]: the in-span projection of
   an incomplete pick, §2.1 gate 5.)

**CASSCF from every start** (`starts_casscf.meta.json`, 8 runs:
ss + SA-13 from B, C, D, F): **all 8 open the span** (dev 1.49–2.48),
⟨L²⟩₀ = 17.5–50.1 (mixing values), p-theft ≈ 3 e universal; 6 of 8 hit
the 100-cycle limit unconverged. **The drift is intrinsic to the
optimizer on this basis and independent of the start** — including the
strongest, character-locked D: ss E = −50.454833 (span dev 1.88,
⟨L²⟩ = 50.08). Combined with §4.4–4.5: no initialization strategy
repairs what the unconstrained objective function wants to break.

---

## 5. Conceptual questions (every assumption flagged)

Status labels: **measured** (a number in a meta file), **theorem**
(mathematically exact), **assumed** (working hypothesis, supported but
not proven).

### 5.1 Why is the CASSCF minimum not a term? **[theorem + measured]**
A CASSCF stationary point minimizes the *model* (finite-CAS CI × free
orbitals). Anisotropic orbital rotations buy correlation the spherical
orbitals cannot; the measured depth is 0.26–0.29 Ha (U-14e). Term states
are eigenstates of the exact Hamiltonian; term character must be imposed
(constraints, §4.8) or supplied (the frozen spherical reference, §4.2) —
it is not an emergent property of unconstrained finite-space minima.

### 5.2 Which stationary point is "the" answer? **[definitional]**
The lowest one *on the symmetric manifold* — defined only by constrained
optimization (§4.8b, not yet executed). The global minimum is span-open
and is precisely what must be rejected for spectroscopy.

### 5.3 Why do minimal bases behave? **[measured]**
One radial per l, no polarization functions, little correlation to gain:
the anisotropic descent has no room. Ti sto-3g ss stays at dev 3e-14
with the clean d² configuration; Ti def2 opens (0.66 e of p-theft).
"Behaves" refers to symmetry, not to chemistry quality.

### 5.3b The sto-3g → def2 mechanism **[partially confirmed]**
Channel decomposition with status flags: polarization theft [measured];
radial differentiation within a shell [interpretation — the Hund
analogue: m-dependent radial extents reduce the direct Coulomb
repulsion; consistent with frozen-7s (drift with p ≈ 0) and with the
one-radial stability of sto-3g; NOT yet directly measured — the
C[r,m]-profile check is open, §8]. See §4.4 for the full paragraph.

### 5.4 Is the m-averaging too rude? **[measured, partially confirmed]**
The bookkeeping is exact (per-contraction shell blocks, same-m coupling,
cross-l zeroing — verified by gates B/M/Z and the worked example). The
750 mHa cost of averaging the U density is (i) real ensemble-formation
physics and (ii) partly the one-shot protocol: the self-consistent
ensemble recovers ~0.09 Ha of it (−49.4016 → −49.4897) but cannot close
the span (§4.3). What fraction is intrinsic vs protocol is *partially*
resolved. [Scope: light atoms and small open shells do not show this
cost at any comparable scale — the effect grows with the number of
near-degenerate valence shells.]

### 5.5 Are important orbitals missing from the CAS? **[measured — no]**
`tests/test_no_audit.py` (`no_audit.meta.json`): full-space
natural-occupancy spectra (the OAO-metric operator eigenvalues, NOT the
congruence CᵀγC — pyscf MOs are only S-orthonormal):
```python
s, U = np.linalg.eigh(S)                       # Lowdin OAO basis
dm_o = (U @ diag(sqrt(s)) @ U.T) @ dm @ (...).T   # C^-1 dm C^-T
```
Every CAS is saturated: U sph-CASCI has 13 important NOs = 13 columns
(top dead column 5.6e-16, no core drain, min above-band 2.0000); the
span-open CASSCF solutions likewise. "Missing orbitals" is refuted as the
drift cause, consistent with the CAS(6,16) refutation.

### 5.6 Density sphericity vs span closure **[theorem + measured; one assumption flagged]**
The textbook argument (equal-weight multiplet SA → spherical density by
Unsöld → shell-degenerate orbitals → pure-L CI) is correct **at a
spherical stationary point** [theorem], which the sph-ROHF/CASCI route
supplies by construction. It does not describe optimizer output: the
saddle is real [measured, 0.26–0.29 Ha], the predicted sphericity is
real [measured, Ti def2: 1.2e-3], and it is nevertheless irrelevant —
the span condition is violated radially [measured]. **Flagged
assumption**: that span closure is the *complete* operative condition.
Supported by: the theorem (§3.2) + every measured case in §5.7 + the
commutator separation (§3.3). Not proven: that no other independent
failure mode exists (no systematic search for counterexamples was made).

### 5.7 L² on non-closed solutions — the consolidated table
(ordered carbon → atoms → U; status = span-closed ✓ / span-open ✗)

| solution | span | ⟨L²⟩(root 0) or ladder dev | reading |
|---|---|---|---|
| carbon SA-3 / SA-9 | ✓ (2e-6) | ladder exact 1e-11 | ³P ladder |
| carbon ss (E-raw) | ✗ 0.20 | ⟨L²⟩ = 1.9176 | (mixed): ³P ⊕ ~4% excited-L |
| Si sto ss | ✗ 2.6e-3 | dev 1.4e-5 | (mixed, near-closed): ³P ⊕ 1e-5 |
| Ti sto ss | ✓ 3.0e-14 | dev 2.3e-10 | ³F + ³P (the symmetric point IS the minimum) |
| Sc def2 ss / SA | ✗ 2.1 / 3.0 | dev 7.5e-2 / 1.1e-3 (routes disagree by 12) | (mixed); the dm-vs-operator gap = the leak (§0.5) |
| Ti def2 ss / SA-60 | ✗ 1.7 / 1.0 | dev 0.20 / 0.48 | mixing |
| Ti def2 SA-190 (full) | ✗ 2.28 | dev 2.1e-2 | averaging size closed |
| U raw+CASCI | ✗ 1.09 | ⟨L²⟩ = 33.3 | between ⁵G(20) and ⁵I(42) |
| U ss (sph start) | ✗ 1.2–2.2 | ⟨L²⟩ = 7.2–83.7 | e.g. 48.97 = ⁵I ⊕ ~50% ⁵K |
| U SA-13 / SA-60 | ✗ 1.5–1.7 / 1.4 | dev 0.35–0.50 | mixing |
| U ensemble basis | ✗ 1.23 | ⟨L²⟩ = 17.3 (CASCI in it) | (mixed); radial-mixed columns |
| U ss in frozen-7s CAS(4,12) | ✗ 2.16 | ⟨L²⟩ = 48.97, p-pop ≈ 0.01 | theft is a symptom |
| U symmetrized wrong-shell (from CASSCF, §4.7) | ✓ | integer [6,12,72], +25 Ha | integers ≠ correct (wrong shell) |
| U starts C–F direct / symm | ✗ | ⟨L²⟩ = 4.3–17.7 / +31 Ha | l-smeared / mis-matched |
| **sph-ROHF/CASCI, all 7 systems** | **✓ (1e-16–1e-12)** | **exact everywhere** | **the reference** |

Reading rules: non-integer ⇒ outside every l-closed span (mixing value,
informative); integer alone ⇒ nothing (three live counterexamples in the
table). The p-theft column of earlier drafts is omitted here: §4.4
showed it is a symptom, not a mechanism (frozen-7s drifted with
p-pop ≈ 0).

---

## 6. Energy tables (annotated)

All energies in Hartree (1 Ha = 219 476 cm⁻¹ = 27.2114 eV); term gaps in
mHa where marked. Every entry carries its route (§0.2) and data source;
the annotations say WHY each number sits where it does.

| quantity | value / Ha | E / eV | annotation (route, why, source) |
|---|---|---|---|
| E(fresh Aufbau SCF) | −50.151644 | −1364.6958 | raw-ROHF (§4.1), span-open by construction (m-selected determinant density); the SCF start of most routes (`u14e_energy_ncheck`) |
| E(m-averaged density, one shot) | −49.401574 | −1344.2853 | the §4.2 density evaluated WITHOUT SCF self-consistency; 750 mHa ABOVE the raw SCF — the measured cost of forcing the m-averaged (ensemble) configuration against the exchange-favoring Hund assignment (§4.2 statement 3, §5.4) |
| E(ensemble-SCF, locked) | −49.489717 | −1346.6838 | §4.3: self-consistency recovers 0.09 Ha of that cost but the columns go radial-mixed → span open → not term-valid (`u14e_ensscf`) |
| E_CASCI(6,13), sph reference | **−50.164365** | −1365.0419 | THE reference: the route works because the m-average supplies a density whose Fock commutes with l² (so pure-l columns exist) AND the complete-shell span satisfies closure (§3.2) — both theorem premises delivered by construction (`u14e_roots60`) |
| E_CASCI(6,13), raw basis | −50.387840 | −1371.1230 | §4.1: 223.47 mHa = 6.0809 eV BELOW the reference = 16.8 term gaps (⁵I→⁵L = 13.33 mHa = 0.3627 eV) — the purity premium: the open span is a richer variational space, term-blind (`u14e_energy_ncheck`) |
| E_CASCI(6,16), sph reference | −50.168752 | −1365.1613 | §2.3: +4.4 mHa from 7p correlation; duplicate terms appear (`u14e_cas16`) |
| freezing 7s² → CAS(4,12) | −50.155587 | −1364.8031 | +8.78 mHa vs the (6,13) reference; invariants complete (§3.1); used to refute the p-theft mechanism (`u14e_frozen7s`) |
| freezing 7s²+7p³ → CAS(4,15) | −50.155589 | −1364.8031 | +13.16 mHa vs (6,16) (`frozen7s_invariants`) |
| ss-CASSCF stationary points | −50.4399…−50.4594 | — | §4.4: 4 distinct points, all span-open — the true minima of the unconstrained model (`u14e_energy_ncheck` stationarity) |
| symmetry-breaking advantage | 0.26–0.29 Ha | — | the saddle depth = reference minus the span-open minima; the target interval for §4.8b's constrained optimization |
| atoms: ΔE_CASCI(raw→sph) | ≤ 3 mHa (Sc sto: 16.9, space-size caveat) | — | on light atoms the two bases are energetically interchangeable; on U they are not (`atom_table` meta) |

---

## 7. Code map, rerun commands, architecture

```
lmom/
  angular.py      get_L, l2_op                                  (23 ln)
  moments.py      THE engine: e1_matrix, moment_mats, apply_moment,
                  sector_matrix/_eigenvalues, moment_matrix/
                  expectation, trace_moment/_2, _expand        (261 ln)
  expectation.py  dm-contraction route, span/core certificates,
                  conventions PAIRING/SIGN                     (227 ln)
  reference.py    spherical_reference, m_average_density,
                  force_config, _m_separate (OFF)              (139 ln)
  diag.py         purity_devs, Lowdin shell populations         (51 ln)
  data/           every number of this report: *.npz + *.meta.json
                  (incl. carbon_reference.meta.json — the C ³P row)
tests/            one script per experiment + battery_lib.py
                  (n_expectation, n_checks_dm, no_audit, energy_of_density,
                  casci_run, mo_passport, sz_expectation, lz_check)
  test_carbon_verification.py  the gates B/M/Z + UDT
  test_u14e_moments.py         U term identification (stored data)
  test_mini_gates.py           the worked example + incomplete-shell gates
  test_casscf.py               the carbon SA/ss baseline
  test_atoms_table.py          Si/Sc/Ti x 2 bases, 4 routes
  test_u14e_roots.py           CASCI(6,13) 60-root study
  test_u14e_cas16.py           CASCI(6,16)
  test_u14e_verify.py          trace/integrality invariants
  test_u14e_energy_ncheck.py   energies + N-checks + stationarity
  test_u14e_raw_control.py     the §4.1 negative control
  test_u14e_casscf.py          the ss/SA-13/SA-60 comparison
  test_u14e_casscf_symm.py     §4.7 post-hoc symmetrization
  test_u14e_casscf_cas16.py    CASSCF in CAS(6,16) (mechanism)
  test_u14e_frozen7s.py        the frozen-7s² study
  test_frozen7s_invariants.py  frozen-7s invariant completion
  test_u14e_population.py      Lowdin populations (p-theft)
  test_u14e_ensscf.py          §4.3 ensemble-SCF
  test_u14e_ensscf_msep.py     the m-separation attempt (FAIL, kept; §4.2 stmt 2)
  test_commutator.py           §3.3 the probe
  test_no_audit.py             §5.5 the audit
  test_sa_anisotropy.py        §4.5 the mechanism experiment
  test_ti_sa_full.py           SA-190
  test_starts_matrix.py        §4.9 CASCI wave
  test_starts_casscf.py        §4.9 CASSCF wave
  make_big_table.py            regenerates RESULTS_BIGTABLE.md/.csv
```

Rerun (conda env `pyscf`): `python tests/test_carbon_verification.py`,
`test_u14e_moments.py` (uses stored U data), `test_casscf.py`,
`test_atoms_table.py`, then `tests/make_big_table.py`.

**Architecture assessment.** The engine/certificates/reference split is
clean and proved sufficient for the whole study; the code invariants
(§3.1) travel with the engine, which is why every refactor since has been
smoke-tested in minutes. Two honest warts: (i) the module name
`moments2` was historical — renamed `moments` in v2; (ii) the
certificates live inside `expectation.py` while their natural home is a
one-call `verify(mol, mf, mc)` — the recommended next refactor if the
project grows. `battery_lib.py` is the emerging shared test surface;
`u14e_rohf.meta.json` and `u14e_rohf_raw.meta.json` are corrupt
(truncated old-session saves; superseded by `u14e_rohf_sph` and
`u14e_rohf_raw_control`) — data hygiene, noted, not blocking.

**pyscf traps encountered** (all pinned in code comments): ROHF 1-d
fractional occ → `occa = (occ>0)`; `contract_1e` pack_tril
symmetrization; MOM integerizes; Davidson mixes degenerate blocks; MO
columns reorder at every eigendecomposition (hence `mo_passport`);
`mc.get_h1eff()` returns a tuple; no `make_ham` in 2.13; natural
occupancies = eigenvalues of C⁻¹γC⁻ᵀ, not CᵀγC.

---

## 8. Open items

1. **Constrained (in-loop projected) CASSCF** — designed (§4.8b: the
   get_grad format is known; project the STEP because of the AH leak).
   Delivers E_constrained and closes the saddle-depth question.
2. **nroots → 200 in CAS(6,16)** — complete the second ⁵L (5/17 window
   cut) and hunt ⁵F/⁵D duplicates. Note: CASCI(6,16) DOES deliver pure
   terms already — evidence: complete multiplets (2L+1 roots each),
   exact m² compositions, cluster devs ~1e-10, commutator 3.2e-13
   (`u14e_cas16`, `commutator` metas).
3. **NIST centers-of-gravity comparison** — LS terms vs J-resolved
   levels (the model has no SOC; average J-levels into term centers
   before ordering comparisons).
4. **U-32e** (libgrpp route) — the engine is ready; the sector dims are
   the only obstacle (Lanczos route exists).
5. **The m-resolved basis** (§4.2, statement 2) — a complex-capable CI
   route (pyscf's direct_spin1 rejects complex Hamiltonians — verified)
   or the completeness-checked multiplet picks / C[r,m] profile
   diagnostics in the real AO basis.
6. Regenerate the two corrupt metas (§7) if those runs are ever needed.
7. **C[r,m]-profile diagnostic** on the stored drifted CASSCF solutions
   (readable in the real AO shell basis, §4.2 statement 2) — upgrades the
   radial-differentiation mechanism of §5.3b from interpretation to
   measurement; optionally add the Coulomb-vs-correlation decomposition
   of the 0.26–0.29 Ha.

*Closed since the previous draft: the UKS-start question — fully
answered by the start matrix (§4.9): no UKS start yields a term-valid
reference in either treatment, and CASSCF breaks from all of them.*

---

*Report v2. All numbers from `lmom/data/*.meta.json`; scripts in
`tests/`; companion: `HANDOFF.md`, `RESULTS_BIGTABLE.md`.*
