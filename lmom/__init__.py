"""lmom — the angular-momentum (L^2, Lz^2) analysis toolkit.

Module map (canonical):
  angular     get_L (the AO one-body l-matrices), l2_op
  moments    THE term machinery: exact sector L^2/Lz^2 from one-string
              matrices (per-component E.E construction), eigenvalues,
              expectations, root-space ladders, trace invariants,
              _expand (CI embedding)
  expectation the dm-contraction route (L2Lz_expectation_dm), the span
              and core certificates, print_lm — the independent
              cross-check route
  reference   spherical_reference (the m-averaged symmetric reference)
              and force_config (index-pinned m-averaged occupations)
  diag        purity_devs, Lowdin shell populations (reference-quality
              diagnostics)

Reading protocol for a solution: certificates (span + core) -> L^2
ladder (integer?) -> dm-vs-operator agreement -> energies/occupancies.
See README.md for the full story.
"""