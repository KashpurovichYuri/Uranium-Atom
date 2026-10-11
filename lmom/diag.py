"""lmom/diag.py — reference-quality diagnostics shared by the batteries.

- purity_devs: per-MO l^2-diagonal deviation from the nearest l(l+1)
  (a pure-l column has l(l+1); deviation quantifies l-character loss).
- lowdin_shell_pops: per-MO gross populations by AO angular momentum
  (Lowdin: |(S^{1/2} m)_mu|^2 summed over the AOs of each shell l) —
  names WHICH primitive shells mix into a column, complementing the
  total-l purity metric.
Imports: numpy, pyscf gto constants via mol only; no lmom deps.
"""
import numpy as np

_SHELLS = 'spdfghik'


def purity_devs(L, cols, l_max=6):
    """devs[c] = min_l |d_c - l(l+1)| with d_c = sum_k (l_k l_k)[c,c]."""
    l2 = np.diag(sum(k @ k for k in L)).real
    targets = np.array([l * (l + 1) for l in range(l_max + 1)])
    return np.array([np.min(np.abs(targets - l2[c])) for c in cols])


def lowdin_shell_pops(mol, mo, cols):
    """Per-column Lowdin gross population by AO shell-l.
    Returns an array (ncols, l_max+1-ish) of populations."""
    S = mol.intor('int1e_ovlp')
    w, v = np.linalg.eigh(S)
    S12 = (v * np.sqrt(w)) @ v.T
    # l per AO from the basis shells (handles contracted/merged shells)
    l_of_ao = np.zeros(S.shape[0], dtype=int)
    from pyscf import gto
    loc = mol.ao_loc_nr()
    for i in range(mol.nbas):
        la = int(mol._bas[i, gto.ANG_OF])
        l_of_ao[loc[i]:loc[i + 1]] = la
    pops = np.zeros((len(cols), int(l_of_ao.max()) + 1))
    for j, c in enumerate(cols):
        m = S12 @ np.asarray(mo)[:, c]
        for l in range(pops.shape[1]):
            pops[j, l] = float((m[l_of_ao == l] ** 2).sum())
    return pops


def pops_summary(pops, label=''):
    """A compact per-column string 's:0.02 p:0.13 d:... '."""
    out = []
    for row in pops:
        s = ' '.join(f'{_SHELLS[l]}:{row[l]:.2f}'
                     for l in range(len(row)) if row[l] > 1e-3)
        out.append(s)
    return f'[{label}] ' + ' | '.join(out)
