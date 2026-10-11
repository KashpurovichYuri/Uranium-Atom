"""THE EXACT MOMENT-OPERATOR MACHINERY (L^2, Lz^2) — the sector-basis
construction from one-string matrix elements.

GLOSSARY
--------
E_pq = a+_p a_q            one-body (spin-orbital) excitation operator
E^a_pq = a+_{p,a} a_{q,a}  alpha-spin excitation (acts on alpha strings)
E_pq (spin-summed) = E^a_pq + E^b_pq — the lift of a one-electron
                           operator to Fock space; E sums over spins,
                           so 'sum over electrons' is automatic and the
                           electron number N never appears in formulas
kron(X, Y)                 Kronecker PRODUCT (tensor product of
                           matrices, numpy.kron): acts on (alpha-string,
                           beta-string) pairs as X[a',a]*Y[b',b]. NOT the
                           Kronecker delta (that appears separately as
                           d_qr in the algebra below)
sector                     the Fock-space block with FIXED (na, nb)
                           (= fixed N and M_S) over the working columns;
                           exactly the space CASCI solves in: a CI vector
                           is an array [alpha-string addr, beta-string
                           addr] of dim C(n,na)*C(n,nb). L^2 commutes
                           with N and Sz, so it is block-diagonal per
                           sector — we build the one block the CI lives in
sector matrix              the matrix of L^2 inside that sector

Frame (N-electron)
------------------
L = sum_{i=1}^{N} l(i)   (one-body; the 'two-electron' notation l(1)+l(2)
                          below is just the N=2 special case)
L^2 = sum_i l^2(i) + 2 sum_{i<j} l(i).l(j)
      one-body            two-body: ALL pairs (15 pairs for 6 electrons)
Using the spin-free identity  E_pq E_rs = d_qr E_ps - a+_p a+_r a_s a_q
one finds the EXACT second-quantized form (the one-body pieces cancel
against the contraction inside E.E):
    L^2 = sum_k sum_{pqrs} l_k[p,r] l_k[q,s] (E^a+E^b)_pq (E^a+E^b)_rs
        = sum_k [ E^k_a E^k_a + E^k_b E^k_b + 2 E^k_a E^k_b ]
   - the SAME component index k in both factors (the scalar/dot product
     l(1).l(2)); the orbital index pairs (p,r) and (q,s) are INDEPENDENT.
   - NO cross-component terms (lx ly on the same electron does not exist
     in L^2; summing l_k into A = lx+ly+lz and squaring pollutes).
The frozen core contributes exactly 0 (closed l-shells are L=0 states)
with vanishing cross terms — verified by the core certificate.
With the one-string matrices E1_k[a][i,j] = <j| sum_pq l_k[p,q] a+_p a_q
|i> built per component k in the alpha-string space (na electrons) and
the beta-string space (nb), the sector matrix is EXACTLY
    W = sum_k [ kron(E1_ka@E1_ka, I) + kron(I, E1_kb@E1_kb)
                + 2 kron(E1_ka, E1_kb) ]
      = sum_k (kron(E1_ka, I) + kron(I, E1_kb))^2     (>= 0, PSD!)
PSD: W is a literal square of a Hermitian matrix, so all eigenvalues are
>= 0 automatically — as L(L+1) >= 0 requires. (The old broken route
produced eigenvalue -2, which was the diagnostic red flag.)
and the application is the reshape trick (V = psi.reshape(na_str, nb_str)):
    W psi <-> sum_k [ E1_ka @ (E1_ka @ V) + (V @ E1_kb.T) @ E1_kb.T
                      + 2 E1_ka @ V @ E1_kb.T ]
O(n_str^3) per application, EXACT, ALL two-body pair terms included
(the previous one-body-only route missed the 2 l(1).l(2) pair part —
the root cause of the {4x9} anomaly; a matrix identity was used where
Fock-space algebra was required: 'apply twice' = contraction only, the
normal-ordered pair remainder a+_p a+_r a_s a_q is NOT generated).
Lz^2: the same with the single component l_z.

NOTE on conventions: this module uses the STANDARD second-quantized
convention sum o[p,q] a+_p a_q (created p, annihilated q, coefficient
o[p,q]). pyscf's fci.direct_spin1.contract_1e packs f1e with pack_tril
and silently symmetrizes, so it CANNOT apply the non-symmetric
complex-Hermitian l-matrices — do not mix it in.
"""
import numpy as np
from pyscf.fci import cistring

from lmom.angular import get_L


def _sign_rm(occ, q):
    """(-1)^(number of occupied orbitals below q in the string)."""
    return -1 if (sum(1 for x in occ if x < q) % 2) else 1


def e1_matrix(lmat, norb, nelec):
    """The one-string matrix of the one-body operator
        sum_pq lmat[p,q] a+_p a_q
    in the (nelec, 0)-string space, STANDARD convention. Verified
    against explicit fermionic algebra on random complex matrices."""
    strs = np.array(sorted(cistring.gen_strings4orblist(range(norb),
                                                        nelec)))
    addr = {int(s): k for k, s in enumerate(strs)}
    dim = len(strs)
    dt = np.result_type(lmat, np.float64)
    if dim == 0:
        return np.zeros((0, 0), dtype=dt)
    M = np.zeros((dim, dim), dtype=dt)
    for k, s in enumerate(strs):
        occ = [p for p in range(norb) if (int(s) >> p) & 1]
        for q in occ:
            sq = _sign_rm(occ, q)
            for p in range(norb):
                c = lmat[p, q]
                if abs(c) < 1e-14:
                    continue
                if p == q:
                    M[k, k] += sq * sq * c
                    continue
                if (int(s) >> p) & 1:
                    continue
                s2 = (int(s) ^ (1 << q)) | (1 << p)
                occ2 = [x for x in occ if x != q]
                sp = _sign_rm(occ2, p)
                M[addr[int(s2)], k] += sq * sp * c
    return M


def moment_mats(mol, mo_coeff, cols, na_elec, nb_elec, which='L2'):
    """The one-string matrix pairs [(Ea_k, Eb_k), ...], one per spherical
    COMPONENT k (all three for which='L2'; only z for which='Lz2').
    L^2 has NO cross-component terms: it is the per-component sum
        sum_k [E_k^a.E_k^a + E_k^b.E_k^b + 2 E_k^a.E_k^b],
    so the matrices must NOT be summed before squaring."""
    L = get_L(mol, mo_coeff)
    ix = np.ix_(cols, cols)
    norb = len(cols)
    if which == 'L2':
        lks = [L[k][ix] for k in range(3)]
    elif which == 'Lz2':
        lks = [L[2][ix]]
    else:
        raise ValueError(which)
    mats = []
    for lk in lks:
        Ea = e1_matrix(lk, norb, na_elec)
        if nb_elec > 0:
            Eb = e1_matrix(lk, norb, nb_elec)
        else:
            Eb = np.zeros((1, 1), dtype=Ea.dtype)   # the empty beta string
        mats.append((Ea, Eb))
    return mats


def apply_moment(mats, na_str, nb_str, psi):
    """W psi via the reshape trick (psi: the full sector vector);
    mats = [(Ea_k, Eb_k), ...] from moment_mats."""
    V = np.asarray(psi).reshape(na_str, nb_str)
    out = np.zeros_like(V)
    for Ea, Eb in mats:
        out = out + Ea @ (Ea @ V)
        if nb_str > 0:
            out = out + (V @ Eb.T) @ Eb.T
            out = out + 2.0 * (Ea @ V @ Eb.T)
    return out.ravel()


def sector_matrix(mats, na_str, nb_str):
    """The dense sector matrix (only for small sectors)."""
    from scipy.sparse import kron as sp_kron
    from scipy.sparse import identity as sp_id
    if nb_str == 0:
        return sum(Ea @ Ea for Ea, _ in mats)
    W = sp_kron(sum(Ea @ Ea for Ea, _ in mats), sp_id(nb_str)) \
        + sp_kron(sp_id(na_str), sum(Eb @ Eb for _, Eb in mats))
    for Ea, Eb in mats:
        W = W + 2.0 * sp_kron(Ea, Eb)
    return W.toarray()


def sector_eigenvalues(mats, na_str, nb_str, k=None):
    """The eigenvalues of L^2 (or Lz^2) in the (na, nb) sector.
    Small sectors: the dense diagonalization; large: Lanczos on the
    reshape-trick linear operator (k = the number wanted)."""
    d = na_str * max(nb_str, 1)
    if d <= 4000:
        W = sector_matrix(mats, na_str, nb_str)
        W = 0.5 * (W + W.conj().T)
        ev = np.linalg.eigvalsh(W)
        return ev if k is None else np.sort(ev)[:k]
    from scipy.sparse.linalg import LinearOperator, eigsh
    def mv(v):
        return apply_moment(mats, na_str, nb_str, v)
    op = LinearOperator((d, d), matvec=mv, dtype=float)
    kk = min(k or 6, d - 2)
    ev = eigsh(op, k=kk, which='SA', return_eigenvectors=False)
    return np.sort(ev)


def moment_matrix(mats, na_str, nb_str, vecs):
    """The k x k matrix M[i,j] = <v_i|W|v_j> for the k (possibly
    unnormalized, real/complex) sector vectors in vecs. This is the
    term-decomposition step: eigvals(M) = the term ladder L(L+1) when
    vecs span the CASCI root space of an l-closed span."""
    k = len(vecs)
    emb = [np.asarray(v).ravel() for v in vecs]
    M = np.zeros((k, k), dtype=complex)
    for j, v in enumerate(emb):
        w = apply_moment(mats, na_str, nb_str, v)
        for i in range(k):
            M[i, j] = np.vdot(emb[i], w)
    return 0.5 * (M + M.conj().T)


def moment_expectation(mats, na_str, nb_str, psi):
    """<psi|L^2|psi> for a (possibly unnormalized) sector vector."""
    w = apply_moment(mats, na_str, nb_str, psi)
    return float(np.real(np.vdot(np.asarray(psi).ravel(), w.ravel())))


def trace_moment(mats, na_str, nb_str):
    """Analytic Tr W (the sector L^2 matrix) from the one-string
    matrices only -- the exact per-component identity
        Tr W_k = nb_str * Tr(E1a^2) + na_str * Tr(E1b^2)
                 + 2 Tr(E1a) Tr(E1b)
    (for nb=0 sectors: Tr W = na_str * Tr(E1a^2)).
    Independent verification against sum(eigenvalues)."""
    t = 0.0
    for Ea, Eb in mats:
        if Eb.size:
            t += (nb_str * np.real(np.trace(Ea @ Ea))
                  + na_str * np.real(np.trace(Eb @ Eb))
                  + 2.0 * np.real(np.trace(Ea) * np.trace(Eb)))
        else:
            t += na_str * np.real(np.trace(Ea @ Ea))
    return float(t)


def trace_moment2(mats, na_str, nb_str):
    """Tr W^2 = the sum of squared eigenvalues, via the dense W
    (Frobenius square). trace_moment is fully analytic; this builds W
    and serves as its eigen-side counterpart."""
    W = sector_matrix(mats, na_str, nb_str)
    return float(np.real(np.sum(W.conj() * W)))


def _expand(nci, ncore, ci_act, nelecas, n_full=None):
    """Embed the active CI into the determinant space over n columns
    (n = ncore+nci, or n_full = the full MO count).

    A determinant = an integer bitmask (bit i = the orbital i occupied);
    the CI vector is indexed [addr_alpha, addr_beta] where each address
    enumerates the fixed-(nelec) occupation strings in pyscf's canonical
    order. The embedding: the core columns 0..ncore-1 doubly occupied in
    EVERY determinant, the active occupation strings shifted left by
    ncore, amplitudes copied verbatim (the core bits sit below all active
    bits, so relative phases transfer identically).

    Example (ncore=1, nci=2, nelecas=(1,0), n=3): the active det |orb 2>
    becomes the full det 0b101 = {|0 up>, |2 up>}.
    """
    n = n_full if n_full is not None else ncore + nci
    na, nb = ncore + nelecas[0], ncore + nelecas[1]
    core = (1 << ncore) - 1
    ci = np.zeros((cistring.num_strings(n, na),
                   cistring.num_strings(n, nb)))

    def sector_str(ne, i):
        return 0 if ne == 0 else int(cistring.addr2str(nci, ne, i))
    for ia in range(ci_act.shape[0]):
        sa = sector_str(nelecas[0], ia)
        for ib in range(ci_act.shape[1]):
            sb = sector_str(nelecas[1], ib)
            x = core | (sa << ncore)
            y = core | (sb << ncore)
            ci[cistring.str2addr(n, na, x),
               cistring.str2addr(n, nb, y)] = ci_act[ia, ib]
    return ci
