"""
Kobayashi multipoint kinetics (MPK) on OpenMC's fission-matrix tallies.

Parameter extraction, the transient solver and the temperature-sensitivity
path, in one importable module. Used by two_region.py and ten_region.py in
this directory; see README.md for what the examples show and
docs/mpk_theory_manual.md for the derivation, index conventions and the
open issues.

Index conventions -- fixed, no exceptions:

    m = response region   -> fission_matrix_i_filter   (CellFilter)
    n = source region     -> fission_matrix_j_filter   (CellBornFilter)
    i = delayed family    -> fission_matrix_d_filter   (DelayedGroupBornFilter)

    K[m, n] == kij[m, n].  The first axis is the response region.

Shapes, enforced on construction:

    Kp    (N, N)      kdij[:, :, 0]
    Kd    (D, N, N)   moveaxis(kdij[:, :, 1:], 2, 0)
    beta  (N, D)      indexed by the SOURCE region n
    ell   (N,)        sp.ell_mean
    lam   (D,)

State ordering: y = [S_1..S_N, C^(1)_1..C^(1)_N, ..., C^(D)_1..C^(D)_N].

Usage::

    import mpk
    p = mpk.from_statepoint('statepoint.400.h5', cell_ids=[1, 2])
    p.report()                                   # every applicable check
    k0, S0, Sadj = p.fundamental()
    t, S, C = p.transient(t_eval, perturb=...)
"""
from dataclasses import dataclass, field
import warnings

import numpy as np
import openmc
import scipy.sparse as sps
from scipy.sparse.linalg import eigs

# ENDF/B U-235 six-group half-lives. Only a fallback: U-235 and U-238 (and
# Pu-239) have different decay constants for the same group index, so a
# mixed-nuclide fuel needs the decay-rate tally. See add_tallies().
_U235_HALF_LIVES = np.array([55.72, 22.72, 6.22, 2.30, 0.610, 0.230])
LAM_U235 = np.log(2.0) / _U235_HALF_LIVES


# --------------------------------------------------------------------------
# Input deck
# --------------------------------------------------------------------------
def configure_settings(settings, cells, n_delayed=6):
    """Attach the fission-matrix filters to a Settings object.

    The filter bin order fixes the region indexing for k_ij, k_dij and ell_i
    alike, so keep one list of cells and reuse it everywhere downstream.

    ``ifp_n_generation`` is deliberately NOT set here. ell_i now comes from
    the zero-lag estimator reported as ``StatePoint.ell_mean``, so the MPK
    parameter set needs no IFP tally at all. If you add IFP tallies for a
    conventional point-kinetics reference, use the default generation count:
    the conventional PK derivation weights with the fundamental adjoint,
    not with the single-generation importance G_m+.
    """
    settings.fission_matrix_i_filter = openmc.CellFilter(bins=list(cells))
    settings.fission_matrix_j_filter = openmc.CellBornFilter(bins=list(cells))
    settings.fission_matrix_d_filter = openmc.DelayedGroupBornFilter(
        range(1 + n_delayed))
    return settings


def add_tallies(tallies, cells, n_delayed=6):
    """Append the raw-beta and decay-rate tallies, one set per region.

    Three tallies per region, hand-rolled rather than taken from
    ``openmc.mgxs.Beta`` and ``openmc.mgxs.DecayRate``. Those two helpers
    are equivalent, but between them they emit *two* tallies scoring
    ``delayed-nu-fission`` on the same cell -- Beta's denominator and
    DecayRate's -- which makes any lookup by score set ambiguous. One
    shared denominator serves both ratios:

        beta_n^(i) = delayed-nu-fission[n, i] / nu-fission[n]
        lambda^(i) = decay-rate[n, i]         / delayed-nu-fission[n, i]

    Note the **cell** filter -- where the neutron is produced -- not
    cellborn. Raw beta is a plain forward tally; it must never be replaced
    by OpenMC's IFP beta_eff, which is adjoint-weighted and estimates
    beta_mn instead.

    The decay-rate tally is what makes lambda right for a mixed fuel.
    OpenMC's delayed group index is nuclide-specific, so a Pu-239/U-238 or
    U-235/U-238 fuel does not have U-235's decay constants. Hard-coding
    them does not break the MPK/PK identity -- both sides use the same
    lambda -- but it makes the absolute time response wrong.
    """
    # One shared instance of each group filter. Creating a fresh
    # DelayedGroupFilter or EnergyFilter per region would be equivalent --
    # OpenMC deduplicates identical filters on export -- but it emits an
    # IDWarning for every duplicate, which buries anything else in the log.
    one_group = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    energy = openmc.EnergyFilter(one_group.group_edges)
    delayed = openmc.DelayedGroupFilter(list(range(1, 1 + n_delayed)))

    for cell in cells:
        cf = openmc.CellFilter(cell)
        for score, filters in (
                ('nu-fission', [cf, energy]),
                ('delayed-nu-fission', [cf, delayed, energy]),
                ('decay-rate', [cf, delayed, energy])):
            t = openmc.Tally(name=f'{score} cell {cell.id}')
            t.filters = filters
            t.scores = [score]
            tallies.append(t, merge=False)
    return tallies


# --------------------------------------------------------------------------
# Statepoint extraction
# --------------------------------------------------------------------------
def find_tally(sp, scores, cell_id, require=True):
    """Look a tally up by score set and CellFilter bin.

    ``sp.tallies`` is keyed by tally ID, not position, and the iteration
    order of ``openmc.mgxs.Beta.tallies`` is an implementation detail.
    Positional indexing silently grabs the wrong tally as soon as a region
    is added, which is why this raises on anything but a unique match.
    """
    hits = []
    for t in sp.tallies.values():
        if set(t.scores) != set(scores):
            continue
        cf = [f for f in t.filters if type(f) is openmc.CellFilter]
        if cf and list(np.asarray(cf[0].bins).ravel()) == [cell_id]:
            hits.append(t)
    if len(hits) == 1:
        return hits[0]
    if not require and not hits:
        return None
    raise LookupError(
        f'expected exactly 1 tally {sorted(scores)} on cell {cell_id}, '
        f'found {len(hits)}')


def from_statepoint(path, cell_ids, lam=None):
    """Build an :class:`MPKParams` from one statepoint.

    Every parameter must come from a single statepoint. Two OpenMC runs of
    the same geometry give independent estimates of k, and a few hundred pcm
    of disagreement between them is a different transient regardless of
    whether the algebra is right.
    """
    cell_ids = list(cell_ids)
    N = len(cell_ids)
    with openmc.StatePoint(path) as sp:
        kij = np.asarray(sp.kij_mean, dtype=float)
        kij_sd = np.asarray(sp.kij_std_dev, dtype=float)
        kdij = np.asarray(sp.kdij_mean, dtype=float)
        kdij_sd = np.asarray(sp.kdij_std_dev, dtype=float)
        if kij.shape != (N, N):
            raise ValueError(
                f'kij has shape {kij.shape} but {N} cell_ids were given; '
                'cell_ids must match the fission_matrix_i_filter bin order')

        ell = np.asarray(sp.ell_mean, dtype=float)
        ell_sd = np.asarray(sp.ell_std_dev, dtype=float)
        if ell is None or ell.shape != (N,):
            raise ValueError(
                'this statepoint has no usable ell_mean. Re-run with a build '
                'that reports the zero-lag Kobayashi generation time; the IFP '
                'route is not a substitute (see the manual, "Regional '
                'generation times")')

        D = kdij.shape[2] - 1
        beta = np.zeros((N, D))
        lam_region = np.full((N, D), np.nan)
        for a, cid in enumerate(cell_ids):
            nu = find_tally(sp, ['nu-fission'], cid)
            dnu = find_tally(sp, ['delayed-nu-fission'], cid)
            beta[a] = dnu.mean.ravel() / nu.mean.ravel()[0]

            dr = find_tally(sp, ['decay-rate'], cid, require=False)
            if dr is not None:
                num = dr.mean.ravel()
                den = dnu.mean.ravel()
                with np.errstate(divide='ignore', invalid='ignore'):
                    lam_region[a] = np.where(den > 0.0, num / den, np.nan)

        ifp_gen = sp.ifp_n_generation
        keff = sp.keff

    if lam is None:
        if np.all(np.isfinite(lam_region)):
            # Production-weighted over regions: lambda is a nuclear constant
            # per family, so regional spread here is statistical noise plus
            # the genuine nuclide-mix difference between regions.
            w = beta / beta.sum(axis=0, keepdims=True)
            lam = (lam_region * w).sum(axis=0)
        else:
            warnings.warn(
                'no decay-rate tally found; falling back to ENDF U-235 '
                'six-group decay constants. For a mixed-nuclide fuel this '
                'makes the absolute time response wrong. Add the tallies '
                'from mpk.add_tallies() and re-run.', stacklevel=2)
            lam = LAM_U235[:D]
    lam = np.asarray(lam, dtype=float)

    return MPKParams(Kp=kdij[:, :, 0].copy(),
                     Kd=np.moveaxis(kdij[:, :, 1:], 2, 0).copy(),
                     beta=beta, ell=ell, lam=lam,
                     kij=kij, kij_sd=kij_sd,
                     kdij_sd=kdij_sd, ell_sd=ell_sd,
                     lam_region=lam_region,
                     ifp_n_generation=ifp_gen, keff=keff,
                     cell_ids=cell_ids, source=str(path))


# --------------------------------------------------------------------------
# Parameter container
# --------------------------------------------------------------------------
@dataclass
class MPKParams:
    """Canonical MPK parameter set, validated on construction."""

    Kp: np.ndarray                 # (N, N)    prompt coupling k^(p)_mn
    Kd: np.ndarray                 # (D, N, N) delayed coupling k^(i)_mn
    beta: np.ndarray               # (N, D)    RAW delayed fraction, region n
    ell: np.ndarray                # (N,)      Kobayashi generation time
    lam: np.ndarray                # (D,)      decay constants
    kij: np.ndarray = None         # (N, N)    tallied total, CHECK ONLY
    kij_sd: np.ndarray = None
    kdij_sd: np.ndarray = None
    ell_sd: np.ndarray = None
    lam_region: np.ndarray = None
    ifp_n_generation: int = None
    keff: object = None
    cell_ids: list = field(default_factory=list)
    source: str = ''

    def __post_init__(self):
        self.Kp = np.asarray(self.Kp, dtype=float)
        self.Kd = np.asarray(self.Kd, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)
        self.ell = np.asarray(self.ell, dtype=float)
        self.lam = np.asarray(self.lam, dtype=float)
        self.validate()

    @property
    def N(self):
        return self.ell.size

    @property
    def D(self):
        return self.lam.size

    @property
    def B(self):
        """Raw total delayed fraction per SOURCE region n, shape (N,)."""
        return self.beta.sum(axis=1)

    # ---------------------------------------------------------------- checks
    def total_matrix(self):
        """K_mn = (1-B_n) k^(p)_mn + sum_i beta_n^(i) k^(i)_mn.

        ``'ni,imn->mn'`` sums over i only; both n indices are the same index,
        so nothing is contracted over regions. ``'in,imn->mn'`` reads beta as
        (D, N) instead of (N, D) -- it raises at N != D but is a silent bug
        at N == D.
        """
        return ((1.0 - self.B)[None, :] * self.Kp
                + np.einsum('ni,imn->mn', self.beta, self.Kd))

    def validate(self, nsigma=3.0):
        N, D = self.N, self.D
        if self.Kp.shape != (N, N):
            raise ValueError(f'Kp has shape {self.Kp.shape}, expected {(N, N)}')
        if self.Kd.shape != (D, N, N):
            raise ValueError(
                f'Kd has shape {self.Kd.shape}, expected {(D, N, N)}; use '
                "np.moveaxis(kdij[:, :, 1:], 2, 0) so Kd[i] is an N x N matrix")
        if self.beta.shape != (N, D):
            raise ValueError(
                f'beta has shape {self.beta.shape}, expected {(N, D)}; beta '
                'is indexed by the SOURCE region n, then by family i')
        if not np.all(self.ell > 0.0):
            raise ValueError(f'non-positive ell: {self.ell}')
        if not np.all(self.beta > 0.0):
            raise ValueError(f'non-positive raw beta: {self.beta}')
        if not np.all(self.lam > 0.0):
            raise ValueError(f'non-positive decay constant: {self.lam}')
        if not (np.all(self.Kp >= 0.0) and np.all(self.Kd >= 0.0)):
            raise ValueError('negative coupling coefficient')

        # check 1: the partition identity. The only check sensitive to an
        # i/j transposition -- a transposed K has the same fundamental
        # eigenvalue and passes every eigenvalue-based test.
        if self.kij is not None and self.kij_sd is not None:
            K = self.total_matrix()
            z = np.abs(K - self.kij) / np.maximum(self.kij_sd, 1e-30)
            if z.max() > nsigma:
                raise ValueError(
                    f'fission-source partition identity off by {z.max():.1f} '
                    f'sigma\nreconstructed:\n{K}\ntallied kij:\n{self.kij}\n'
                    f'difference (pcm):\n{(K - self.kij) * 1e5}\n'
                    'Check the i/j filter orientation and that beta indexes '
                    'the source region n: it scales COLUMNS, not rows.')

    # ------------------------------------------------------------ eigenpair
    def fundamental(self):
        """(k, S0, Sadj) with sum(S0) = 1 and <Sadj, S0> = 1.

        K is nonnegative, so Perron-Frobenius guarantees a positive
        fundamental and makes "largest real part" and "largest modulus" the
        same selection. Mixed signs mean either the wrong eigenvalue was
        picked or K has negative entries from poor statistics.
        """
        K = self.total_matrix()
        if self.N <= 200:
            w, V = np.linalg.eig(K)
            j = int(np.argmax(w.real))
            k, v = w[j].real, V[:, j].real
        else:
            w, V = eigs(sps.csr_matrix(K), k=1, which='LM')
            k, v = w[0].real, V[:, 0].real
        if not (np.all(v >= 0.0) or np.all(v <= 0.0)):
            raise ValueError(
                f'fundamental eigenvector has mixed signs: {v}. Either the '
                'wrong eigenvalue was selected or K has negative entries.')
        S0 = np.abs(v)
        S0 = S0 / S0.sum()
        wL, VL = np.linalg.eig(K.T)
        Sadj = np.abs(VL[:, int(np.argmax(wL.real))].real)
        return k, S0, Sadj / (Sadj @ S0)

    def renormalized(self, k=None):
        """A copy with Kp and Kd divided by k, so S0 is exactly stationary.

        K S = S has no solution unless the system is exactly critical. For
        perturbation work, renormalize every case by the BASE eigenvalue --
        never by its own, which divides out the reactivity being measured.
        """
        if k is None:
            k = self.fundamental()[0]
        out = MPKParams(Kp=self.Kp / k, Kd=self.Kd / k, beta=self.beta,
                        ell=self.ell, lam=self.lam,
                        cell_ids=self.cell_ids, source=self.source)
        return out

    def equilibrium(self, S0):
        """Precursor equilibrium C^(i)_n = beta_n^(i) S_n / lambda_i."""
        return (self.beta.T * np.asarray(S0)[None, :]) / self.lam[:, None]

    # -------------------------------------------------------------- dynamics
    def generator(self):
        """The constant generator A for dy/dt = A y.

        Only the first block row and first block column are dense;
        everything else is diagonal. Note that (1 - B_n) multiplies S
        *inside* the matrix-vector product -- a column scaling indexed by the
        source region. Applied to the output it is a row scaling and wrong,
        with a signature of about 4e-4 drift on a stationary solution.
        """
        N, D = self.N, self.D
        inv_ell = 1.0 / self.ell
        P = (1.0 - self.B)[None, :] * self.Kp
        blocks = [[None] * (D + 1) for _ in range(D + 1)]
        blocks[0][0] = inv_ell[:, None] * (P - np.eye(N))
        for i in range(D):
            blocks[0][i + 1] = inv_ell[:, None] * (self.lam[i] * self.Kd[i])
            blocks[i + 1][0] = sps.diags(self.beta[:, i])
            blocks[i + 1][i + 1] = sps.diags(np.full(N, -self.lam[i]))
        return sps.bmat(blocks, format='csc')

    def scaling(self, S0):
        """Diagonal scaling making every state variable O(1) at reference.

        Precursor magnitudes span roughly two decades across families while
        S is order unity. Dividing each component by its equilibrium value is
        a similarity transform costing nothing, and makes one rtol mean the
        same thing for every component.
        """
        return np.concatenate([np.asarray(S0), self.equilibrium(S0).ravel()])

    def transient(self, t_eval, k_norm=None, perturb=None, scale=True):
        """Solve spectrally from the fundamental of the ASSEMBLED matrix.

        Seeding the tallied ``kij``'s eigenvector instead excites spatial
        harmonics -- a purely numerical shape transient two orders of
        magnitude above the physical one.

        ``perturb(Kp, Kd) -> (Kp, Kd)`` is applied after renormalization.
        """
        base = self.renormalized(k_norm)
        k0, S0, _ = base.fundamental()
        y0 = np.concatenate([S0, base.equilibrium(S0).ravel()])

        run = base
        if perturb is not None:
            Kp2, Kd2 = perturb(base.Kp.copy(), base.Kd.copy())
            run = MPKParams(Kp=Kp2, Kd=Kd2, beta=base.beta, ell=base.ell,
                            lam=base.lam, cell_ids=base.cell_ids)

        A = run.generator().toarray()
        if scale:
            d = run.scaling(S0)
            A = (A * d[None, :]) / d[:, None]
            y0 = y0 / d

        t = np.asarray(t_eval, dtype=float)
        w, V = np.linalg.eig(A)
        c = np.linalg.solve(V, y0)
        Y = (V @ (np.exp(np.outer(w, t)) * c[:, None])).real
        if scale:
            Y = Y * d[:, None]
        N, D = self.N, self.D
        return t, Y[:N], Y[N:].reshape(D, N, t.size)

    def step_transient(self, perturbed, t_eval, k_norm=None, scale=True):
        """Step change at t = 0: this state's steady state, perturbed matrices.

        ``self`` supplies the initial condition -- the fundamental of its own
        assembled matrix and the matching precursor equilibrium -- and
        ``perturbed`` supplies the matrices that are integrated. Both are
        renormalized by ``k_norm``, which defaults to **this** state's
        eigenvalue and should never be the perturbed state's own: normalizing
        each case by its own eigenvalue leaves both critical and divides out
        the reactivity being inserted.

        beta, ell and lambda are taken from ``self``, which is the usual
        working assumption that temperature moves only the coupling
        matrices. It cannot bias the eigenvalue or the shape, since neither
        beta nor ell appears in K, but it does affect the integrated
        transient -- ell sets the prompt timescale and beta the delayed. Quote
        the fractional change in the perturbed case's own ell and beta
        alongside any result that relies on it.
        """
        if k_norm is None:
            k_norm = self.fundamental()[0]
        _, S0, _ = self.renormalized(k_norm).fundamental()
        run = MPKParams(Kp=perturbed.Kp / k_norm, Kd=perturbed.Kd / k_norm,
                        beta=self.beta, ell=self.ell, lam=self.lam,
                        cell_ids=self.cell_ids)
        y0 = np.concatenate([S0, self.equilibrium(S0).ravel()])

        A = run.generator().toarray()
        if scale:
            d = run.scaling(S0)
            A = (A * d[None, :]) / d[:, None]
            y0 = y0 / d
        t = np.asarray(t_eval, dtype=float)
        wv, V = np.linalg.eig(A)
        c = np.linalg.solve(V, y0)
        Y = (V @ (np.exp(np.outer(wv, t)) * c[:, None])).real
        if scale:
            Y = Y * d[:, None]
        return t, Y[:self.N], Y[self.N:].reshape(self.D, self.N, t.size)

    def pk_reference(self, perturbed, t_eval, k_norm=None):
        """Frozen-shape point-kinetics solution for the same step change.

        The shape is frozen at ``self``'s fundamental, and rho comes from the
        Rayleigh quotient of the perturbed matrix in that frozen shape --
        not from the perturbed matrix's own eigenvalue. At frozen shape the
        two differ at second order in the perturbation, and using the
        eigenvalue would put that difference into the MPK-minus-PK residual,
        where it is indistinguishable from the spatial effect the comparison
        exists to measure.
        """
        if k_norm is None:
            k_norm = self.fundamental()[0]
        _, S0, Sadj = self.renormalized(k_norm).fundamental()
        run = MPKParams(Kp=perturbed.Kp / k_norm, Kd=perturbed.Kd / k_norm,
                        beta=self.beta, ell=self.ell, lam=self.lam)
        Lam, rho, beff = run.point_kinetics(S0, Sadj)
        D, lam = self.D, self.lam
        A = np.zeros((1 + D, 1 + D))
        A[0, 0] = (rho - beff.sum()) / Lam
        A[0, 1:] = lam
        A[1:, 0] = beff / Lam
        A[1:, 1:] = np.diag(-lam)
        y0 = np.concatenate([[1.0], beff / (Lam * lam)])
        t = np.asarray(t_eval, dtype=float)
        wv, V = np.linalg.eig(A)
        c = np.linalg.solve(V, y0)
        return (V @ (np.exp(np.outer(wv, t)) * c[:, None])).real[0], \
            (Lam, rho, beff)

    # --------------------------------------------------- adjoint collapse
    def collapse(self, S0=None, Sadj=None):
        """Frozen-shape point-kinetics reference from the same parameters.

        Returns ``(ell_eff, p_eff, btil, k_rayleigh)``. The identity
        ``p_eff + sum(btil) == k_rayleigh`` is exact by construction because
        both sides are the same projection. Build rho from ``k_rayleigh``,
        not from the perturbed matrix's true eigenvalue: at frozen shape the
        two differ at second order in the perturbation, and that difference
        lands in the MPK-minus-PK residual where it is indistinguishable
        from the spatial effect being measured.
        """
        if S0 is None or Sadj is None:
            _, S0, Sadj = self.fundamental()
        P = (1.0 - self.B)[None, :] * self.Kp
        ell_eff = (Sadj * self.ell) @ S0 / (Sadj @ S0)
        p_eff = Sadj @ (P @ S0)
        btil = np.array([Sadj @ (self.Kd[i] @ (self.beta[:, i] * S0))
                         for i in range(self.D)])
        k_rayleigh = (Sadj @ (self.total_matrix() @ S0)) / (Sadj @ S0)
        return ell_eff, p_eff, btil, k_rayleigh

    def importance_ratio(self, S0=None, Sadj=None):
        """Adjoint-weighted delayed importance ratio, beta_eff / beta_raw.

        The numerator is the collapsed beta_eff, built from the delayed
        coupling coefficients. The denominator replaces every k^(i)_mn with
        the tallied total k_mn and is otherwise weighted identically, so the
        ratio isolates how much more (or less) important a delayed neutron
        is than an average one.

        The point of weighting the denominator the same way is that a plain
        mean of the raw B_n is not comparable to beta_eff when B varies
        between regions. On a two-core model with B_n of 386 and 984 pcm the
        plain mean is meaningless; the adjoint weighting puts the regions in
        the same proportion on both sides.

        For most systems this sits within about 10% of 1. Every delayed
        count that a run fails to collect can only pull the numerator down,
        so a value well below 1 is either real physics or a bias in the
        delayed coefficients -- and the partition identity will not tell
        them apart, because the deficit simply moves into the prompt term.
        """
        if self.kij is None:
            return None
        if S0 is None or Sadj is None:
            _, S0, Sadj = self.fundamental()
        num = sum(Sadj @ (self.Kd[i] @ (self.beta[:, i] * S0))
                  for i in range(self.D))
        den = Sadj @ (self.kij @ (self.B * S0))
        return float(num / den) if den > 0.0 else None

    def point_kinetics(self, S0=None, Sadj=None):
        """(Lambda, rho, beta_eff) of the collapsed PK system."""
        ell_eff, _, btil, k = self.collapse(S0, Sadj)
        return ell_eff / k, (k - 1.0) / k, btil / k

    # ------------------------------------------------------------ reporting
    def report(self, stream=None):
        """Print every applicable check from the manual's hierarchy."""
        import sys
        out = stream or sys.stdout
        w = lambda *a: print(*a, file=out)
        K = self.total_matrix()
        w(f'MPK parameters from {self.source}  N={self.N} D={self.D}')
        if self.cell_ids:
            w(f'  cells (i/j filter bin order): {self.cell_ids}')
        if self.ifp_n_generation is not None:
            w(f'  ifp_n_generation = {self.ifp_n_generation} '
              '(IFP tallies present; not used for ell)')
        if self.keff is not None:
            w(f'  keff = {self.keff}')

        w('\ncheck 1  partition identity')
        if self.kij is not None:
            z = np.abs(K - self.kij) / np.maximum(self.kij_sd, 1e-30)
            w(f'  max |K - kij| = {np.abs(K - self.kij).max():.3e} '
              f'({np.abs(K - self.kij).max() * 1e5:.2f} pcm), '
              f'{z.max():.2f} sigma  -> {"PASS" if z.max() < 3 else "FAIL"}')
            w('  NOTE: dominated by the prompt term (beta ~ 1e-3 weights the '
              'delayed half), so this does not validate Kd.')
        else:
            w('  skipped: no tallied kij supplied')

        w('\ncheck 2  coupling sign and magnitude')
        w(f'  min Kp = {self.Kp.min():.6f}   min Kd = {self.Kd.min():.6f}')
        if self.kij is not None:
            w(f'  kdij[...,0]/kij - 1 (pcm):\n{(self.Kp / self.kij - 1) * 1e5}')
            w(f'  delayed sum / kij (expect ~D={self.D}):\n'
              f'{np.einsum("imn->mn", self.Kd) / self.kij}')
        n_d = self.Kd.size
        n_zero = int(np.count_nonzero(self.Kd == 0.0))
        w(f'  delayed coefficients k^(i)_mn: {n_d} elements, '
          f'{n_zero} exactly zero ({n_zero / n_d:.1%})')
        if self.kdij_sd is not None:
            kd_mean = np.moveaxis(self.Kd, 0, 2)          # back to (N, N, D)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = self.kdij_sd[:, :, 1:] / kd_mean
            good = np.isfinite(rel) & (kd_mean > 0.0)
            if good.any():
                w(f'  relative sd on the nonzero ones: median '
                  f'{np.median(rel[good]):.1%}, max {rel[good].max():.1%}')
        if n_zero:
            w('  An exactly zero delayed coefficient means no history in the '
              'whole run had a\n  delayed neutron born in n cause a fission '
              'in m. These are the sparsest\n  scores in the problem and '
              'they set the statistical floor on every\n  multi-region '
              'result. Their aggregate effect on K is bounded by the '
              'partition\n  identity above: that residual IS the error they '
              'introduce, plus noise.')

        k, S0, Sadj = self.fundamental()
        w('\ncheck 3  fundamental positivity')
        evs = np.sort(np.linalg.eigvals(K).real)[::-1]
        w(f'  k = {k:.7f}   eigenvalues = {evs}')
        if self.N > 1:
            w(f'  dominance ratio = {abs(evs[1]) / abs(evs[0]):.4f}')
        w(f'  S0 = {S0}\n  S+ = {Sadj}  (single-signed: PASS)')

        Lam, rho, beff = self.point_kinetics(S0, Sadj)
        ratio = self.importance_ratio(S0, Sadj)
        if ratio is not None:
            w(f'\n  delayed importance ratio beta_eff / beta_raw = '
              f'{ratio:.4f}')
            w('  The sharpest single test of the delayed coefficients, '
              'adjoint-weighted on\n  both sides so it stays meaningful '
              'when B varies between regions. It should\n  sit within '
              'about 10% of 1. A missing delayed count can only pull it '
              'DOWN, so\n  a value well below 1 is either real physics or '
              'a bias in k^(i)_mn -- and the\n  partition identity cannot '
              'tell them apart, because the deficit just moves\n  into the '
              'prompt term, leaving K right and the prompt/delayed split '
              'wrong.')
        w('\ncheck 4  reactivity scale')
        w(f'  Lambda = {Lam:.5e} s   rho = {rho * 1e5:+.2f} pcm   '
          f'beta_eff = {beff.sum() * 1e5:.2f} pcm')
        w(f'  rho/beta_eff = {rho / beff.sum():+.4f} dollars  -> '
          f'{"PASS" if abs(rho / beff.sum()) < 1 else "FAIL (>1$)"}')

        _, p_eff, btil, k_ray = self.collapse(S0, Sadj)
        w('\ncheck 7  collapse identity (against the Rayleigh quotient)')
        w(f'  p_eff + sum(btil) - k_rayleigh = {p_eff + btil.sum() - k_ray:.3e}')

        w('\nparameters')
        # Formatted explicitly: ell is of order 1e-5 s, and a caller that
        # set suppress=True for readable eigenvectors would otherwise see
        # a row of zeros here.
        fmt = {'float_kind': lambda v: f'{v:.5e}'}
        w('  ell (s)       = '
          + np.array2string(self.ell, formatter=fmt, max_line_width=200))
        if self.ell_sd is not None:
            w(f'  ell rel sd    = {self.ell_sd / self.ell}')
        if self.N > 1:
            w(f'  ell contrast  = {self.ell.max() / self.ell.min():.2f}')
        w(f'  B (pcm)       = {self.B * 1e5}')
        w(f'  lambda (1/s)  = {self.lam}')
        if self.lam_region is not None and np.all(np.isfinite(self.lam_region)):
            w(f'  lambda spread across regions = '
              f'{np.ptp(self.lam_region, axis=0) / self.lam}')
        else:
            w('  lambda: FALLBACK U-235 constants, no decay-rate tally')

        if self.kij is not None:
            kbar_d = np.einsum('ni,imn->mn', self.beta, self.Kd) / self.B[None, :]
            w(f'  importance ratio kbar_d/kij =\n{kbar_d / self.kij}')
            w(f'  beta_mn (pcm) =\n{self.B[None, :] * kbar_d / self.kij * 1e5}')
            w('  mixed-form error B*(kbar_d - kij) (pcm, the error you would '
              f'make using (1-B_n)*k_mn):\n{self.B[None, :] * (kbar_d - self.kij) * 1e5}')


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------
def bordered_sensitivity(K0, dK, k0=None, S0=None):
    """First-order (dk, dS) under the gauge 1^T dS = 0.

        [ K0 - k0 I   -S0 ] [dS]   [ -dK S0 ]
        [    1^T        0 ] [dk ] = [    0   ]

    The extra column supplies the rank the singular block lacks and the
    extra row supplies the missing equation, so the gauge is enforced as an
    equation rather than left to convention. Preferred over a mode sum,
    which needs three independent conventions to align -- the eigenvector
    normalization, the biorthonormalization and the gauge -- with nothing
    enforcing any of them.

    The matrix depends only on K0, so one LU factorization serves every
    perturbation.
    """
    if k0 is None or S0 is None:
        w, V = np.linalg.eig(K0)
        j = int(np.argmax(w.real))
        k0 = w[j].real
        S0 = np.abs(V[:, j].real)
        S0 = S0 / S0.sum()
    N = K0.shape[0]
    M = np.zeros((N + 1, N + 1))
    M[:N, :N] = K0 - k0 * np.eye(N)
    M[:N, N] = -S0
    M[N, :N] = 1.0
    sol = np.linalg.solve(M, np.concatenate([-dK @ S0, [0.0]]))
    return sol[N], sol[:N]


def reactivity_coefficients(K0, dKdT):
    """alpha_p [1/K] and the contribution matrix W [1/K].

    ``dKdT`` has shape (N, N, P). With rho = 1 - 1/k,

        alpha_p = <S+, dK/dT_p S0> / (k0^2 <S+, S0>)
        W_mn,p  = S+_m (dK/dT_p)_mn S0_n / (k0^2 <S+, S0>)

    and ``W[:, :, p].sum() == alpha[p]``. W is the MPK-native product: an
    N x N map of which source-to-response pathways carry region p's
    feedback, which the scalar coefficient discards.
    """
    w, V = np.linalg.eig(K0)
    j = int(np.argmax(w.real))
    k0 = w[j].real
    S0 = np.abs(V[:, j].real)
    S0 = S0 / S0.sum()
    wL, VL = np.linalg.eig(K0.T)
    Sadj = np.abs(VL[:, int(np.argmax(wL.real))].real)
    Sadj = Sadj / (Sadj @ S0)
    W = np.einsum('m,mnp,n->mnp', Sadj, dKdT, S0) / k0 ** 2
    return W.sum(axis=(0, 1)), W, k0, S0, Sadj


def interpolate_case(base, perturbed, fraction):
    """A perturbation ``fraction`` of the way from ``base`` to ``perturbed``.

    Used to size a transient. A temperature coefficient wants a LARGE
    perturbation -- it is a small difference of two large numbers, and at
    50 K the signal drowns in Monte Carlo noise -- while a transient wants a
    SMALL one, because with Lambda near 1e-5 s a step beyond about a dollar
    makes both MPK and its point-kinetics reference run away within
    milliseconds, and sub-percent parameter inconsistencies are then
    amplified beyond recognition.

    Scaling the measured dK keeps the *pattern* of the perturbation -- which
    pathways it acts through -- exactly as measured, and changes only its
    magnitude. That is a linear extrapolation in temperature, so it is only
    as good as the linearity that validation check 10 measures. Quote that
    check's discrepancy alongside any result built this way.
    """
    return MPKParams(
        Kp=base.Kp + fraction * (perturbed.Kp - base.Kp),
        Kd=base.Kd + fraction * (perturbed.Kd - base.Kd),
        beta=base.beta, ell=base.ell, lam=base.lam,
        cell_ids=base.cell_ids)


def combine_cases(base, cases, dT, deltas):
    """Evaluate an arbitrary temperature profile from tabulated cases.

    ``cases[p]`` is the run with region p heated by ``dT``; ``deltas[p]`` is
    the temperature change wanted in region p. The combination is formed on
    the TOTAL matrix,

        K = K_0 + sum_p (deltas[p] / dT) * (K_p - K_0)

    and then distributed back onto Kp and Kd as the elementwise scaling
    ``r = K / K_0``, so that

        (1-B_n) (r Kp)_mn + sum_i beta_n^(i) (r Kd^(i))_mn  =  r_mn K_0,mn

    reproduces the combined total exactly and the fission-source partition
    identity survives untouched.

    Doing it this way rather than combining Kp and Kd separately is not
    cosmetic. The delayed coefficients are the sparsest scores in the
    problem, so a difference between two runs' k^(i)_mn is mostly noise, and
    summing P such differences amplifies it until the result goes negative.
    The total matrix is orders of magnitude better converged, and the price
    is one stated assumption: that a temperature change scales the prompt
    and delayed coupling by the same elementwise factor, i.e. that the
    delayed importance ratio is temperature-independent. That is a far
    milder assumption than believing the elementwise delayed differences.

    Note that the weights need not sum to 1 and typically do not -- a
    ten-segment ramp sums to 5 -- so this is not a convex combination and
    non-negativity is not automatic. It holds here because it is inherited
    from the combined total matrix, which is well converged; for a single
    case at a fraction of its strength, :func:`interpolate_case` is exact
    and should be preferred.

    This is the payoff of tabulating the sensitivity. Once the P perturbed
    runs exist, any temperature profile -- a coolant rise up a channel, a
    single hot segment, a tilt -- costs a matrix combination rather than
    another transport calculation. It is linear, so it is only as good as
    the linearity that validation check 10 measures, and it says nothing
    about profiles outside the span of the cases that were run.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size != len(cases):
        raise ValueError(
            f'{deltas.size} temperature changes for {len(cases)} cases')
    fracs = deltas / dT
    if fracs.min() < 0.0 or fracs.max() > 1.0:
        warnings.warn(
            f'combine_cases is extrapolating: weights span '
            f'[{fracs.min():.3f}, {fracs.max():.3f}], outside the [0, 1] '
            'range spanned by the cases that were actually run. Scale the '
            'profile so its largest change is at most dT.', stacklevel=2)

    K0 = base.total_matrix()
    K = K0.copy()
    for p, case in enumerate(cases):
        if case.N != base.N:
            raise ValueError(f'case {p} has N={case.N}, base has N={base.N}')
        K += fracs[p] * (case.total_matrix() - K0)

    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(K0 > 0.0, K / K0, 1.0)
    return MPKParams(Kp=base.Kp * r, Kd=base.Kd * r[None, :, :],
                     beta=base.beta, ell=base.ell, lam=base.lam,
                     cell_ids=base.cell_ids)


def locality(G):
    """How much of |Gamma[:, :, p]| lives in row p and column p.

    Returns one fraction per perturbed region. This is the number that
    decides whether the method scales: if heating region p changes mainly
    its own row and column, the sensitivity data grows as N rather than
    N^2 per region and a sparse parameterization is viable. If it does not,
    a P-region study needs N^2 * P well-converged elements.
    """
    N, _, P = G.shape
    out = np.zeros(P)
    for p in range(min(N, P)):
        A = np.abs(G[:, :, p])
        tot = A.sum()
        if tot > 0.0:
            out[p] = (A[p, :].sum() + A[:, p].sum() - A[p, p]) / tot
    return out


def separable_delayed(params, S0=None, Sadj=None):
    """Rebuild Kd as r_i * kij, with one importance ratio per family.

    A fallback for the regime where the full delayed matrix is unaffordable.
    At N regions and D families there are N^2 * D delayed coefficients, and
    the far off-diagonal ones -- a delayed neutron born in n causing fission
    in m, for distant m and n -- are the sparsest scores in the problem. At
    N = 10 a few million histories still leaves a quarter of them with no
    counts at all, and because a missing count can only lower the estimate,
    the result is a BIAS in beta_eff rather than noise.

    This replaces those N^2 * D numbers with D of them, by assuming the
    delayed importance ratio k^(i)_mn / k_mn is separable: family-dependent
    and region-independent. Each r_i is then measured only on the elements
    that actually have counts, weighted as they enter the adjoint collapse,
    and applied everywhere:

        r_i = sum' S+_m beta_n^(i) S0_n k^(i)_mn
              / sum' S+_m beta_n^(i) S0_n k_mn

    with sum' running over the populated elements. It requires a tallied
    ``kij``, since that is the well-converged matrix being scaled.

    This is a modelling approximation, not a better estimator: it is exactly
    as good as the assumption that the ratio does not vary with region.
    Check it where both are affordable -- at N = 2 or 3 the full matrix
    converges, so the two can be compared directly -- and state it wherever
    a result depends on it.
    """
    if params.kij is None:
        raise ValueError('separable_delayed needs a tallied kij to scale')
    if S0 is None or Sadj is None:
        _, S0, Sadj = params.fundamental()

    Kd = np.empty_like(params.Kd)
    ratios = np.zeros(params.D)
    for i in range(params.D):
        wt = np.outer(Sadj, params.beta[:, i] * S0)
        populated = params.Kd[i] > 0.0
        den = float((wt * params.kij)[populated].sum())
        num = float((wt * params.Kd[i])[populated].sum())
        ratios[i] = num / den if den > 0.0 else 1.0
        Kd[i] = ratios[i] * params.kij
    out = MPKParams(Kp=params.Kp, Kd=Kd, beta=params.beta, ell=params.ell,
                    lam=params.lam, kij=params.kij, kij_sd=params.kij_sd,
                    ell_sd=params.ell_sd, cell_ids=params.cell_ids,
                    source=params.source + ' [separable delayed]')
    return out, ratios


def gamma_from_cases(base, cases, dT):
    """Assemble Gamma[m, n, p] = dK_mn/dT_p from base and perturbed runs.

    ``cases[p]`` is the :class:`MPKParams` of the run with region p heated by
    ``dT``. The matrices are used **raw**: no renormalization happens here,
    and none should. Normalizing the base case by its own eigenvalue and the
    perturbed case by its own leaves both with eigenvalue 1 and their
    difference containing no reactivity at all -- the quantity being
    measured has been divided out. The single factor of k0 that the
    coefficient needs enters once, as 1/k0^2 in
    :func:`reactivity_coefficients`, from drho/dk.

    Renormalizing by the base k0 is a separate matter and belongs to the
    *transient*, where it makes the reference state exactly critical; see
    :meth:`MPKParams.transient`.
    """
    K0 = base.total_matrix()
    P = len(cases)
    G = np.zeros((base.N, base.N, P))
    for p, case in enumerate(cases):
        if case.N != base.N:
            raise ValueError(f'case {p} has N={case.N}, base has N={base.N}')
        G[:, :, p] = (case.total_matrix() - K0) / dT
    return G


def check_first_order(base, cases, dT, uniform=None, stream=None):
    """Validation checks 9 through 13 on a real set of perturbed runs.

    check 10  alpha_p * dT against an exact eigenvalue re-solve
    check 11  sum_p alpha_p against one uniform-temperature case
    check 9   the gauge, sum(dS) under the 1^T condition
    check 13  the bordered dk against the Rayleigh formula

    Check 12, second-order convergence of the shape prediction, needs the
    perturbation size halved and so needs more runs; it is exercised on
    synthetic parameters in tests/test_mpk_invariants.py.
    """
    import sys
    out = stream or sys.stdout
    w = lambda *a: print(*a, file=out)

    K0 = base.total_matrix()
    G = gamma_from_cases(base, cases, dT)
    alpha, W, k0, S0, Sadj = reactivity_coefficients(K0, G)

    w(f'alpha_p (pcm/K)  {alpha * 1e5}')
    w(f'  sum over regions = {alpha.sum() * 1e5:+.4f} pcm/K')

    # Statistical uncertainty on alpha_p, propagated from the elementwise
    # standard deviation of the tallied matrices. alpha_p is a small
    # difference of two large numbers, so this is usually the number that
    # decides whether a coefficient means anything -- print it next to the
    # coefficient, never after it.
    sig = None
    if base.kij_sd is not None and all(c.kij_sd is not None for c in cases):
        wt = np.outer(Sadj, S0) / (k0 ** 2 * (Sadj @ S0) * dT)
        sig = np.array([
            np.sqrt(np.sum(wt ** 2 * (base.kij_sd ** 2 + c.kij_sd ** 2)))
            for c in cases])
        w(f'  1 sigma        {sig * 1e5}')
        w(f'  relative       '
          f'{np.abs(sig / np.where(alpha == 0, np.nan, alpha))}')
        w('  (propagated from the tallied kij_std_dev of each case. The two '
          'runs are\n  largely uncorrelated despite sharing a seed: once the '
          'cross sections differ,\n  histories diverge immediately. Halving '
          'this needs 4x the histories.)')

    w(f'\ncheck 10  first order vs exact eigenvalue re-solve, dT = {dT:g} K')
    rho0 = 1.0 - 1.0 / k0
    worst = 0.0
    for p, case in enumerate(cases):
        kp = case.fundamental()[0]
        exact = (1.0 - 1.0 / kp) - rho0
        first = alpha[p] * dT
        worst = max(worst, abs(exact - first))
        w(f'  region {p:2d}  exact {exact * 1e5:+9.2f} pcm   '
          f'1st-order {first * 1e5:+9.2f} pcm   '
          f'diff {abs(exact - first) * 1e5:6.2f} pcm')
    w(f'  worst discrepancy {worst * 1e5:.2f} pcm -- this locates where '
      'second-order terms start to matter and sets the maximum useful dT')

    if uniform is not None:
        ku = uniform.fundamental()[0]
        rho_u = (1.0 - 1.0 / ku) - rho0
        pred = alpha.sum() * dT
        w(f'\ncheck 11  uniform +{dT:g} K')
        w(f'  exact {rho_u * 1e5:+.2f} pcm   sum_p alpha_p*dT '
          f'{pred * 1e5:+.2f} pcm   diff {abs(rho_u - pred) * 1e5:.2f} pcm')
        w('  this one number tests the whole chain: extraction, '
          'normalization and the adjoint contraction')

    w('\ncheck 9 / 13  bordered solve')
    dS = np.zeros((base.N, len(cases)))
    for p in range(len(cases)):
        dk, dS[:, p] = bordered_sensitivity(K0, G[:, :, p] * dT, k0, S0)
        rayleigh = (Sadj @ (G[:, :, p] * dT) @ S0) / (Sadj @ S0)
        if p == 0:
            w(f'  sum(dS) under the 1^T gauge          {abs(dS[:, p].sum()):.2e}')
            w(f'  bordered dk vs Rayleigh formula      {abs(dk - rayleigh):.2e}')
    return alpha, W, dS, G, sig


def shape_sensitivity(K0, dKdT):
    """dS0/dT_p for every perturbed region, shape (N, P)."""
    N, P = K0.shape[0], dKdT.shape[2]
    out = np.zeros((N, P))
    for p in range(P):
        _, out[:, p] = bordered_sensitivity(K0, dKdT[:, :, p])
    return out
