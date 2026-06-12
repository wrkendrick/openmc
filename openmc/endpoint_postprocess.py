"""Postprocessing for OpenMC endpoint-track output.

Two capabilities, both built on :mod:`endpoint_tracks`:

1. :func:`plot_paths` -- draw neutron trajectories (birth -> collisions -> death)
   as a 2-D projection, optionally over an OpenMC geometry slice.

2. :func:`fission_transfer_matrix` -- overlay an N x N Cartesian grid of square
   regions on a 2-D projection and build the (N^2 x N^2) matrix whose entry
   ``k_ij`` is the probability that a neutron born in region ``i`` creates a
   fission in region ``j``. Plus helpers to visualize it.

Projection convention matches OpenMC's plot ``basis``:
``'xy'`` -> (x, y), ``'xz'`` -> (x, z), ``'yz'`` -> (y, z). The horizontal axis
is the first letter, the vertical axis the second.

Region indexing (row-major, horizontal fastest):
``region = iv * N + ih`` for grid column ``ih`` and row ``iv``. Use
:func:`region_to_grid`, :func:`grid_to_region`, and :func:`region_extent` to map
between region index, grid cell, and spatial extent.

Only numpy + matplotlib are required for the analysis; the optional geometry
background uses whatever OpenMC plotting entry point your build exposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize

try:  # work both as a package module and as a standalone file
    from .endpoint_tracks import EndpointTracks, read_endpoint_tracks
except ImportError:  # pragma: no cover
    from endpoint_tracks import EndpointTracks, read_endpoint_tracks

__all__ = [
    "plot_paths",
    "fission_transfer_matrix",
    "FissionTransferResult",
    "fission_transfer_matrix_batched",
    "BatchedFissionTransferResult",
    "plot_transfer_matrix",
    "plot_source_region_map",
    "plot_relative_error",
    "region_to_grid",
    "grid_to_region",
    "region_extent",
]

_BASIS_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


# ---------------------------------------------------------------------------
# Projection / grid utilities
# ---------------------------------------------------------------------------
def _axes(basis: str) -> tuple[int, int]:
    if basis not in _BASIS_AXES:
        raise ValueError(f"basis must be one of {list(_BASIS_AXES)}, got {basis!r}")
    return _BASIS_AXES[basis]


def _project(positions: np.ndarray, basis: str) -> np.ndarray:
    """(..., 3) xyz -> (..., 2) in the chosen plane."""
    h, v = _axes(basis)
    positions = np.asarray(positions, dtype=float)
    return np.stack([positions[..., h], positions[..., v]], axis=-1)


def _neutron_states_xyz(neutron) -> np.ndarray:
    """Ordered (k, 3) xyz of a neutron: birth, collisions..., death."""
    states = np.concatenate(
        [np.atleast_1d(neutron.birth), neutron.collisions, np.atleast_1d(neutron.death)]
    )
    r = states["r"]
    return np.column_stack([r["x"], r["y"], r["z"]])


def region_to_grid(region: int, n: int) -> tuple[int, int]:
    """region index -> (ih, iv) grid column/row."""
    return int(region % n), int(region // n)


def grid_to_region(ih: int, iv: int, n: int) -> int:
    """(ih, iv) -> region index."""
    return int(iv) * n + int(ih)


def region_extent(region: int, n: int, bounds: Sequence[float]) -> tuple:
    """Spatial extent (h0, h1, v0, v1) of a region cell."""
    lo_h, hi_h, lo_v, hi_v = bounds
    dh = (hi_h - lo_h) / n
    dv = (hi_v - lo_v) / n
    ih, iv = region_to_grid(region, n)
    return (lo_h + ih * dh, lo_h + (ih + 1) * dh, lo_v + iv * dv, lo_v + (iv + 1) * dv)


def _auto_bounds(tracks: EndpointTracks, basis: str, margin: float = 0.0) -> tuple:
    pts = np.vstack([tracks.birth_positions(), tracks.death_positions()])
    xy = _project(pts, basis)
    lo_h, lo_v = xy.min(axis=0)
    hi_h, hi_v = xy.max(axis=0)
    if margin:
        span_h, span_v = hi_h - lo_h, hi_v - lo_v
        lo_h -= margin * span_h
        hi_h += margin * span_h
        lo_v -= margin * span_v
        hi_v += margin * span_v
    return (float(lo_h), float(hi_h), float(lo_v), float(hi_v))


def _region_indices(xy: np.ndarray, bounds: Sequence[float], n: int) -> np.ndarray:
    """(M, 2) projected points -> (M,) region index, or -1 if outside the grid."""
    lo_h, hi_h, lo_v, hi_v = bounds
    h, v = xy[:, 0], xy[:, 1]
    ih = np.floor((h - lo_h) / (hi_h - lo_h) * n).astype(int)
    iv = np.floor((v - lo_v) / (hi_v - lo_v) * n).astype(int)
    inside = (ih >= 0) & (ih < n) & (iv >= 0) & (iv < n)
    region = np.where(inside, iv * n + ih, -1)
    return region


# ---------------------------------------------------------------------------
# Optional OpenMC geometry background
# ---------------------------------------------------------------------------
def _geometry_axes(model, basis, origin, width, ax, **plot_kwargs):
    """Try to render a geometry slice and return an Axes. Defensive across
    OpenMC versions; returns a blank Axes if no plot entry point works."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    if model is None:
        return ax

    common = dict(basis=basis, origin=origin, width=width, axes=ax)
    common = {k: v for k, v in common.items() if v is not None}
    common.update(plot_kwargs)

    for target, kwargs in (
        (getattr(model, "plot", None), common),
        (getattr(getattr(model, "geometry", None), "plot", None), common),
        (getattr(getattr(getattr(model, "geometry", None), "root_universe", None),
                 "plot", None), common),
    ):
        if target is None:
            continue
        try:
            returned = target(**kwargs)
            # some versions ignore `axes` and return a fresh Axes
            return returned if hasattr(returned, "plot") else ax
        except Exception as exc:  # noqa: BLE001 - build-specific signatures vary
            warnings.warn(f"geometry plot via {target} failed ({exc}); trying next.")
            continue

    warnings.warn(
        "Could not render a geometry background with this OpenMC build. "
        "Pass your own `ax=` (e.g. from model.plot(...)) to overlay paths."
    )
    return ax


# ---------------------------------------------------------------------------
# 1. Path visualization
# ---------------------------------------------------------------------------
def plot_paths(
    tracks: EndpointTracks,
    *,
    ax=None,
    model=None,
    basis: str = "xy",
    origin=None,
    width=None,
    max_paths: int = 500,
    color_by: str = "energy",
    only_fission: bool = False,
    show_births: bool = True,
    show_deaths: bool = True,
    grid_n: Optional[int] = None,
    grid_bounds=None,
    linewidth: float = 0.6,
    alpha: float = 0.6,
    seed: int = 0,
    geometry_kwargs: Optional[dict] = None,
):
    """Overlay neutron trajectories on a geometry slice.

    Each neutron contributes a polyline birth -> collisions -> death (a straight
    birth->death chord if collisions were not recorded).

    Parameters
    ----------
    tracks : EndpointTracks
        Loaded endpoint tracks.
    ax : matplotlib Axes, optional
        Draw onto this Axes. If None, one is created (and a geometry background
        is drawn if `model` is given).
    model : openmc.Model, optional
        If given (and `ax` is None), the geometry slice is rendered underneath.
        Tries model.plot -> geometry.plot -> root_universe.plot, defensively.
    basis, origin, width :
        Passed to the geometry plot and used for the projection. `basis` is one
        of 'xy', 'xz', 'yz'.
    max_paths : int
        Cap on the number of trajectories drawn (randomly sampled if exceeded);
        thousands of overlapping lines are unreadable.
    color_by : {'energy', 'fission', 'solid'}
        'energy'  -> color each path by its birth energy (log scale).
        'fission' -> fission deaths vs. other terminations in two colors.
        'solid'   -> single color.
    only_fission : bool
        Draw only neutrons whose terminal event was a fission.
    show_births, show_deaths : bool
        Scatter birth points (o) and death points (x); fission deaths are circled.
    grid_n, grid_bounds :
        If `grid_n` is set, draw the N x N region grid (using `grid_bounds`, or
        the geometry/data bounds) for context.

    Returns
    -------
    matplotlib.axes.Axes
    """
    neutrons = [nn for nn in tracks if (nn.fission_death or not only_fission)]
    if not neutrons:
        raise ValueError("No neutrons to plot (check `only_fission`).")

    rng = np.random.default_rng(seed)
    if len(neutrons) > max_paths:
        sel = rng.choice(len(neutrons), size=max_paths, replace=False)
        neutrons = [neutrons[k] for k in sel]

    ax = _geometry_axes(
        model, basis, origin, width, ax, **(geometry_kwargs or {})
    )

    segments, energies, fission_flags = [], [], []
    for nn in neutrons:
        xy = _project(_neutron_states_xyz(nn), basis)
        segments.append(xy)
        energies.append(float(nn.birth["E"]))
        fission_flags.append(bool(nn.fission_death))
    energies = np.asarray(energies)
    fission_flags = np.asarray(fission_flags)

    if color_by == "energy":
        emin = max(energies[energies > 0].min(), 1e-5) if np.any(energies > 0) else 1e-5
        lc = LineCollection(
            segments, cmap="viridis",
            norm=LogNorm(vmin=emin, vmax=energies.max()),
            linewidths=linewidth, alpha=alpha,
        )
        lc.set_array(energies)
        ax.add_collection(lc)
        cb = ax.figure.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("birth energy [eV]")
    elif color_by == "fission":
        colors = np.where(fission_flags, "#c0392b", "#7f8c8d")
        ax.add_collection(
            LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha)
        )
    elif color_by == "solid":
        ax.add_collection(
            LineCollection(segments, colors="#2c3e50", linewidths=linewidth, alpha=alpha)
        )
    else:
        raise ValueError("color_by must be 'energy', 'fission', or 'solid'.")

    if show_births:
        b = _project(np.array([nn.birth["r"].tolist() for nn in neutrons]), basis)
        ax.scatter(b[:, 0], b[:, 1], s=6, c="#27ae60", marker="o",
                   label="birth", zorder=3, edgecolors="none")
    if show_deaths:
        d = _project(np.array([nn.death["r"].tolist() for nn in neutrons]), basis)
        ax.scatter(d[~fission_flags, 0], d[~fission_flags, 1], s=8, c="k",
                   marker="x", label="death", zorder=3, linewidths=0.6)
        if fission_flags.any():
            ax.scatter(d[fission_flags, 0], d[fission_flags, 1], s=18,
                       facecolors="none", edgecolors="#c0392b", marker="o",
                       label="fission death", zorder=4, linewidths=1.0)

    if grid_n is not None:
        gb = grid_bounds or _data_or_axis_bounds(ax, tracks, basis)
        _draw_grid(ax, grid_n, gb)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{basis[0]} [cm]")
    ax.set_ylabel(f"{basis[1]} [cm]")
    ax.autoscale_view()
    if show_births or show_deaths:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    return ax


def _data_or_axis_bounds(ax, tracks, basis):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if xlim != (0.0, 1.0):  # axis already framed by geometry
        return (xlim[0], xlim[1], ylim[0], ylim[1])
    return _auto_bounds(tracks, basis)


def _draw_grid(ax, n, bounds):
    lo_h, hi_h, lo_v, hi_v = bounds
    for h in np.linspace(lo_h, hi_h, n + 1):
        ax.axvline(h, color="0.5", lw=0.4, alpha=0.5, zorder=1)
    for v in np.linspace(lo_v, hi_v, n + 1):
        ax.axhline(v, color="0.5", lw=0.4, alpha=0.5, zorder=1)


# ---------------------------------------------------------------------------
# 2. Fission transfer matrix
# ---------------------------------------------------------------------------
@dataclass
class FissionTransferResult:
    """Result of :func:`fission_transfer_matrix`.

    Attributes
    ----------
    n : int
        Grid is n x n; matrix is (n^2, n^2).
    bounds : (h0, h1, v0, v1)
        Spatial extent of the grid in the projection plane.
    basis : str
    normalize : {'birth', 'fission'}
        'birth'   -> k_ij = P(fission in j | born in i) = counts_ij / births_i
        'fission' -> k_ij = P(fission in j | born in i AND fissions) (rows sum 1)
    matrix : (n^2, n^2)
        The k_ij probability matrix.
    counts : (n^2, n^2) int
        Raw counts: neutrons born in i with a fission death in j.
    births : (n^2,) int
        Neutrons born in each region (denominator for 'birth').
    fissions_source : (n^2,) int
        Neutrons born in i that fission anywhere (denominator for 'fission').
    stderr : (n^2, n^2)
        Binomial standard error of each k_ij.
    n_born_out, n_fission_out : int
        Neutrons dropped because birth was outside the grid, and (born-in-grid)
        neutrons whose fission landed outside the grid.
    edges_h, edges_v : (n+1,)
        Grid cell edges.
    """

    n: int
    bounds: tuple
    basis: str
    normalize: str
    matrix: np.ndarray
    counts: np.ndarray
    births: np.ndarray
    fissions_source: np.ndarray
    stderr: np.ndarray
    n_born_out: int
    n_fission_out: int
    edges_h: np.ndarray
    edges_v: np.ndarray

    @property
    def k(self) -> np.ndarray:
        """Alias for `matrix` (the k_ij entries)."""
        return self.matrix


def fission_transfer_matrix(
    tracks: EndpointTracks,
    n: int,
    *,
    bounds=None,
    basis: str = "xy",
    normalize: str = "birth",
) -> FissionTransferResult:
    """Build the (n^2 x n^2) fission transfer matrix k_ij.

    ``k_ij`` is the probability that a neutron born in region ``i`` creates a
    fission in region ``j`` (its terminal event is a fission located in ``j``).

    Parameters
    ----------
    tracks : EndpointTracks
        Loaded endpoint tracks. For ``normalize='birth'`` the file must contain
        all births (run with ``fission_only=False``); otherwise the denominator
        is incomplete and only ``normalize='fission'`` is meaningful.
    n : int
        Number of grid cells per axis (so n^2 regions).
    bounds : (h0, h1, v0, v1), optional
        Spatial extent of the grid in the projection plane. Defaults to the
        data extent; pass the geometry bounds for a grid aligned to your model.
    basis : {'xy', 'xz', 'yz'}
    normalize : {'birth', 'fission'}
        See :class:`FissionTransferResult`.

    Returns
    -------
    FissionTransferResult
    """
    if normalize not in ("birth", "fission"):
        raise ValueError("normalize must be 'birth' or 'fission'.")
    if getattr(tracks, "fission_only", False) and normalize == "birth":
        warnings.warn(
            "These tracks were written with fission_only=True, so non-fission "
            "births are missing and the 'birth' normalization underestimates the "
            "denominator. Use normalize='fission' (conditional) instead, or "
            "re-run with fission_only=False.",
        )

    if bounds is None:
        bounds = _auto_bounds(tracks, basis)
    bounds = tuple(float(b) for b in bounds)

    birth_xy = _project(tracks.birth_positions(), basis)
    death_xy = _project(tracks.death_positions(), basis)
    fission = np.array([nn.fission_death for nn in tracks], dtype=bool)

    birth_reg = _region_indices(birth_xy, bounds, n)
    death_reg = _region_indices(death_xy, bounds, n)

    nreg = n * n
    in_grid_birth = birth_reg >= 0

    births = np.bincount(birth_reg[in_grid_birth], minlength=nreg).astype(np.int64)
    fissions_source = np.bincount(
        birth_reg[in_grid_birth & fission], minlength=nreg
    ).astype(np.int64)

    # counts_ij: fission neutrons with both birth and fission in the grid
    valid = in_grid_birth & fission & (death_reg >= 0)
    flat = birth_reg[valid] * nreg + death_reg[valid]
    counts = np.bincount(flat, minlength=nreg * nreg).reshape(nreg, nreg).astype(np.int64)

    denom = births if normalize == "birth" else fissions_source
    denom_col = denom[:, None].astype(float)
    matrix = np.divide(
        counts, denom_col, out=np.zeros_like(counts, dtype=float), where=denom_col > 0
    )
    # binomial standard error sqrt(p(1-p)/N)
    stderr = np.divide(
        matrix * (1.0 - matrix), denom_col,
        out=np.zeros_like(matrix), where=denom_col > 0,
    )
    np.sqrt(stderr, out=stderr)

    n_born_out = int((~in_grid_birth).sum())
    n_fission_out = int((in_grid_birth & fission & (death_reg < 0)).sum())

    lo_h, hi_h, lo_v, hi_v = bounds
    return FissionTransferResult(
        n=n, bounds=bounds, basis=basis, normalize=normalize,
        matrix=matrix, counts=counts, births=births,
        fissions_source=fissions_source, stderr=stderr,
        n_born_out=n_born_out, n_fission_out=n_fission_out,
        edges_h=np.linspace(lo_h, hi_h, n + 1),
        edges_v=np.linspace(lo_v, hi_v, n + 1),
    )


def plot_transfer_matrix(result: FissionTransferResult, *, ax=None, log=False,
                         cmap="magma", show_grid_lines=True):
    """Heatmap of the full (n^2 x n^2) matrix; rows = source region i, cols =
    fission region j. Both axes are ordered by region index (iv*n + ih)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    M = result.matrix
    norm = LogNorm(vmin=max(M[M > 0].min(), 1e-6), vmax=M.max()) if (log and np.any(M > 0)) \
        else Normalize(vmin=0, vmax=M.max() if M.max() > 0 else 1)
    im = ax.imshow(M, origin="upper", cmap=cmap, norm=norm, aspect="equal",
                   interpolation="nearest")
    nreg = result.n * result.n
    if show_grid_lines and result.n <= 16:
        for b in range(result.n, nreg, result.n):
            ax.axhline(b - 0.5, color="white", lw=0.4, alpha=0.4)
            ax.axvline(b - 0.5, color="white", lw=0.4, alpha=0.4)
    ax.set_xlabel("fission region  j")
    ax.set_ylabel("source region  i")
    ax.set_title(
        f"k_ij = P(fission in j | born in i)"
        + ("" if result.normalize == "birth" else ", conditional on fissioning")
    )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("probability")
    return ax


def plot_source_region_map(result: FissionTransferResult, source_region: int, *,
                           ax=None, cmap="inferno", mark_source=True):
    """Reshape one source region's row back to the N x N grid: a spatial map of
    where neutrons born in `source_region` go on to cause fission."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    row = result.matrix[source_region].reshape(result.n, result.n)  # [iv, ih]
    lo_h, hi_h, lo_v, hi_v = result.bounds
    im = ax.imshow(row, origin="lower", extent=(lo_h, hi_h, lo_v, hi_v),
                   cmap=cmap, aspect="equal", interpolation="nearest")
    if mark_source:
        s0, s1, t0, t1 = region_extent(source_region, result.n, result.bounds)
        ax.add_patch(plt.Rectangle((s0, t0), s1 - s0, t1 - t0, fill=False,
                                   edgecolor="cyan", lw=1.8, zorder=3))
    ih, iv = region_to_grid(source_region, result.n)
    ax.set_title(f"fission destinations for births in region {source_region} "
                 f"(ih={ih}, iv={iv})")
    ax.set_xlabel(f"{result.basis[0]} [cm]")
    ax.set_ylabel(f"{result.basis[1]} [cm]")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(fission here | born in source region)")
    return ax


# ---------------------------------------------------------------------------
# 2b. Fission transfer matrix with Monte Carlo batch statistics
# ---------------------------------------------------------------------------
def _accumulate_counts(birth_reg, death_reg, fission, nreg, idx=None):
    """Counts and denominators over an (optional) subset of neutron indices.

    Returns
    -------
    counts : (nreg, nreg)  fissions from born-i to fission-j
    births : (nreg,)       neutrons born in each region
    fissions_source : (nreg,)  neutrons born in i that fission anywhere
    """
    if idx is not None:
        br, dr, fis = birth_reg[idx], death_reg[idx], fission[idx]
    else:
        br, dr, fis = birth_reg, death_reg, fission
    in_birth = br >= 0
    births = np.bincount(br[in_birth], minlength=nreg).astype(float)
    fissions_source = np.bincount(br[in_birth & fis], minlength=nreg).astype(float)
    valid = in_birth & fis & (dr >= 0)
    counts = (np.bincount(br[valid] * nreg + dr[valid], minlength=nreg * nreg)
              .reshape(nreg, nreg).astype(float))
    return counts, births, fissions_source


def _make_batches(n_total, fission, sim_batch, n_per_batch, batch_unit,
                  shuffle, seed, drop_partial):
    """Partition neutron indices into statistical batches.

    batch_unit:
      'neutron'   -- n_per_batch recorded neutrons per batch (default).
      'fission'   -- n_per_batch fissioning neutrons per batch (births in the
                     same contiguous span ride along as the denominator).
      'sim_batch' -- group by the simulation batch encoded in the file; then
                     n_per_batch simulation batches make one statistical batch.
    """
    order = np.arange(n_total)
    if shuffle:
        if batch_unit == "sim_batch":
            raise ValueError("shuffle is incompatible with batch_unit='sim_batch'.")
        np.random.default_rng(seed).shuffle(order)

    if batch_unit == "neutron":
        edges = list(range(0, n_total + 1, n_per_batch))
        batches = [order[a:b] for a, b in zip(edges[:-1], edges[1:])]
        tail = order[edges[-1]:]
        if len(tail) and not drop_partial:
            batches.append(tail)

    elif batch_unit == "fission":
        fis = fission[order].astype(int)
        # batch id = (# fissions strictly before this position) // n_per_batch
        bid = (np.cumsum(fis) - fis) // n_per_batch
        total_fis = int(fis.sum())
        n_full = total_fis // n_per_batch
        batches = [order[bid == b] for b in range(n_full)]
        if total_fis % n_per_batch and not drop_partial and n_full < bid.max() + 1:
            batches.append(order[bid == n_full])

    elif batch_unit == "sim_batch":
        uniq = list(dict.fromkeys(sim_batch.tolist()))  # preserve first-seen order
        groups = [uniq[a:a + n_per_batch] for a in range(0, len(uniq), n_per_batch)]
        if drop_partial and groups and len(groups[-1]) < n_per_batch and len(groups) > 1:
            groups = groups[:-1]
        batches = [order[np.isin(sim_batch, g)] for g in groups]

    else:
        raise ValueError("batch_unit must be 'neutron', 'fission', or 'sim_batch'.")

    return [b for b in batches if len(b) > 0]


@dataclass
class BatchedFissionTransferResult:
    """Batch-statistics version of the fission transfer matrix.

    Each statistical batch yields one realization ``K_b[i,j]`` of the transfer
    probability; the reported value is the across-batch mean and the standard
    error of that mean, exactly as OpenMC reports tally means/std_dev.

    A batch contributes to row ``i`` only if it has at least one neutron born in
    region ``i`` (otherwise that row's ratio is undefined for the batch), so the
    effective batch count is tracked per row in ``n_contributing``.

    Attributes
    ----------
    mean : (n^2, n^2)        across-batch mean of k_ij
    std_mean : (n^2, n^2)    standard error of the mean (the reported sigma)
    std_batch : (n^2, n^2)   std dev of the per-batch realizations
    rel_err : (n^2, n^2)     std_mean / mean (NaN where mean == 0)
    pooled : (n^2, n^2)      ratio-of-sums estimate (all neutrons at once), for
                             cross-checking the batch mean
    n_batches : int          total statistical batches formed
    n_contributing : (n^2,)  batches contributing to each source row
    batches : list or None   per-batch matrices if keep_batches=True
    """

    n: int
    bounds: tuple
    basis: str
    normalize: str
    n_per_batch: int
    batch_unit: str
    n_batches: int
    mean: np.ndarray
    std_mean: np.ndarray
    std_batch: np.ndarray
    rel_err: np.ndarray
    pooled: np.ndarray
    n_contributing: np.ndarray
    edges_h: np.ndarray
    edges_v: np.ndarray
    batches: Optional[list] = None

    # Duck-compatible with FissionTransferResult so plot_transfer_matrix and
    # plot_source_region_map work unchanged.
    @property
    def matrix(self) -> np.ndarray:
        return self.mean

    @property
    def stderr(self) -> np.ndarray:
        return self.std_mean

    @property
    def k(self) -> np.ndarray:
        return self.mean

    def summary(self) -> str:
        active = self.mean > 0
        re = self.rel_err[active & np.isfinite(self.rel_err)]
        lines = [
            f"BatchedFissionTransferResult: {self.n}x{self.n} grid "
            f"({self.n * self.n} regions), {self.n_batches} batches "
            f"of ~{self.n_per_batch} ({self.batch_unit}), normalize={self.normalize!r}",
            f"  source rows with <2 contributing batches (sigma undefined): "
            f"{int((self.n_contributing < 2).sum())} / {self.n * self.n}",
        ]
        if re.size:
            lines.append(
                f"  relative error over nonzero entries: "
                f"median={np.median(re):.3f}, mean={re.mean():.3f}, max={re.max():.3f}"
            )
        return "\n".join(lines)


def fission_transfer_matrix_batched(
    tracks: EndpointTracks,
    n: int,
    n_per_batch: int,
    *,
    bounds=None,
    basis: str = "xy",
    normalize: str = "birth",
    batch_unit: str = "neutron",
    shuffle: bool = False,
    seed: int = 0,
    drop_partial: bool = True,
    keep_batches: bool = False,
) -> BatchedFissionTransferResult:
    """Fission transfer matrix with per-batch mean and standard error.

    The recorded neutrons are partitioned into statistical batches; each batch
    ``b`` gives a full matrix ``K_b[i,j] = counts_b[i,j] / denom_b[i]``. Across
    batches, this accumulates the running sum and sum-of-squares (just like an
    OpenMC tally) and reports::

        mean[i,j]     = (1/M_i) * sum_b K_b[i,j]
        std_batch[i,j]= sample std dev of {K_b[i,j]}
        std_mean[i,j] = std_batch[i,j] / sqrt(M_i)        # the reported sigma

    where ``M_i`` is the number of batches with at least one birth in region i.

    Parameters
    ----------
    tracks : EndpointTracks
    n : int
        Grid is n x n (n^2 regions).
    n_per_batch : int
        Batch size, in the unit set by `batch_unit`.
    bounds, basis, normalize :
        As in :func:`fission_transfer_matrix`. For ``normalize='birth'`` the
        denominator is births; run with ``fission_only=False``.
    batch_unit : {'neutron', 'fission', 'sim_batch'}
        What `n_per_batch` counts. 'neutron' (default) keeps each batch's birth
        denominator well-populated; 'fission' fixes the fission count per batch
        (matching "n_per_batch fissioning histories"); 'sim_batch' aligns
        statistical batches with the simulation batches recorded in the file
        (the safest choice for k-eigenvalue runs, where histories within a
        generation are correlated).
    shuffle : bool
        Randomize neutron order before chunking (not allowed with 'sim_batch').
        Use for fixed-source data if the recorded order has structure; leave
        False to preserve simulation locality.
    drop_partial : bool
        Drop a final undersized batch (recommended for clean statistics).
    keep_batches : bool
        Store every per-batch matrix in `.batches` (memory ~ M * n^4 floats).

    Returns
    -------
    BatchedFissionTransferResult
    """
    if normalize not in ("birth", "fission"):
        raise ValueError("normalize must be 'birth' or 'fission'.")
    if getattr(tracks, "fission_only", False) and normalize == "birth":
        warnings.warn(
            "Tracks written with fission_only=True; non-fission births are "
            "missing, so 'birth' normalization is biased. Use normalize='fission'."
        )

    if bounds is None:
        bounds = _auto_bounds(tracks, basis)
    bounds = tuple(float(b) for b in bounds)

    birth_reg = _region_indices(_project(tracks.birth_positions(), basis), bounds, n)
    death_reg = _region_indices(_project(tracks.death_positions(), basis), bounds, n)
    fission = np.array([nn.fission_death for nn in tracks], dtype=bool)
    sim_batch = np.array([nn.identifier[0] for nn in tracks], dtype=np.int64)

    nreg = n * n
    batch_idx = _make_batches(len(tracks), fission, sim_batch, n_per_batch,
                              batch_unit, shuffle, seed, drop_partial)
    if not batch_idx:
        raise ValueError("No batches formed; check n_per_batch and batch_unit.")

    S1 = np.zeros((nreg, nreg))
    S2 = np.zeros((nreg, nreg))
    Mrow = np.zeros(nreg, dtype=np.int64)
    pooled_counts = np.zeros((nreg, nreg))
    pooled_denom = np.zeros(nreg)
    kept = [] if keep_batches else None

    for idx in batch_idx:
        counts_b, births_b, fis_b = _accumulate_counts(
            birth_reg, death_reg, fission, nreg, idx
        )
        denom_b = births_b if normalize == "birth" else fis_b
        has = denom_b > 0
        Kb = np.zeros((nreg, nreg))
        Kb[has] = counts_b[has] / denom_b[has, None]

        S1[has] += Kb[has]
        S2[has] += Kb[has] ** 2
        Mrow[has] += 1
        pooled_counts += counts_b
        pooled_denom += denom_b
        if keep_batches:
            kept.append(Kb)

    Mcol = Mrow[:, None].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(Mcol > 0, S1 / Mcol, 0.0)
        var_batch = np.where(Mcol > 1, (S2 - S1 ** 2 / Mcol) / (Mcol - 1), np.nan)
        var_batch = np.clip(var_batch, 0.0, None)  # guard tiny negatives
        std_batch = np.sqrt(var_batch)
        std_mean = np.where(Mcol > 1, std_batch / np.sqrt(Mcol), np.nan)
        pooled = np.where(pooled_denom[:, None] > 0,
                          pooled_counts / pooled_denom[:, None], 0.0)
        rel_err = np.where(mean > 0, std_mean / mean, np.nan)

    n_underdetermined = int((Mrow < 2).sum())
    if n_underdetermined:
        warnings.warn(
            f"{n_underdetermined} source region(s) had <2 contributing batches; "
            f"their sigma is undefined (NaN). Increase n_per_batch or run longer."
        )

    lo_h, hi_h, lo_v, hi_v = bounds
    return BatchedFissionTransferResult(
        n=n, bounds=bounds, basis=basis, normalize=normalize,
        n_per_batch=n_per_batch, batch_unit=batch_unit, n_batches=len(batch_idx),
        mean=mean, std_mean=std_mean, std_batch=std_batch, rel_err=rel_err,
        pooled=pooled, n_contributing=Mrow,
        edges_h=np.linspace(lo_h, hi_h, n + 1),
        edges_v=np.linspace(lo_v, hi_v, n + 1),
        batches=kept,
    )


def plot_relative_error(result, *, ax=None, cmap="cividis", min_mean=0.0,
                        vmax=None):
    """Heatmap of the relative error std_mean/mean over the matrix. Entries with
    ``mean <= min_mean`` (and undefined sigma) are masked grey."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    re = np.ma.masked_invalid(np.where(result.matrix > min_mean, result.rel_err, np.nan))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.85")
    im = ax.imshow(re, origin="upper", cmap=cmap_obj, aspect="equal",
                   interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_xlabel("fission region  j")
    ax.set_ylabel("source region  i")
    ax.set_title("relative error of k_ij  (sigma / mean)")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("relative error")
    return ax


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "endpoint_tracks.h5"
    tr = read_endpoint_tracks(path)
    print(tr)
    res = fission_transfer_matrix(tr, n=8)
    print(f"matrix shape {res.matrix.shape}, "
          f"{res.n_born_out} births out-of-grid, "
          f"{res.n_fission_out} fissions out-of-grid")
    plot_paths(tr, color_by="energy", grid_n=8)
    plot_transfer_matrix(res)
    plt.show()
