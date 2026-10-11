"""Fission-source partition identity for the k_ij / k_dij / ell_i estimators.

This is the backbone correctness check for the multipoint-kinetics parameter
set. Every neutron produced in region n is either prompt or a member of
exactly one delayed family, so the total coupling coefficient decomposes
exactly:

    k_mn = (1 - B_n) k^(p)_mn + sum_i beta_n^(i) k^(i)_mn

The three ingredients on the right come from tallies that are accumulated
independently of k_ij: k^(p) and k^(i) are k_dij bins 0 and 1..D, and
beta_n^(i) is an ordinary forward openmc.mgxs.Beta tally. Reconstructing
k_ij from them and comparing is therefore a real check and not a tautology.

It is also the *only* check sensitive to an i/j transposition in the C++
estimator, because a transposed matrix has the same fundamental eigenvalue
and passes every eigenvalue-based test. The weights (1 - B_n) and
beta_n^(i) are indexed by the source region n, so they scale columns of
k_ij; transposing the matrix or putting beta on the wrong axis breaks the
identity elementwise.

What it does NOT test is the delayed coefficients themselves. The delayed
half of the partition enters weighted by beta_n^(i) ~ 1e-3, so the
reconstruction is dominated by the prompt term and the identity stays well
inside one sigma even when individual k^(i)_mn are badly under-converged.
Measured here: the identity holds to 0.4 sigma at 2e5 histories while the
delayed bins are still 40% low. Converging k^(i)_mn needs the history
counts of the theory manual's statistics section, and no cheap test
substitutes for that.

The model is deliberately asymmetric -- a small inner fuel sphere inside a
much larger outer fuel shell -- so that k_ij[1,0] and k_ij[0,1] differ by
more than an order of magnitude and the orientation of the i/j filter pair
is observable at all. A symmetric model cannot distinguish them.
"""

import glob

import numpy as np
import openmc
import pytest

from tests.testing_harness import PyAPITestHarness

# Delayed families resolved by the DelayedGroupBornFilter (bin 0 is prompt)
D = 6


class FissionMatrixPartitionTestHarness(PyAPITestHarness):
    def _get_results(self):
        outstr = super()._get_results()

        statepoint = glob.glob(self._sp_name)[0]
        with openmc.StatePoint(statepoint) as sp:
            kij = np.asarray(sp.kij_mean)
            kij_sd = np.asarray(sp.kij_std_dev)
            kdij = np.asarray(sp.kdij_mean)
            ell = np.asarray(sp.ell_mean)
            ell_sd = np.asarray(sp.ell_std_dev)

            assert kij.shape == (2, 2)
            assert kdij.shape == (2, 2, 1 + D)
            assert ell.shape == (2,)

            # --- i/j orientation.
            #
            # i (CellFilter) is the response region, where next-generation
            # neutrons are produced; j (CellBornFilter) is the source
            # region, where the current-generation neutron was born. The
            # outer shell holds most of the fuel, so a neutron born in the
            # small inner sphere produces far more fission in the outer
            # shell than the reverse. k_ij[1,0] must therefore be much
            # larger than k_ij[0,1]. If the two filters were swapped at
            # accumulation, this comparison inverts.
            assert kij[1, 0] > 2.0 * kij[0, 1], (
                f'k_ij orientation looks transposed: k_ij[1,0] = '
                f'{kij[1, 0]:.6f} is not much larger than k_ij[0,1] = '
                f'{kij[0, 1]:.6f} (expected a factor of about 5 for this '
                'geometry). i must be the response (CellFilter) region and '
                'j the source (CellBornFilter) region.')

            # --- k_dij bin convention.
            #
            # Bin 0 is the prompt coupling coefficient k^(p)_mn and bins
            # 1..D are the bare delayed coefficients k^(i)_mn. Each is
            # "expected production per neutron of its own type", so none
            # carries a delayed fraction and they do NOT sum to k_ij.
            # Misreading bin 0 as (1-B)k would put it ~700 pcm low;
            # misreading the array as a partition would inflate the
            # delayed source by a factor of about D.
            # --- raw delayed fractions, indexed by the SOURCE region n
            beta = np.zeros((2, D))
            for n, cell_id in enumerate(self._cell_ids):
                nu = _find_tally(sp, ['nu-fission'], cell_id)
                dnu = _find_tally(sp, ['delayed-nu-fission'], cell_id)
                beta[n] = dnu.mean.ravel() / nu.mean.ravel()[0]
            assert np.all(beta > 0.0)
            B = beta.sum(axis=1)
            beta_eff_scale = f'{B.mean():.4f}'

            Kp = kdij[:, :, 0]
            Kd = np.moveaxis(kdij[:, :, 1:], 2, 0)
            assert np.all(Kp >= 0.0) and np.all(Kd >= 0.0)
            prompt_ratio = Kp / kij
            assert np.all(np.abs(prompt_ratio - 1.0) < 0.05), (
                f'k_dij bin 0 is not the prompt coupling coefficient: '
                f'k_dij[...,0]/k_ij = {prompt_ratio}')
            # The delayed bins are bare coupling coefficients, each "per
            # neutron of its own family", so they sum to roughly D times
            # k_ij. If they were instead a partition of k_ij by birth type
            # the sum would be about beta_eff, i.e. ~0.008 -- three orders
            # of magnitude lower. The bounds below are deliberately loose
            # because individual delayed bins are the sparsest scores in
            # the problem and are not converged at a test-sized history
            # count; they are tight enough to settle the convention, which
            # is what this assertion is for. Misreading the array as a
            # partition would make the precursor source ~D*S per second
            # instead of ~beta*S and overflow any integrator in milliseconds.
            delayed_ratio = kdij[:, :, 1:].sum(axis=2) / kij
            assert np.all(delayed_ratio > 1.0), (
                f'k_dij delayed bins look like a partition of k_ij rather '
                f'than bare k^(i): sum/k_ij = {delayed_ratio}, expected of '
                f'order {D}, a partition would give about {beta_eff_scale}')
            assert np.all(delayed_ratio < 2.0 * D), (
                f'k_dij delayed bins are larger than bare k^(i) can be: '
                f'sum/k_ij = {delayed_ratio}, expected of order {D}')

            # --- the partition identity itself.
            #
            # 'ni,imn->mn' sums over i only; both n indices are the same
            # index, so nothing is contracted over regions. Writing
            # 'in,imn->mn' reads beta as (D, N) instead of (N, D) and is a
            # silent bug whenever N == D.
            K = (1.0 - B)[None, :] * Kp + np.einsum('ni,imn->mn', beta, Kd)
            z = np.abs(K - kij) / np.maximum(kij_sd, 1e-30)
            assert z.max() < 5.0, (
                f'fission-source partition identity off by {z.max():.1f} '
                f'sigma.\nreconstructed:\n{K}\ntallied k_ij:\n{kij}\n'
                f'difference (pcm):\n{(K - kij) * 1e5}\nin sigma:\n{z}\n'
                'Check the i/j filter orientation and that beta indexes '
                'the source region n (it scales columns, not rows).')

            # --- ell_i, the Kobayashi generation time.
            #
            # Scored at zero generation lag in the same fission-site loop,
            # so the 1/v track integral and the importance weighting belong
            # to the same particle. The inner sphere is fed preferentially
            # by fast neutrons born nearby, the outer shell also by
            # thermalized returns from the water reflector, so ell_0 must
            # be clearly shorter than ell_1. A region-independent ell is
            # the signature of an adjoint-weighted (IFP-style) estimator
            # having been substituted for G_i+.
            assert np.all(np.isfinite(ell)) and np.all(ell > 0.0)
            assert np.all(ell < 1.0e-3), f'implausible ell_i: {ell}'
            # The outer fuel shell is many thermal mean free paths thick,
            # so the ~1e-4 s reflector-return population fissions there and
            # never reaches the inner sphere: measured on this model, 0.0%
            # of production in the inner region is below 1 eV against ~20%
            # in the outer shell. ell therefore differs between the two
            # regions by orders of magnitude, not percent. A near-unity
            # contrast means an adjoint-weighted estimator has been
            # substituted for the zero-lag G_i+ weighting, which averages
            # the two populations together.
            assert ell[1] > 10.0 * ell[0], (
                f'ell is nearly region-independent: ell = {ell}. The '
                'zero-lag estimator should separate the fast and '
                'thermal-return populations between these two regions.')
            assert np.all(ell_sd > 0.0)

            # --- IFP spatial filters are evaluated at the fission site.
            #
            # The ell_i recipe used before the zero-lag estimator existed
            # relied on this, and the multi-region IFP results of any
            # future comparison still do: a CellFilter on an IFP score must
            # partition the unfiltered tally exactly, and a cell holding no
            # fissionable material must score exactly zero.
            unf = _ifp_pair(sp, None)
            fuel_parts = [_ifp_pair(sp, cid) for cid in self._cell_ids]
            water = _ifp_pair(sp, self._water_id)
            assert water == (0.0, 0.0), (
                f'non-fissionable cell scored a nonzero IFP tally: {water}')
            for which, idx in (('numerator', 0), ('denominator', 1)):
                total = sum(part[idx] for part in fuel_parts)
                assert abs(total - unf[idx]) <= 1.0e-12 * abs(unf[idx]), (
                    f'filtered IFP {which} ({total!r}) does not partition '
                    f'the unfiltered tally ({unf[idx]!r}); the spatial '
                    'filter is not being evaluated at the fission site.')

            # --- provenance
            assert sp.ifp_n_generation == 1, (
                'ifp_n_generation was not recorded in the statepoint; '
                'IFP scores from different generation counts are weighted '
                'differently and are not comparable.')

            outstr += 'kij_mean:\n'
            outstr += '\n'.join(f'{x:12.6E}' for x in kij.ravel()) + '\n'
            outstr += 'kdij_mean:\n'
            outstr += '\n'.join(f'{x:12.6E}' for x in kdij.ravel()) + '\n'
            outstr += 'ell_mean:\n'
            outstr += '\n'.join(f'{x:12.6E}' for x in ell.ravel()) + '\n'

        return outstr


def _find_tally(sp, scores, cell_id):
    """Look a tally up by score set and CellFilter bin.

    sp.tallies is keyed by tally ID, not position, and the iteration order of
    openmc.mgxs.Beta's tallies is an implementation detail -- positional
    indexing silently grabs the wrong tally as soon as a region is added.
    """
    hits = []
    for t in sp.tallies.values():
        if set(t.scores) != set(scores):
            continue
        cf = [f for f in t.filters if type(f) is openmc.CellFilter]
        if cf and list(np.asarray(cf[0].bins).ravel()) == [cell_id]:
            hits.append(t)
    if len(hits) != 1:
        raise LookupError(
            f'expected exactly 1 tally {scores} on cell {cell_id}, '
            f'found {len(hits)}')
    return hits[0]


def _ifp_pair(sp, cell_id):
    """(ifp-time-numerator, ifp-denominator) for one cell, or unfiltered."""
    scores = ['ifp-time-numerator', 'ifp-denominator']
    if cell_id is None:
        hits = [t for t in sp.tallies.values()
                if set(t.scores) == set(scores) and not t.filters]
        assert len(hits) == 1
        t = hits[0]
    else:
        t = _find_tally(sp, scores, cell_id)
    return (float(t.get_values(scores=['ifp-time-numerator']).ravel()[0]),
            float(t.get_values(scores=['ifp-denominator']).ravel()[0]))


@pytest.fixture
def partition_model():
    model = openmc.Model()

    fuel = openmc.Material(material_id=1, name='fuel')
    fuel.add_nuclide('U235', 0.5, 'ao')
    fuel.add_nuclide('U238', 0.5, 'ao')
    fuel.add_element('O', 2.0, 'ao')
    fuel.set_density('g/cm3', 10.2)

    water = openmc.Material(material_id=2, name='water')
    water.add_element('H', 2.0, 'ao')
    water.add_element('O', 1.0, 'ao')
    water.set_density('g/cm3', 0.7)
    water.add_s_alpha_beta('c_H_in_H2O')
    model.materials = openmc.Materials([fuel, water])

    # The fuel sphere split at r = 10 cm. The inner region holds about 24%
    # of the fuel volume: asymmetric enough that k_ij[1,0] and k_ij[0,1]
    # differ by a factor of ~5 and the i/j orientation is observable, but
    # not so lopsided that the smaller region's delayed k_dij bins go
    # unpopulated at a test-sized history count (at r = 5 cm the inner
    # region holds 3% of the fuel and its delayed bins are empty).
    inner, crit = 10.0, 16.1
    outer = 3.0 * crit
    s1 = openmc.Sphere(r=inner)
    s2 = openmc.Sphere(r=crit)
    s3 = openmc.Sphere(r=outer, boundary_type='vacuum')
    c1 = openmc.Cell(cell_id=1, fill=fuel, region=-s1)
    c2 = openmc.Cell(cell_id=2, fill=fuel, region=+s1 & -s2)
    c3 = openmc.Cell(cell_id=3, fill=water, region=+s2 & -s3)
    model.geometry = openmc.Geometry([c1, c2, c3])

    settings = openmc.Settings(run_mode='eigenvalue')
    settings.particles = 4000
    settings.batches = 90
    settings.inactive = 10
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([-crit] * 3, [crit] * 3),
        constraints={'fissionable': True})
    settings.fission_matrix_i_filter = openmc.CellFilter(bins=[c1, c2])
    settings.fission_matrix_j_filter = openmc.CellBornFilter(bins=[c1, c2])
    settings.fission_matrix_d_filter = openmc.DelayedGroupBornFilter(
        range(1 + D))
    # One generation: the IFP control tallies below are only used to confirm
    # that a spatial filter on an IFP score is evaluated at the fission site
    settings.ifp_n_generation = 1
    model.settings = settings

    tallies = openmc.Tallies()
    one_group = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    for cell in (c1, c2):
        betas = openmc.mgxs.Beta(domain=cell, domain_type='cell',
                                 energy_groups=one_group,
                                 delayed_groups=list(range(1, 1 + D)))
        for key in betas.tallies:
            tallies.append(betas.tallies[key], merge=False)

    t = openmc.Tally(tally_id=100, name='ifp_unfiltered')
    t.scores = ['ifp-time-numerator', 'ifp-denominator']
    tallies.append(t)
    for i, cell in enumerate((c1, c2, c3)):
        t = openmc.Tally(tally_id=101 + i, name=f'ifp_cell{cell.id}')
        t.filters = [openmc.CellFilter(cell)]
        t.scores = ['ifp-time-numerator', 'ifp-denominator']
        tallies.append(t)
    model.tallies = tallies

    return model


def test_fission_matrix_partition(partition_model):
    harness = FissionMatrixPartitionTestHarness(
        'statepoint.90.h5', model=partition_model)
    harness._cell_ids = [1, 2]
    harness._water_id = 3
    harness.main()
