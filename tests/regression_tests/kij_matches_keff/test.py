import numpy as np
import openmc

from tests.testing_harness import PyAPITestHarness


class KijMatchesKeffTestHarness(PyAPITestHarness):
    """Region-to-region fission matrix (k_ij) regression test.

    For a model with exactly one fissionable region, k_ij is a 1x1 matrix
    whose only entry is, by definition, the same eigenvalue as k-effective:
    every fission neutron is born in that one region, so "neutrons produced
    in region i per neutron born in region j" collapses to "neutrons
    produced per neutron born," which is k-effective itself. This test
    exists to catch any future regression in the k_ij estimator (see the
    bugs fixed in commit "Fix k_ij fission-matrix estimator to agree with
    k-effective") that would cause the two to diverge again.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A bare critical-ish sphere of low-enriched fuel surrounded by a
        # water reflector/moderator -- small and fast, but with a soft
        # enough spectrum that (n,xn) multiplicity reactions (which is what
        # exposed the k_ij/keff bug) still occur often enough to matter.
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

        self._model.materials = openmc.Materials([fuel, water])

        crit_radius = 16.1
        outer_radius = 3.0 * crit_radius
        s1 = openmc.Sphere(r=crit_radius)
        s2 = openmc.Sphere(r=outer_radius, boundary_type='vacuum')
        fuel_cell = openmc.Cell(cell_id=1, fill=fuel, region=-s1)
        water_cell = openmc.Cell(cell_id=2, fill=water, region=+s1 & -s2)
        self._model.geometry = openmc.Geometry([fuel_cell, water_cell])

        settings = openmc.Settings()
        settings.run_mode = 'eigenvalue'
        settings.particles = 2000
        settings.batches = 100
        settings.inactive = 10

        uniform_dist = openmc.stats.Box(
            [-crit_radius] * 3, [crit_radius] * 3)
        settings.source = openmc.IndependentSource(
            space=uniform_dist, constraints={'fissionable': True})

        # k_ij with i = j = the (only) fuel cell
        settings.fission_matrix_i_filter = openmc.CellFilter(
            bins=[fuel_cell])
        settings.fission_matrix_j_filter = openmc.CellBornFilter(
            bins=[fuel_cell])

        self._model.settings = settings

    def _get_results(self):
        """Digest info in the statepoint and return as a string."""
        with openmc.StatePoint(self._sp_name) as sp:
            keff = sp.keff
            kij_mean = float(sp.kij_mean[0, 0])
            kij_std_dev = float(sp.kij_std_dev[0, 0])

        # The physics invariant this test protects: with a single
        # fissionable region, k_ij must equal k-effective. The two are
        # independent Monte Carlo estimators (different tallies entirely),
        # so they won't agree bit-for-bit.
        #
        # The bugs fixed in the commit referenced above biased k_ij by only
        # a few tenths of a percent -- a systematic shift that is real and
        # reproduces at any particle/batch count, but that a fast,
        # CI-sized run (as used here) does not have the statistical power
        # to distinguish from noise at high confidence (empirically, ~1-1.5
        # combined standard deviations at this problem size; getting that
        # bias to several sigma requires ~10x more histories and a couple
        # minutes of runtime). The loose check below is therefore a coarse
        # sanity guard against gross errors (e.g. a completely broken
        # tally), not the precise regression check -- that job belongs to
        # the ordinary bit-for-bit results_true.dat comparison performed by
        # the rest of this harness, which *did* catch the original bugs
        # (verified by reverting the fix and confirming this test fails).
        combined_std_dev = np.sqrt(keff.s**2 + kij_std_dev**2)
        diff = abs(kij_mean - keff.n)
        assert diff < 10 * combined_std_dev, (
            f'k_ij ({kij_mean:.6f} +/- {kij_std_dev:.6f}) does not agree '
            f'with k-effective ({keff.n:.6f} +/- {keff.s:.6f}) for this '
            'single-fissionable-region model, where the two are '
            f'mathematically required to be equal (difference = {diff:.6f}, '
            f'{diff / combined_std_dev:.1f} combined standard deviations).'
        )

        outstr = 'k-combined:\n'
        outstr += f'{keff.n:12.6E} {keff.s:12.6E}\n'
        outstr += 'k_ij:\n'
        outstr += f'{kij_mean:12.6E} {kij_std_dev:12.6E}\n'
        return outstr


def test_kij_matches_keff():
    harness = KijMatchesKeffTestHarness(
        'statepoint.100.h5', model=openmc.Model())
    harness.main()
