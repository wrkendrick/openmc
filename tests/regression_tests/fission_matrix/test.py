"""Test the region-to-region fission matrix (k_ij / k_dij) tallies, based on
the multipoint-kinetics fission-matrix estimators of Tommasi, Aufiero, and
Vicoli."""

import glob

import numpy as np
import openmc
import pytest

from tests.testing_harness import PyAPITestHarness


class FissionMatrixTestHarness(PyAPITestHarness):
    def _get_results(self):
        outstr = super()._get_results()

        statepoint = glob.glob(self._sp_name)[0]
        with openmc.StatePoint(statepoint) as sp:
            # Sanity checks that don't depend on stochastic noise
            assert sp.kij_mean.shape == (2, 2)
            assert np.all(np.isfinite(sp.kij_mean))
            assert np.all(sp.kij_mean >= 0.0)
            assert sp.kdij_mean.shape == (2, 2, 7)
            assert np.all(np.isfinite(sp.kdij_mean))
            assert np.all(sp.kdij_mean >= 0.0)

            outstr += 'kij_mean:\n'
            outstr += '\n'.join(f'{x:12.6E}' for x in sp.kij_mean.ravel())
            outstr += '\n'
            outstr += 'kij_std_dev:\n'
            outstr += '\n'.join(f'{x:12.6E}' for x in sp.kij_std_dev.ravel())
            outstr += '\n'
            outstr += 'kdij_mean:\n'
            outstr += '\n'.join(f'{x:12.6E}' for x in sp.kdij_mean.ravel())
            outstr += '\n'

        return outstr


@pytest.fixture()
def fission_matrix_model():
    model = openmc.Model()

    # Material
    material = openmc.Material(name='fuel')
    material.add_nuclide('U235', 1.0)
    material.set_density('g/cm3', 16.0)

    # Geometry: a single fissile sphere split into two weakly-coupled
    # regions by a plane through its center
    radius = 10.0
    sphere = openmc.Sphere(r=radius, boundary_type='vacuum')
    plane = openmc.XPlane(x0=0.0)
    cell_left = openmc.Cell(region=-sphere & -plane, fill=material)
    cell_right = openmc.Cell(region=-sphere & +plane, fill=material)
    model.geometry = openmc.Geometry([cell_left, cell_right])

    # Settings
    model.settings.particles = 1000
    model.settings.batches = 20
    model.settings.inactive = 5
    model.settings.run_mode = 'eigenvalue'

    lower_left = (-radius, -radius, -radius)
    upper_right = (radius, radius, radius)
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(lower_left, upper_right),
        constraints={'fissionable': True})

    # Region-to-region fission matrix (k_ij / k_dij)
    i_filter = openmc.CellFilter([cell_left, cell_right])
    j_filter = openmc.CellBornFilter([cell_left, cell_right])
    d_filter = openmc.DelayedGroupBornFilter(list(range(7)))
    model.settings.fission_matrix_i_filter = i_filter
    model.settings.fission_matrix_j_filter = j_filter
    model.settings.fission_matrix_d_filter = d_filter

    return model


def test_fission_matrix(fission_matrix_model):
    harness = FissionMatrixTestHarness(
        'statepoint.20.h5', model=fission_matrix_model)
    harness.main()
