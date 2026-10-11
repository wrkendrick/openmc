"""Unit tests for the region-to-region fission matrix API.

Covers the Python side only: the DelayedGroupBornFilter and the
Settings.fission_matrix_* properties, including their XML round trip and the
validation that stops a mismatched i/j filter pair reaching the solver. The
estimators themselves are exercised by
tests/regression_tests/fission_matrix_partition/.
"""

import numpy as np
import pytest

import openmc


def test_delayed_group_born_filter():
    f = openmc.DelayedGroupBornFilter(range(7))
    assert f.num_bins == 7
    # 0 (prompt) is a valid bin here, unlike DelayedGroupFilter
    assert list(f.bins) == list(range(7))
    assert f.short_name == 'DelayedGroupBorn'

    with pytest.raises(ValueError):
        openmc.DelayedGroupBornFilter([-1])

    # DelayedGroupFilter, by contrast, rejects 0
    with pytest.raises(ValueError):
        openmc.DelayedGroupFilter([0])


def test_delayed_group_born_filter_roundtrip():
    f = openmc.DelayedGroupBornFilter(range(7), filter_id=77)
    elem = f.to_xml_element()
    assert elem.get('type') == 'delayedgroupborn'

    g = openmc.Filter.from_xml_element(elem)
    assert isinstance(g, openmc.DelayedGroupBornFilter)
    assert g.id == 77
    assert list(g.bins) == list(f.bins)


def test_fission_matrix_settings_roundtrip(run_in_tmpdir):
    s = openmc.Settings(run_mode='eigenvalue', batches=10, inactive=2,
                        particles=100)
    s.source = openmc.IndependentSource(space=openmc.stats.Point())
    s.fission_matrix_i_filter = openmc.CellFilter([3, 4])
    s.fission_matrix_j_filter = openmc.CellBornFilter([3, 4])
    s.fission_matrix_d_filter = openmc.DelayedGroupBornFilter(range(7))
    s.ifp_n_generation = 1
    s.export_to_xml()

    t = openmc.Settings.from_xml()
    assert isinstance(t.fission_matrix_i_filter, openmc.CellFilter)
    assert isinstance(t.fission_matrix_j_filter, openmc.CellBornFilter)
    assert isinstance(t.fission_matrix_d_filter,
                      openmc.DelayedGroupBornFilter)
    # The filter bin order fixes the region indexing for every downstream
    # quantity (k_ij, k_dij, ell_i), so it has to survive the round trip.
    assert list(t.fission_matrix_i_filter.bins) == [3, 4]
    assert list(t.fission_matrix_j_filter.bins) == [3, 4]
    assert list(t.fission_matrix_d_filter.bins) == list(range(7))
    assert t.ifp_n_generation == 1


def test_fission_matrix_settings_roundtrip_without_d_filter(run_in_tmpdir):
    s = openmc.Settings(run_mode='eigenvalue', batches=10, inactive=2,
                        particles=100)
    s.source = openmc.IndependentSource(space=openmc.stats.Point())
    s.fission_matrix_i_filter = openmc.CellFilter([1])
    s.fission_matrix_j_filter = openmc.CellBornFilter([1])
    s.export_to_xml()

    t = openmc.Settings.from_xml()
    assert t.fission_matrix_d_filter is None


def test_fission_matrix_filter_type_validation():
    s = openmc.Settings()

    # i must be a CellFilter or MeshFilter
    with pytest.raises(TypeError):
        s.fission_matrix_i_filter = openmc.CellBornFilter([1])
    # j must be a CellBornFilter or MeshBornFilter
    with pytest.raises(TypeError):
        s.fission_matrix_j_filter = openmc.CellFilter([1])
    # d must be a DelayedGroupBornFilter, not the ordinary DelayedGroupFilter
    with pytest.raises(TypeError):
        s.fission_matrix_d_filter = openmc.DelayedGroupFilter([1, 2])


def test_fission_matrix_filter_pairing_validation(run_in_tmpdir):
    """i and j must be the matching cell-based or mesh-based pair."""
    s = openmc.Settings(run_mode='eigenvalue', batches=10, inactive=2,
                        particles=100)
    s.source = openmc.IndependentSource(space=openmc.stats.Point())

    # Only one of the pair given
    s.fission_matrix_i_filter = openmc.CellFilter([1])
    with pytest.raises(ValueError, match='[Bb]oth'):
        s.export_to_xml()

    # A CellFilter for i needs a CellBornFilter for j, not a MeshBornFilter
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-1., -1., -1.)
    mesh.upper_right = (1., 1., 1.)
    mesh.dimension = (2, 2, 2)
    s.fission_matrix_j_filter = openmc.MeshBornFilter(mesh)
    with pytest.raises(ValueError, match='CellBornFilter'):
        s.export_to_xml()
