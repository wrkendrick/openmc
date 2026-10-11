"""Guard against the generation-averaging bias in k_dij.

k_dij(i, j, d) is estimated as the across-generation mean of a per-generation
ratio p/s, where s is the total starting weight of neutrons of birth type d
born in region j. For a generation in which no such neutron was born, s is
zero and that generation carries no information about the element.

Folding a zero into the mean for those generations biases the estimate toward
zero in proportion to how often it happens. With N regions and D delayed
families, a single (j, d) source bin holds roughly beta^(d) / N of the bank --
a few hundredths of a percent -- so at a few thousand particles per generation
the sparser families have an empty bin most of the time and the bias reaches a
factor of several.

Two properties make the bias easy to miss and worth a dedicated test:

  * it scales with particles per GENERATION, not with total histories, so
    running more batches does not reduce it and the result looks converged
  * the fission-source partition identity does not catch it. The delayed half
    of the partition is weighted by beta ~ 1e-3, so a large relative deficit
    in k^(i)_mn is a small absolute deficit in K, and it simply moves into
    the prompt term -- leaving K right and the prompt/delayed split wrong.

The quantity that does catch it is the adjoint-weighted delayed importance
ratio: beta_eff from the delayed coefficients, over the same collapse with
every k^(i)_mn replaced by the tallied k_mn. Physically that sits within
about 10% of 1. Under the bias it collapses toward zero.

This model is deliberately configured to make the bias large if it is
present: four regions and a small generation size, so the sparse (j, d) bins
are empty in most generations. With the bias it measured about 0.6; without
it, about 1.0.
"""

import glob

import numpy as np
import openmc
import pytest

from tests.testing_harness import PyAPITestHarness

D = 6
CELL_IDS = [1, 2, 3, 4]


class DelayedBiasTestHarness(PyAPITestHarness):
    def _get_results(self):
        outstr = super()._get_results()

        statepoint = glob.glob(self._sp_name)[0]
        with openmc.StatePoint(statepoint) as sp:
            kij = np.asarray(sp.kij_mean)
            kdij = np.asarray(sp.kdij_mean)
            Kd = np.moveaxis(kdij[:, :, 1:], 2, 0)

            n = len(CELL_IDS)
            beta = np.zeros((n, D))
            for a, cell_id in enumerate(CELL_IDS):
                nu = _find(sp, 'nu-fission', cell_id)
                dnu = _find(sp, 'delayed-nu-fission', cell_id)
                beta[a] = dnu.mean.ravel() / nu.mean.ravel()[0]
            B = beta.sum(axis=1)

            K = (1.0 - B)[None, :] * kdij[:, :, 0] \
                + np.einsum('ni,imn->mn', beta, Kd)
            w, V = np.linalg.eig(K)
            j = int(np.argmax(w.real))
            S0 = np.abs(V[:, j].real)
            S0 /= S0.sum()
            wL, VL = np.linalg.eig(K.T)
            Sadj = np.abs(VL[:, int(np.argmax(wL.real))].real)
            Sadj /= Sadj @ S0

            # The adjoint-weighted delayed importance ratio. Both sides are
            # weighted identically, so this stays meaningful even when B
            # varies between regions -- a plain beta_eff / mean(B) does not.
            num = sum(Sadj @ (Kd[i] @ (beta[:, i] * S0)) for i in range(D))
            den = Sadj @ (kij @ (B * S0))
            ratio = float(num / den)

            n_zero = int(np.count_nonzero(Kd == 0.0))
            assert 0.85 < ratio < 1.25, (
                f'delayed importance ratio {ratio:.4f} is outside the '
                'physically plausible range. A value well below 1 on this '
                'model means k_dij is being averaged over generations whose '
                '(j, d) source bin was empty, which biases it toward zero; '
                'those generations must be excluded from the mean, not '
                f'counted as zeros. ({n_zero} of {Kd.size} delayed '
                'coefficients came back exactly zero.)')

            outstr += f'delayed_importance_ratio:\n{ratio:12.6E}\n'

        return outstr


def _find(sp, score, cell_id):
    hits = [t for t in sp.tallies.values()
            if t.scores == [score]
            and any(type(f) is openmc.CellFilter
                    and list(np.asarray(f.bins).ravel()) == [cell_id]
                    for f in t.filters)]
    assert len(hits) == 1, f'{len(hits)} tallies {score} on cell {cell_id}'
    return hits[0]


@pytest.fixture
def delayed_bias_model():
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

    # Four concentric fuel shells of equal volume, so every region holds a
    # quarter of the source and no single bin dominates
    crit = 16.1
    radii = [crit * ((i + 1) / 4) ** (1 / 3) for i in range(4)]
    surfaces = [openmc.Sphere(r=r) for r in radii]
    outer = openmc.Sphere(r=3.0 * crit, boundary_type='vacuum')

    cells = [openmc.Cell(cell_id=1, fill=fuel, region=-surfaces[0])]
    for i in range(1, 4):
        cells.append(openmc.Cell(cell_id=i + 1, fill=fuel,
                                 region=+surfaces[i - 1] & -surfaces[i]))
    cells.append(openmc.Cell(cell_id=5, fill=water,
                             region=+surfaces[-1] & -outer))
    model.geometry = openmc.Geometry(cells)

    settings = openmc.Settings(run_mode='eigenvalue')
    # A small generation size with many batches: the configuration that
    # makes the bias large while keeping the total history count cheap.
    # Raising batches alone does not reduce the bias, which is the point.
    settings.particles = 2000
    settings.batches = 150
    settings.inactive = 15
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([-crit] * 3, [crit] * 3),
        constraints={'fissionable': True})
    fissile = cells[:4]
    settings.fission_matrix_i_filter = openmc.CellFilter(bins=fissile)
    settings.fission_matrix_j_filter = openmc.CellBornFilter(bins=fissile)
    settings.fission_matrix_d_filter = openmc.DelayedGroupBornFilter(
        range(1 + D))
    model.settings = settings

    tallies = openmc.Tallies()
    one_group = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
    energy = openmc.EnergyFilter(one_group.group_edges)
    delayed = openmc.DelayedGroupFilter(list(range(1, 1 + D)))
    for cell in fissile:
        cf = openmc.CellFilter(cell)
        t = openmc.Tally(name=f'nu-fission {cell.id}')
        t.filters = [cf, energy]
        t.scores = ['nu-fission']
        tallies.append(t, merge=False)
        t = openmc.Tally(name=f'delayed-nu-fission {cell.id}')
        t.filters = [cf, delayed, energy]
        t.scores = ['delayed-nu-fission']
        tallies.append(t, merge=False)
    model.tallies = tallies

    return model


def test_fission_matrix_delayed_bias(delayed_bias_model):
    harness = DelayedBiasTestHarness(
        'statepoint.150.h5', model=delayed_bias_model)
    harness.main()
