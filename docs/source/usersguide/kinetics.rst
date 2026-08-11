.. _kinetics:

===================
Kinetics parameters
===================

OpenMC has the capability to estimate the following adjoint-weighted effective
generation time :math:`\Lambda_{\text{eff}}` and the effective delayed neutron
fraction :math:`\beta_{\text{eff}}`. These parameters are calculated using the
iterated fission probability (IFP) method [Hurwitz_1964]_ based on a similar
approach as in `Serpent 2 <https://doi.org/10.1016/j.anucene.2013.10.032>`_. The
implementation in OpenMC is limited to eigenvalue calculations and is described
in more details in [Dorville_2025]_.

----------------------------------
Iterated Fission Probability (IFP)
----------------------------------

With IFP, additional information needs to be recorded during the simulation
compared to a typical eigenvalue calculation. OpenMC stores an additional
set of values (neutron lifetime or delayed neutron group number for
:math:`\Lambda_{\text{eff}}` or :math:`\beta_{\text{eff}}`, respectively)
for every fission neutron simulated. Each set of values corresponds to
the values that are associated to the :math:`N_{\text{gen}}` direct ancestors
of any given fission neutron.

:math:`N_{\text{gen}}` is referred to as the number of generations in the
IFP method and corresponds to the number of generations between the birth of
a fission neutron and the time its score is added to the IFP tally. By default,
OpenMC considers 10 generations but this value can be modified by the user via
the ``ifp_n_generation`` settings in the Python API::

    settings.ifp_n_generation = 5

``ifp_n_generation`` should be greater than 0, but should also be lower than
or equal to the number of inactive batches declared for the calculation.
The respect of these constraints is verified by OpenMC before any calculation.

OpenMC will automatically detect the type of data that needs to be stored based
on the tally scores selected by the user. This guarantees that only information
of interest are stored during a simulation and avoids using extra memory when
only one parameter is needed. The following table shows the tally scores that
are needed to compute kinetics parameters in OpenMC:

.. table:: **OpenMC tally scores needed to calculate adjoint-weighted kinetics parameters**
    :align: center

    =============================== ============================ ========================== ========
    OpenMC tally score \\ Parameter :math:`\Lambda_{\text{eff}}` :math:`\beta_{\text{eff}}` Both
    =============================== ============================ ========================== ========
    ``ifp-time-numerator``          X                                                       X
    ``ifp-beta-numerator``                                       X                          X
    ``ifp-denominator``             X                            X                          X
    =============================== ============================ ========================== ========

|

.. note:: Because the memory footprint of additional data is generally non-negligible
    with IFP, it is recommended to choose the value for ``ifp_n_generation`` carefully.
    For example, using one generation for both kinetics parameters corresponds to store
    one additional integer (for the delayed neutron group number used with
    :math:`\beta_{\text{eff}}`) and one floating point value (for the neutron lifetime
    used with :math:`\Lambda_{\text{eff}}`) for every fission neutron simulated once the
    asymptotic regime is reached.

-----------------------------
Obtaining kinetics parameters
-----------------------------

The ``Model`` class can be used to automatically generate all IFP tallies using
the Python API with :attr:`openmc.Settings.ifp_n_generation` greater than 0 and
the :meth:`openmc.Model.add_ifp_kinetics_tallies` method::

    model = openmc.Model(geometry, settings=settings)
    model.add_kinetics_parameters_tallies(num_groups=6)  # Add 6 precursor groups

Alternatively, each of the tallies can be manually defined using group-wise or
total :math:`\beta_{\text{eff}}` specified by providing a 6-group
:class:`openmc.DelayedGroupFilter`::

    beta_tally = openmc.Tally(name="group-beta-score")
    beta_tally.scores = ["ifp-beta-numerator"]

    # Add DelayedGroupFilter to enable group-wise tallies
    beta_tally.filters = [openmc.DelayedGroupFilter(list(range(1, 7)))]

Here is an example showing how to declare the three available IFP scores in a
single tally::

    tally = openmc.Tally(name="ifp-scores")
    tally.scores = [
        "ifp-time-numerator",
        "ifp-beta-numerator",
        "ifp-denominator"
    ]

The effective generation time :math:`\Lambda_{\text{eff}}` is calculated
by dividing the result of the ``ifp-time-numerator`` score by the one obtained
for ``ifp-denominator`` and by the :math:`k_{\text{eff}}` of the simulation:

.. math::
    :label: lambda_eff

    \Lambda_{\text{eff}} = \frac{S_{\text{ifp-time-numerator}}}{S_{\text{ifp-denominator}} \times k_{\text{eff}}}

The effective delayed neutron fraction :math:`\beta_{\text{eff}}` is calculated
by dividing the result of the ``ifp-beta-numerator`` score by the one obtained
for ``ifp-denominator``:

.. math::
    :label: beta_eff

    \beta_{\text{eff}} = \frac{S_{\text{ifp-beta-numerator}}}{S_{\text{ifp-denominator}}}

The kinetics parameters can be retrieved directly from a statepoint file using
the :meth:`openmc.StatePoint.ifp_results` method::

    with openmc.StatePoint(output_path) as sp:
        generation_time, beta_eff = sp.get_kinetics_parameters()

----------------------------------------------
Region-to-region fission matrix (k_ij / k_dij)
----------------------------------------------

For multipoint-kinetics analyses of loosely-coupled or spatially extended
systems, OpenMC can additionally estimate a stochastic, region-to-region
fission matrix following the analog Monte Carlo estimators described by
Tommasi, Aufiero, and Vicoli. :math:`k_{ij}` is the average number of
fission neutrons produced in region :math:`i` by a neutron born in region
:math:`j`. :math:`k_{d,ij}` further resolves this quantity by delayed
neutron precursor family :math:`d` (family 0 denotes prompt neutrons),
which is needed to obtain family-resolved effective lifetimes and
sensitivity coefficients for multipoint kinetics.

Both quantities are estimated on a per-generation basis, exactly like
OpenMC's k-effective generation estimators: for each active generation
:math:`n`,

.. math::

    k_{ij}^{(n)} = \frac{p_{ij}^{(n)}}{s_j^{(n)}}

where :math:`s_j^{(n)}` is the total starting weight of source particles
born in region :math:`j` at the start of generation :math:`n`, and
:math:`p_{ij}^{(n)}` is the total weight of new fission neutrons produced
in region :math:`i` during generation :math:`n` by particles born in region
:math:`j`. The reported mean and variance are the sample mean and variance
of :math:`k_{ij}^{(n)}` across active generations,
:math:`E[k_{ij}] = \frac{1}{N}\sum_n k_{ij}^{(n)}` and
:math:`\text{Var}(k_{ij}) = E[(k_{ij}^{(n)})^2] - E[k_{ij}]^2`.

Regions :math:`i` and :math:`j` are defined using a matched pair of spatial
filters -- either :class:`openmc.CellFilter`/:class:`openmc.CellBornFilter`
for discrete cell-based regions, or :class:`openmc.MeshFilter`/
:class:`openmc.MeshBornFilter` for spatially-discretized mesh-based
regions -- assigned to :attr:`openmc.Settings.fission_matrix_i_filter` and
:attr:`openmc.Settings.fission_matrix_j_filter`. The delayed-family
resolution for :math:`k_{d,ij}` is optional and enabled by additionally
setting :attr:`openmc.Settings.fission_matrix_d_filter` to a
:class:`openmc.DelayedGroupBornFilter`::

    cell_a = openmc.Cell(...)
    cell_b = openmc.Cell(...)

    settings.fission_matrix_i_filter = openmc.CellFilter([cell_a, cell_b])
    settings.fission_matrix_j_filter = openmc.CellBornFilter([cell_a, cell_b])

    # Optional: also resolve by delayed neutron precursor family (0 = prompt)
    settings.fission_matrix_d_filter = openmc.DelayedGroupBornFilter(range(7))

This feature is only available in k-eigenvalue calculations. The resulting
matrices can be read back from a statepoint file::

    with openmc.StatePoint(output_path) as sp:
        kij_mean = sp.kij_mean          # shape (n_i, n_j)
        kij_std_dev = sp.kij_std_dev
        kdij_mean = sp.kdij_mean        # shape (n_i, n_j, n_d), if requested
        kdij_std_dev = sp.kdij_std_dev

.. only:: html

   .. rubric:: References

.. [Hurwitz_1964] H. Hurwitz Jr., "Naval Reactors Physics Handbook", volume 1, p. 864.
    Radkowsky, A. (Ed.), Naval Reactors, Division of Reactor Development, U.S.
    Atomic Energy Commission (1964).

.. [Dorville_2025] J. Dorville, L. Labrie-Cleary, and P. K. Romano, "Implementation
    of the Iterated Fission Probability Method in OpenMC to Compute Adjoint-Weighted
    Kinetics Parameters", International Conference on Mathematics and Computational
    Methods Applied to Nuclear Science and Engineering (M&C 2025), Denver, April 27-30,
    2025.
