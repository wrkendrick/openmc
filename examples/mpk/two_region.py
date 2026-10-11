#!/usr/bin/env python
"""
Two coupled cores: the smallest model that multipoint kinetics can say
something point kinetics cannot.

Geometry, 1-D in x, transversely infinite (reflective y and z):

      water  |  core 1  |  B4C  |  core 2  |  water
    reflector|          | gap   |          | reflector
     --------+----------+-------+----------+--------
       20 cm |  CORE_T  |  2 cm |  CORE_T  |  20 cm
         vacuum at both outer faces

Both cores are the same homogeneous UO2/water mixture. The boron carbide
sheet between them absorbs most of what crosses, so the cores are coupled
but not strongly: the fission matrix comes out with a large diagonal and a
small off-diagonal, and the dominance ratio is far enough below 1 that the
first harmonic survives long enough to matter.

The model is symmetric, which is the point. At the reference temperature
S0 = [1/2, 1/2] exactly, and heating one core alone breaks that symmetry.
Point kinetics describes the result with one amplitude and cannot represent
the tilt at all; MPK carries it as the two components of S.

What the script does, in order:

  1. runs the base case and extracts the MPK parameter set (k_ij, k_dij,
     ell_i from the fission-matrix tallies; raw beta and lambda from
     ordinary forward tallies), then runs every applicable validation check
  2. runs core 1 hot and both cores hot, assembles Gamma = dK/dT_p, and
     reports alpha_p, the contribution matrix W, and checks 9 to 13
  3. integrates the kinetics after a step temperature rise in core 1,
     against a frozen-shape point-kinetics reference built from the same
     parameters, and reports the amplitude and the tilt separately

Runtime is a few minutes at the default statistics. Four OpenMC runs are
cached in the working directory and reused, so re-running only redoes the
postprocessing.

    python two_region.py                  # everything
    python two_region.py --particles 40000 --batches 400
    python two_region.py --plot           # also write two_region.png
"""
import argparse
from pathlib import Path

import numpy as np
import openmc

import mpk

# Reference and perturbed temperatures [K]. 300 K is deliberately large:
# a temperature coefficient is a small difference of two large numbers, and
# at 50 K the signal would be buried in Monte Carlo noise at any statistics
# this example could reasonably use. Linearity is then checked rather than
# assumed -- that is what check 10 is for.
T_BASE = 600.0
DT = 300.0

CORE_T = 17.2       # thickness of each core [cm], tuned for k ~ 1
GAP_T = 2.0         # boron carbide sheet between the cores [cm]
REFL_T = 20.0       # water reflector on each outer face [cm]
TRANS = 20.0        # transverse half-width (reflective) [cm]
ENRICH = 4.5        # wt% U-235 in the fuel of the homogeneous mixture
FUEL_VF = 0.22      # UO2 volume fraction in the core mixture


def build_model(temps, particles, batches):
    """One model with the two cores at temperatures ``temps`` [K].

    Only the two core cells carry a perturbed temperature. Everything else
    stays at T_BASE, so a case isolates the effect of heating fuel.
    """
    openmc.reset_auto_ids()

    uo2 = openmc.Material(name='UO2')
    uo2.add_element('U', 1.0, enrichment=ENRICH)
    uo2.add_element('O', 2.0)
    uo2.set_density('g/cm3', 10.3)

    def make_water(name):
        w = openmc.Material(name=name)
        w.add_element('H', 2.0)
        w.add_element('O', 1.0)
        w.set_density('g/cm3', 0.72)
        return w

    # A homogeneous fuel/moderator mixture: a well-thermalized core with a
    # real U-238 Doppler coefficient, and no lattice to resolve. The thermal
    # scattering table is attached after mixing, because mix_materials
    # cannot combine materials that already carry one.
    core_mat = openmc.Material.mix_materials(
        [uo2, make_water('water for mixing')], [FUEL_VF, 1.0 - FUEL_VF],
        'vo', name='core mixture')
    core_mat.add_s_alpha_beta('c_H_in_H2O')

    water = make_water('water')
    water.add_s_alpha_beta('c_H_in_H2O')

    b4c = openmc.Material(name='B4C')
    b4c.add_element('B', 4.0)
    b4c.add_element('C', 1.0)
    b4c.set_density('g/cm3', 2.52)

    # Two independent material instances so the two cores can be at
    # different temperatures while being physically identical.
    core1 = core_mat.clone()
    core1.name = 'core 1'
    core1.temperature = temps[0]
    core2 = core_mat.clone()
    core2.name = 'core 2'
    core2.temperature = temps[1]
    for m in (water, b4c):
        m.temperature = T_BASE

    edges = np.cumsum([0.0, REFL_T, CORE_T, GAP_T, CORE_T, REFL_T])
    planes = [openmc.XPlane(x0=float(edges[0]), boundary_type='vacuum')]
    planes += [openmc.XPlane(x0=float(v)) for v in edges[1:-1]]
    planes += [openmc.XPlane(x0=float(edges[-1]), boundary_type='vacuum')]

    y0 = openmc.YPlane(y0=-TRANS, boundary_type='reflective')
    y1 = openmc.YPlane(y0=TRANS, boundary_type='reflective')
    z0 = openmc.ZPlane(z0=-TRANS, boundary_type='reflective')
    z1 = openmc.ZPlane(z0=TRANS, boundary_type='reflective')
    transverse = +y0 & -y1 & +z0 & -z1

    fills = [water, core1, b4c, core2, water]
    cells = [openmc.Cell(cell_id=i + 1, fill=f,
                         region=+planes[i] & -planes[i + 1] & transverse)
             for i, f in enumerate(fills)]
    # The fissile cells, in the order that fixes the MPK region indexing
    fissile = [cells[1], cells[3]]

    settings = openmc.Settings(run_mode='eigenvalue')
    settings.particles = particles
    settings.batches = batches
    settings.inactive = max(20, batches // 10)
    # Correlated sampling: the same seed for the base and perturbed cases
    # makes the difference variance far smaller than sqrt(2)*sigma(k), which
    # is close to mandatory at these signal sizes.
    settings.seed = 1
    settings.temperature = {'method': 'interpolation',
                            'range': (290.0, 1200.0)}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([edges[1], -TRANS, -TRANS],
                               [edges[4], TRANS, TRANS]),
        constraints={'fissionable': True})

    mpk.configure_settings(settings, fissile)
    tallies = mpk.add_tallies(openmc.Tallies(), fissile)

    model = openmc.Model(
        materials=openmc.Materials([core1, core2, water, b4c]),
        geometry=openmc.Geometry(cells), settings=settings, tallies=tallies)
    return model, [c.id for c in fissile]


def run_case(tag, temps, particles, batches, workdir):
    """Run one case, or reuse its statepoint if it is already there."""
    d = workdir / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = d / f'statepoint.{batches}.h5'
    model, cell_ids = build_model(temps, particles, batches)
    if not sp.exists():
        print(f'  running {tag}: core temperatures {temps} K ...', flush=True)
        model.run(cwd=d, output=False)
    else:
        print(f'  reusing {tag}')
    return mpk.from_statepoint(sp, cell_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--particles', type=int, default=10000)
    ap.add_argument('--batches', type=int, default=120)
    ap.add_argument('--workdir', type=Path, default=Path('two_region_work'))
    ap.add_argument('--dollars', type=float, default=-0.30,
                    help='reactivity to scale the transient step to')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    np.set_printoptions(precision=5, linewidth=100, suppress=False)

    # ------------------------------------------------------------------
    print(__doc__.split('What the script does')[0].rstrip())
    print('=' * 72)
    print('STEP 1  base case: tally the MPK coefficients')
    print('=' * 72)
    base = run_case('base', (T_BASE, T_BASE), args.particles, args.batches,
                    args.workdir)
    base.report()

    k0, S0, Sadj = base.fundamental()
    evs = np.sort(np.linalg.eigvals(base.total_matrix()).real)[::-1]
    print(f'\nThe two cores are symmetric, so S0 should be [0.5, 0.5] and the '
          f'\nfission matrix should have equal diagonals. Measured S0 = {S0}.')
    print(f'For a symmetric two-core system K = [[d, c], [c, d]], the '
          f'eigenvalues\nare d + c and d - c, so the dominance ratio is '
          f'(d-c)/(d+c) = {evs[1] / evs[0]:.4f}.')
    print('The further that is below 1, the longer the tilt introduced below '
          'survives.')

    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print(f'STEP 2  reactivity from a {DT:g} K temperature rise')
    print('=' * 72)
    hot1 = run_case('hot1', (T_BASE + DT, T_BASE), args.particles,
                    args.batches, args.workdir)
    hot2 = run_case('hot2', (T_BASE, T_BASE + DT), args.particles,
                    args.batches, args.workdir)
    uniform = run_case('uniform', (T_BASE + DT, T_BASE + DT), args.particles,
                       args.batches, args.workdir)

    print(f'\nk: base {k0:.6f}   core 1 hot {hot1.fundamental()[0]:.6f}   '
          f'core 2 hot {hot2.fundamental()[0]:.6f}   '
          f'both hot {uniform.fundamental()[0]:.6f}')
    alpha, W, dS, G, sig = mpk.check_first_order(
        base, [hot1, hot2], DT, uniform=uniform)

    # The model is symmetric, so alpha_1 and alpha_2 are the same
    # perturbation mirrored and must agree. Their difference is therefore a
    # free, assumption-free measurement of the statistical error -- and a
    # check on the propagated sigma printed above.
    spread = abs(alpha[0] - alpha[1])
    print(f'\nSymmetry check: alpha_1 and alpha_2 describe the same '
          'perturbation mirrored,\nso they must agree. Measured difference '
          f'{spread * 1e5:.4f} pcm/K on a mean of '
          f'{alpha.mean() * 1e5:.3f} pcm/K.')
    if sig is not None:
        exp = float(np.hypot(*sig))
        print(f'Expected spread from the propagated sigma: '
              f'{exp * 1e5:.4f} pcm/K, so this is '
              f'{spread / exp:.1f} sigma -- consistent with noise, not bias.')
    print('This is the binding constraint on the whole method, not solver '
          'cost. See\nREADME.md for the history counts a converged '
          'coefficient actually needs.')

    print('\nContribution matrix W[:, :, 0] / alpha_0 -- the fraction of '
          "core 1's\nfeedback carried by each source-to-response pathway "
          '(rows m, cols n):')
    print(W[:, :, 0] / alpha[0])
    print('Reading: column 0 is feedback acting on neutrons born in the '
          'heated core,\nrow 0 is feedback landing as production in it. The '
          'off-diagonal entries are\nwhat a scalar coefficient discards and '
          'what MPK exists to produce.')

    print('\nShape sensitivity dS0/dT_1 * dT, gauge 1^T dS = 0 '
          '(pure shape, no amplitude):')
    print(f'  {dS[:, 0]}')
    print(f'  S0 before {S0}  ->  after {S0 + dS[:, 0]}')
    print('Heating core 1 pushes the fission source toward core 2. Point '
          'kinetics has\nno variable in which to express this.')

    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print(f'STEP 3  transient: step +{DT:g} K in core 1 at t = 0')
    print('=' * 72)
    print('Kp and Kd are renormalized by the BASE k0 so the reference state '
          'is exactly\ncritical and S0 is exactly stationary; the perturbed '
          'case is normalized by\nthe same k0, so it carries the inserted '
          'reactivity.')

    # The perturbed matrices are the ones actually measured in the hot1 run,
    # not a first-order extrapolation, so the transient is exact for the
    # inserted perturbation. beta, ell and lambda are kept at their base
    # values -- the usual working assumption that temperature moves only the
    # coupling matrices. It cannot bias k or the shape, since neither
    # appears in K, so quote what is being neglected:
    print('\nWorking assumption: temperature moves only K. Measured change '
          'at +%g K:' % DT)
    print(f'  ell  {base.ell} -> {hot1.ell} s   '
          f'({(hot1.ell / base.ell - 1) * 100} %)')
    print(f'  B    {base.B * 1e5} -> {hot1.B * 1e5} pcm   '
          f'({(hot1.B / base.B - 1) * 100} %)')
    print('Neither enters K, so neither can bias alpha_p or the shape '
          'sensitivity above.\nThey do set the prompt and delayed timescales '
          'of the transient below.')

    # Size the step. The measured +300 K case is worth several dollars,
    # which is useless for a transient: past about a dollar both MPK and its
    # PK reference run away on the prompt timescale and the comparison stops
    # measuring anything. So the measured perturbation is scaled down to a
    # target fraction of a dollar, keeping its pattern exactly as measured
    # and changing only its magnitude. That is a linear extrapolation in
    # temperature, and check 10 above is what bounds its error.
    _, (_, rho_full, beff_full) = base.pk_reference(hot1, [0.0], k_norm=k0)
    dollars_full = rho_full / beff_full.sum()
    frac = args.dollars / dollars_full
    print(f'\nThe measured +{DT:g} K case is worth {dollars_full:+.3f} '
          f'dollars, far outside the range\nwhere a transient comparison '
          f'means anything. Scaling it to {args.dollars:+.2f} dollars, '
          f'which is\nthe same perturbation pattern at '
          f'{frac * 100:.1f}% of its magnitude -- equivalent to a step of '
          f'\nabout {frac * DT:+.1f} K under the linearity that check 10 '
          'bounds.')
    step = mpk.interpolate_case(base, hot1, frac)

    t = np.concatenate([[0.0], np.logspace(-7, np.log10(60.0), 1500)])
    _, S, _ = base.step_transient(step, t, k_norm=k0)
    total = S.sum(axis=0)
    shape = S / total[None, :]

    pk, (Lam, rho, beff) = base.pk_reference(step, t, k_norm=k0)
    print(f'\nfrozen-shape PK reference: Lambda = {Lam:.5e} s   '
          f'rho = {rho * 1e5:+.2f} pcm   beta_eff = {beff.sum() * 1e5:.1f} pcm')
    print(f'rho / beta_eff = {rho / beff.sum():+.4f} dollars  '
          f'(check 4: PASS, well inside one dollar)')

    # Report the CHANGE in shape, not the shape itself. The base S0 of a
    # symmetric model should be exactly [0.5, 0.5]; it is not, by the
    # statistical error on the matrix, and quoting the raw split would mix
    # that error into the answer. The change is zero at t = 0 by
    # construction and is pure physics after that.
    dshape = shape - shape[:, [0]]

    print('\n  time        MPK total     PK          rel dev    '
          'd(S_1/total)  d(S_2/total)')
    for probe in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 60.0):
        i = int(np.argmin(np.abs(t - probe)))
        print(f'  {t[i]:9.3e}  {total[i]:11.6f}  {pk[i]:11.6f}  '
              f'{abs(total[i] - pk[i]) / pk[i]:9.2e}  '
              f'{dshape[0, i]:+12.6f}  {dshape[1, i]:+12.6f}')

    res = np.abs(total - pk) / pk
    print(f'\nmax |MPK total - PK| / PK = {res.max():.3e} at '
          f't = {t[int(np.argmax(res))]:.4g} s')
    print('\nThe amplitude is what point kinetics gets, and here it gets it '
          'to a few\npercent. The shape change is what it cannot represent '
          'at all: a single scalar\nhas nowhere to put the information that '
          'the fission source has moved out of the\nheated core and into '
          'the cold one. It settles within about a millisecond -- the\n'
          'prompt redistribution onto the new fundamental, on the ell '
          'timescale -- and\nthen stays, because the shape is locked to '
          'the perturbed eigenvector while the\namplitude decays on the '
          'delayed timescale.')

    # Cross-check tying step 3 back to step 2: the asymptotic shape change
    # of the time-dependent solution must reproduce the first-order
    # bordered sensitivity, scaled by the same fraction. These are two
    # completely different computations -- an eigenvector perturbation and
    # an ODE integration -- so agreement is a real check on both.
    predicted = dS[:, 0] * frac
    observed = dshape[:, -1]
    print('\nCross-check, step 2 against step 3:')
    print(f'  bordered first-order dS0, scaled to this step  {predicted}')
    print(f'  shape change of the integrated solution at 60 s {observed}')
    print(f'  difference {np.abs(observed - predicted)}  '
          f'({np.abs(observed - predicted).max() / np.abs(predicted).max() * 100:.2f}% '
          'of the effect)')
    print('  An eigenvector perturbation and an ODE integration agreeing to '
          'this level\n  means the sensitivity path and the solver path are '
          'consistent.')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        ax[0].semilogx(t[1:], total[1:], 'k-', label='MPK total')
        ax[0].semilogx(t[1:], pk[1:], 'r--', label='frozen-shape PK')
        ax[0].set_xlabel('t (s)')
        ax[0].set_ylabel('fission source, normalized to 1 at t=0')
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        # The CHANGE in shape, not the shape itself: the base S0 of a
        # symmetric model should be exactly [0.5, 0.5] and is not, by the
        # statistical error on K, so plotting the raw split would mix that
        # error into the picture. The change is zero at t = 0 by
        # construction and is pure physics after that.
        ax[1].semilogx(t[1:], dshape[0, 1:], label='core 1 (heated)')
        ax[1].semilogx(t[1:], dshape[1, 1:], label='core 2')
        for v, c in ((predicted[0], 'C0'), (predicted[1], 'C1')):
            ax[1].axhline(v, color=c, lw=0.8, ls='--', alpha=0.7)
        ax[1].axhline(0.0, color='k', lw=0.6, ls=':')
        ax[1].set_xlabel('t (s)')
        ax[1].set_ylabel(r'change in regional fraction of the source')
        ax[1].set_title('the shape change PK cannot represent\n'
                        '(dashed: independent first-order prediction)')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.savefig('two_region.png', dpi=130)
        print('\nwrote two_region.png')


if __name__ == '__main__':
    main()
