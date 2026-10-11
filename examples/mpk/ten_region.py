#!/usr/bin/env python
"""
Ten axial regions: an axially segmented PWR pin cell.

A single BEAVRS-specification fuel pin, radially infinite (reflective pitch
boundaries) and axially finite with vacuum at both ends, with the fuel
divided into ten equal axial segments:

        vacuum
      +--------+  z = H          region 9
      |  fuel  |                     8
      |  clad  |                     7
      |  water |                    ...
      |        |                     1
      +--------+  z = 0          region 0
        vacuum

The active height is chosen so that axial leakage brings k to about 1, which
makes the fission source a cosine-like axial shape rather than a flat one.
That shape is the thing MPK tracks and point kinetics integrates away.

Three properties make this a more interesting test than the two-region case:

  * the regions are strongly coupled to their neighbours and weakly to
    everything else, so K comes out near-banded rather than near-diagonal
  * the regions are physically identical but not equivalent: an end segment
    leaks and a central segment does not, so ell_m and alpha_p both vary
    with axial position even though every segment has the same material
  * mirror symmetry about the mid-plane means the bottom and top zones must
    have equal coefficients, which is a free consistency check with no model
    assumption in it -- and the one that reveals when the statistics are
    not good enough

What the script does:

  1. base case: tally the full 10x10 coupling matrix, k_dij, ell_m, raw beta
     and lambda, and run every applicable validation check. Prints the
     banded structure of K and the axial profiles of S0 and ell_m.
  2. perturbation: heats three axial zones one at a time, plus one uniform
     case, and builds Gamma = dK/dT_p on that zone basis. Reports alpha per
     zone with a propagated uncertainty, checks it against mirror symmetry
     and against the uniform case, and measures how local Gamma actually is.
     Zones rather than individual segments because of statistics -- see the
     comment on ZONES below, which is the single most practically important
     thing in this example.
  3. transient: applies a realistic axial coolant temperature profile using
     Gamma alone, with no further transport calculation, and integrates the
     kinetics against a frozen-shape point-kinetics reference. The axial
     source shape tilts away from the hot end; PK cannot represent that.

Five OpenMC runs, cached in the working directory and reused on re-run.
Expect of order ten minutes at the default statistics.

    python ten_region.py
    python ten_region.py --particles 60000 --batches 400   # converged-ish
    python ten_region.py --plot
"""
import argparse
from pathlib import Path

import numpy as np
import openmc

import mpk

N = 10              # axial fuel segments
T_BASE = 600.0      # reference temperature of everything [K]
DT = 300.0          # perturbation for the sensitivity tabulation [K]
HEIGHT = 56.0       # active height [cm], tuned so k ~ 1 with axial leakage
PITCH = 1.26        # BEAVRS pin pitch [cm]
FUEL_R = 0.39218
CLAD_R = 0.45720

# Perturbation basis: axial ZONES, not individual segments.
#
# The kinetics keeps all ten regions -- that is the point of the example --
# but the temperature sensitivity is tabulated on three coarser zones. The
# reason is purely statistical. A temperature coefficient is a small
# difference of two large numbers, and heating one segment of ten inserts
# only about 200 pcm, which at any statistics this example can afford is
# comparable to the noise on k. Measured per segment at 1e6 histories per
# case, alpha_p came out with 100% relative uncertainty and alternating
# signs -- and the symmetry check caught it: alpha_0 and alpha_9, which must
# be equal, differed by more than either one's value.
#
# Grouping into three zones triples the signal per case and cuts the number
# of cases from eight to five, so each coefficient is roughly five times
# better determined. This is the theory manual's own advice for the regime
# where statistics bind: perturb radial rings or axial levels rather than
# every cell.
ZONES = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
ZONE_NAMES = ['bottom', 'middle', 'top']


def build_model(temps, particles, batches):
    """Pin cell with the ten fuel segments at temperatures ``temps`` [K]."""
    openmc.reset_auto_ids()

    # BEAVRS beginning-of-cycle specifications, as in openmc.examples
    fuel_proto = openmc.Material(name='UO2 (2.4%)')
    fuel_proto.set_density('g/cm3', 10.29769)
    fuel_proto.add_nuclide('U234', 4.4843e-6)
    fuel_proto.add_nuclide('U235', 5.5815e-4)
    fuel_proto.add_nuclide('U238', 2.2408e-2)
    fuel_proto.add_nuclide('O16', 4.5829e-2)

    clad = openmc.Material(name='Zircaloy')
    clad.set_density('g/cm3', 6.55)
    clad.add_nuclide('Zr90', 2.1827e-2)
    clad.add_nuclide('Zr91', 4.7600e-3)
    clad.add_nuclide('Zr92', 7.2758e-3)
    clad.add_nuclide('Zr94', 7.3734e-3)
    clad.add_nuclide('Zr96', 1.1879e-3)
    clad.temperature = T_BASE

    water = openmc.Material(name='Borated water')
    water.set_density('g/cm3', 0.740582)
    water.add_nuclide('H1', 4.9457e-2)
    water.add_nuclide('O16', 2.4672e-2)
    water.add_nuclide('B10', 8.0042e-6)
    water.add_nuclide('B11', 3.2218e-5)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = T_BASE

    # One material instance per segment so each can carry its own
    # temperature while being physically identical to the others
    fuels = []
    for i, T in enumerate(temps):
        m = fuel_proto.clone()
        m.name = f'fuel segment {i}'
        m.temperature = float(T)
        fuels.append(m)

    fuel_or = openmc.ZCylinder(r=FUEL_R)
    clad_or = openmc.ZCylinder(r=CLAD_R)
    half = PITCH / 2.0
    xl = openmc.XPlane(x0=-half, boundary_type='reflective')
    xr = openmc.XPlane(x0=half, boundary_type='reflective')
    yb = openmc.YPlane(y0=-half, boundary_type='reflective')
    yt = openmc.YPlane(y0=half, boundary_type='reflective')
    zb = openmc.ZPlane(z0=0.0, boundary_type='vacuum')
    zt = openmc.ZPlane(z0=HEIGHT, boundary_type='vacuum')
    box = +xl & -xr & +yb & -yt & +zb & -zt

    # Axial cuts. Only the fuel is segmented: the clad and the water are
    # not fissile, so they carry no MPK region index.
    zs = [zb] + [openmc.ZPlane(z0=HEIGHT * (i + 1) / N) for i in range(N - 1)] \
        + [zt]
    fuel_cells = [
        openmc.Cell(cell_id=10 + i, name=f'fuel {i}', fill=fuels[i],
                    region=-fuel_or & +zs[i] & -zs[i + 1])
        for i in range(N)]
    cells = fuel_cells + [
        openmc.Cell(cell_id=1, name='clad', fill=clad,
                    region=+fuel_or & -clad_or & box),
        openmc.Cell(cell_id=2, name='water', fill=water,
                    region=+clad_or & box),
    ]

    settings = openmc.Settings(run_mode='eigenvalue')
    settings.particles = particles
    settings.batches = batches
    settings.inactive = max(20, batches // 10)
    settings.seed = 1
    settings.temperature = {'method': 'interpolation',
                            'range': (290.0, 1200.0)}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box([-half, -half, 0.0], [half, half, HEIGHT]),
        constraints={'fissionable': True})

    mpk.configure_settings(settings, fuel_cells)
    tallies = mpk.add_tallies(openmc.Tallies(), fuel_cells)

    model = openmc.Model(
        materials=openmc.Materials(fuels + [clad, water]),
        geometry=openmc.Geometry(cells), settings=settings, tallies=tallies)
    return model, [c.id for c in fuel_cells]


def run_case(tag, temps, particles, batches, workdir):
    d = workdir / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = d / f'statepoint.{batches}.h5'
    model, cell_ids = build_model(temps, particles, batches)
    if not sp.exists():
        print(f'  running {tag} ...', flush=True)
        model.run(cwd=d, output=False)
    else:
        print(f'  reusing {tag}')
    return mpk.from_statepoint(sp, cell_ids)


def bar(x, width=38):
    """A crude horizontal bar, so an axial profile reads at a glance."""
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo if hi > lo else 1.0
    return [('#' * max(1, int(round(width * (v - lo) / span)))) for v in x]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--particles', type=int, default=15000)
    ap.add_argument('--batches', type=int, default=200)
    ap.add_argument('--workdir', type=Path, default=Path('ten_region_work'))
    ap.add_argument('--dollars', type=float, default=-0.30)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()
    np.set_printoptions(precision=4, linewidth=120, suppress=True)

    base_temps = np.full(N, T_BASE)

    # ------------------------------------------------------------------
    print('=' * 72)
    print('STEP 1  base case: tally the MPK coefficients')
    print('=' * 72)
    base = run_case('base', base_temps, args.particles, args.batches,
                    args.workdir)
    base.report()
    k0, S0, Sadj = base.fundamental()
    K = base.total_matrix()

    # Is the delayed matrix converged? At N regions and D families there are
    # N^2 * D delayed coefficients and the far off-diagonal ones are the
    # sparsest scores in the problem. Rebuilding them under a separable
    # importance-ratio assumption gives an independent estimate of the same
    # beta_eff: if the two agree, the sparsity is not biting; if they do
    # not, the delayed data is not converged and no amount of postprocessing
    # will fix it.
    sep, ratios = mpk.separable_delayed(base, S0, Sadj)
    b_raw = base.point_kinetics(S0, Sadj)[2].sum()
    b_sep = sep.point_kinetics(S0, Sadj)[2].sum()
    disagree = abs(b_sep / b_raw - 1.0)
    print(f'\nDelayed-matrix convergence test:')
    print(f'  beta_eff from the tallied k^(i)_mn       {b_raw * 1e5:7.1f} pcm')
    print(f'  beta_eff under a separable importance    {b_sep * 1e5:7.1f} pcm')
    print(f'  disagreement                             {disagree:7.1%}')
    print(f'  delayed importance ratio                 '
          f'{base.importance_ratio(S0, Sadj):7.4f}')
    print(f'  per-family importance ratios fitted on the populated elements:')
    print(f'    {ratios}')
    if disagree > 0.05:
        print('  >>> ABOVE 5%: the delayed coefficients are NOT converged at '
              'this statistics.\n  >>> k_ij, k^(p)_mn, ell_m and alpha_p are '
              'all fine -- they are orders of\n  >>> magnitude better '
              'converged -- but the delayed TIME response of the\n  >>> '
              'transient in step 3 is not quantitative. Re-run with\n'
              '  >>> --particles 20000 --batches 220 for a usable delayed '
              'matrix.')
    else:
        print('  Below 5%: the delayed sparsity is not biasing beta_eff '
              'appreciably.')

    print('\nCoupling matrix K[m, n], response m down, source n across:')
    print(K)
    # How much of each column stays within one segment of where it started.
    band = sum(np.trace(K, offset=o) for o in (-1, 0, 1)) / K.sum()
    print(f'\nFraction of K within one segment of the diagonal: {band:.3f}')
    print('The axial migration length is comparable to one segment, so '
          'coupling is\nlocal and K is near-banded. That is the structure a '
          'many-region MPK model\nhas to exploit to be affordable.')

    print('\nAxial profiles (segment 0 at the bottom):')
    print(f'{"seg":>4} {"S0":>9} {"ell (s)":>11} {"B (pcm)":>9}  profile of S0')
    for i, b in enumerate(bar(S0)):
        print(f'{i:>4} {S0[i]:9.5f} {base.ell[i]:11.4e} '
              f'{base.B[i] * 1e5:9.2f}  {b}')
    print('\nEvery segment has identical material, so the variation is '
          'purely geometric:\nthe cosine-like source shape comes from axial '
          'leakage, and ell_m is shorter\nnear the ends because a neutron '
          'there is more likely to leak than to live long\nenough to cause '
          'a late fission.')

    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print(f'STEP 2  reactivity from a {DT:g} K rise, segment by segment')
    print('=' * 72)
    cases = []
    for zone, name in zip(ZONES, ZONE_NAMES):
        temps = base_temps.copy()
        temps[zone] += DT
        cases.append(run_case(f'hot_{name}', temps, args.particles,
                              args.batches, args.workdir))
    uniform = run_case('uniform', base_temps + DT, args.particles,
                       args.batches, args.workdir)

    print(f'\nPerturbation basis: {len(ZONES)} axial zones, not 10 segments.')
    for zone, name in zip(ZONES, ZONE_NAMES):
        print(f'  {name:7s} = segments {zone}')
    print('Why: heating one segment of ten is a ~200 pcm signal, which at '
          'this statistics\nis comparable to the noise. Grouping triples the '
          'signal and cuts the number\nof cases, so each coefficient is '
          'several times better determined. Gamma then\nhas shape '
          f'(N, N, {len(ZONES)}) = (10, 10, {len(ZONES)}): the kinetics still '
          'carries all ten\nregions, only the sensitivity is coarser.')

    alpha, W, dS, G, sig = mpk.check_first_order(
        base, cases, DT, uniform=uniform)

    # Mirror symmetry: the bottom and top zones are the same perturbation
    # reflected, so their coefficients must agree. No model assumption in
    # it, so the difference is a clean measurement of the statistical error.
    spread = abs(alpha[0] - alpha[-1])
    print(f'\nMirror symmetry check: the bottom and top zones are the same '
          'perturbation\nreflected, so alpha must agree between them.')
    print(f'  alpha(bottom) = {alpha[0] * 1e5:+.4f} pcm/K     '
          f'alpha(top) = {alpha[-1] * 1e5:+.4f} pcm/K     '
          f'difference {spread * 1e5:.4f} pcm/K')
    if sig is not None:
        exp = float(np.hypot(sig[0], sig[-1]))
        print(f'  expected spread from the propagated sigma '
              f'{exp * 1e5:.4f} pcm/K, so {spread / exp:.1f} sigma')
        print('  No model assumption enters this, so it is the most '
              'trustworthy error bar\n  available on the coefficient.')

    print(f'\n{"zone":>8} {"alpha (pcm/K)":>14} {"1 sigma":>9} '
          f'{"rel":>7}  segments')
    for i, name in enumerate(ZONE_NAMES):
        s_i = f'{sig[i] * 1e5:9.4f}' if sig is not None else ' ' * 9
        rel = f'{abs(sig[i] / alpha[i]):7.1%}' if sig is not None else ''
        print(f'{name:>8} {alpha[i] * 1e5:14.4f} {s_i} {rel}  {ZONES[i]}')
    print(f'\nSum over zones {alpha.sum() * 1e5:+.4f} pcm/K. The middle zone '
          'carries the most\nfeedback per kelvin because it holds four of '
          'the ten segments and sits where\nthe adjoint-weighted source is '
          'largest -- a change in the coupling matrix has\nthe most leverage '
          'on the eigenvalue there. A single bulk coefficient hides\nthat '
          'distribution entirely, and it is the distribution a coupled '
          'thermal-\nhydraulic calculation needs.')

    # Locality: for a zone the question is whether its sensitivity stays in
    # the rows and columns of its own member segments.
    print('\nLocality of Gamma -- the fraction of |dK/dT_p| in the rows and '
          'columns of\nthe perturbed zone\'s own segments:')
    for i, name in enumerate(ZONE_NAMES):
        A = np.abs(G[:, :, i])
        mask = np.zeros_like(A, dtype=bool)
        mask[ZONES[i], :] = True
        mask[:, ZONES[i]] = True
        frac = A[mask].sum() / A.sum() if A.sum() > 0 else 0.0
        share = mask.sum() / mask.size
        print(f'  {name:7s} {frac:.3f}   (those rows and columns are '
              f'{share:.0%} of the matrix)')
    print('Compare each fraction against the share of the matrix it covers. '
          'Well above\nthat share means the sensitivity is concentrated '
          'where the heat was applied,\nwhich is what would let a '
          'many-region model store Gamma sparsely. Close to\nit means '
          'Gamma is spread out -- or that noise is, which at this '
          'statistics is\nthe more likely reading. This is the open '
          'question that decides whether the\nmethod scales past a handful '
          'of regions, and it needs a converged Gamma to\nanswer properly.')

    print(f'\nContribution matrix for the middle zone, W[:, :, 1] / '
          f'alpha(middle), rows m,\ncolumns n -- which source-to-response '
          'pathways carry its feedback:')
    print(W[:, :, 1] / alpha[1])

    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print('STEP 3  transient under an axial coolant temperature profile')
    print('=' * 72)
    # A linear axial rise, as coolant heats on its way up the channel. This
    # costs no further transport calculation: Gamma already spans it.
    # Coolant entering cold at the bottom and leaving hot at the top. The
    # ramp is normalized so the HOTTEST segment sits exactly at the measured
    # +DT and every other at a fraction of it. That keeps the combination an
    # interpolation between runs that were actually done: each weight is in
    # [0, 1], so the result is a convex combination of non-negative matrices
    # and stays non-negative. Normalizing to a mean of DT instead would put
    # the top segment at 2*DT, extrapolating a linear fit to twice the
    # measured perturbation -- which drives the noisiest near-zero elements
    # of K negative and is caught by MPKParams.validate().
    # Coolant entering cold at the bottom and leaving hot at the top,
    # expressed in the zone basis the sensitivity was tabulated on. The
    # weights are fractions of the measured DT and stay inside [0, 1], so
    # this interpolates between runs that were actually done rather than
    # extrapolating past them.
    profile = np.array([0.25, 0.60, 1.00])
    print('\nAxial coolant temperature rise, as a fraction of the measured '
          f'{DT:g} K perturbation:')
    for name, f in zip(ZONE_NAMES, profile):
        print(f'  {name:7s} {f:.2f}  ->  +{f * DT:.0f} K')
    print('Evaluated from Gamma alone: the five runs above already span this '
          'profile, so\na different one -- a hot spot, a flat rise, a '
          'reversed tilt -- costs a matrix\ncombination rather than another '
          'transport calculation. That is the point of\ntabulating the '
          'sensitivity.')

    probe = mpk.combine_cases(base, cases, DT, profile * DT)
    _, (_, rho_full, beff_full) = base.pk_reference(probe, [0.0], k_norm=k0)
    dollars_full = rho_full / beff_full.sum()
    frac = args.dollars / dollars_full
    deltas = profile * DT * frac
    print(f'\nAt full strength this profile is worth {dollars_full:+.3f} '
          f'dollars, outside the range\nwhere a transient comparison means '
          f'anything. Scaling to {args.dollars:+.2f} dollars gives a\nrise '
          f'of {deltas[0]:+.1f} K at the bottom to {deltas[-1]:+.1f} K at '
          'the top.')

    step = mpk.combine_cases(base, cases, DT, deltas)
    t = np.concatenate([[0.0], np.logspace(-7, np.log10(60.0), 1500)])
    _, S, _ = base.step_transient(step, t, k_norm=k0)
    total = S.sum(axis=0)
    shape = S / total[None, :]
    dshape = shape - shape[:, [0]]

    pk, (Lam, rho, beff) = base.pk_reference(step, t, k_norm=k0)
    print(f'\nfrozen-shape PK reference: Lambda = {Lam:.5e} s   '
          f'rho = {rho * 1e5:+.2f} pcm   beta_eff = {beff.sum() * 1e5:.1f} pcm')
    print(f'rho / beta_eff = {rho / beff.sum():+.4f} dollars  (check 4)')

    print('\n  time        MPK total     PK          rel dev   max |d shape|')
    for q in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 60.0):
        i = int(np.argmin(np.abs(t - q)))
        print(f'  {t[i]:9.3e}  {total[i]:11.6f}  {pk[i]:11.6f}  '
              f'{abs(total[i] - pk[i]) / pk[i]:9.2e}  '
              f'{np.abs(dshape[:, i]).max():13.3e}')

    res = np.abs(total - pk) / pk
    print(f'\nmax |MPK total - PK| / PK = {res.max():.3e} at '
          f't = {t[int(np.argmax(res))]:.4g} s')

    print('\nAxial shape change, start to finish:')
    print(f'{"seg":>4} {"S0":>9} {"final":>9} {"change":>10}  '
          'direction of the shift')
    for i in range(N):
        d = dshape[i, -1]
        arrow = ('<' * max(1, int(round(abs(d) * 4000)))) if d < 0 else \
                ('>' * max(1, int(round(abs(d) * 4000))))
        print(f'{i:>4} {shape[i, 0]:9.5f} {shape[i, -1]:9.5f} '
              f'{d:+10.6f}  {arrow}')
    print('\nThe top of the pin is hottest, so it loses the most reactivity '
          'and the\nfission source shifts downward. The total is what point '
          'kinetics reports;\nthe redistribution is invisible to it, and it '
          'is exactly what a coupled\nthermal-hydraulic calculation needs '
          'back from the kinetics.')

    # Cross-check: the asymptotic shape change of the integrated solution
    # against the first-order bordered sensitivity for the same profile.
    dK = sum((deltas[p] / DT) * (cases[p].total_matrix() - K)
             for p in range(len(cases)))
    _, dS_pred = mpk.bordered_sensitivity(K, dK, k0, S0)
    print('\nCross-check, sensitivity path against solver path:')
    print(f'  bordered first-order dS0  {dS_pred}')
    print(f'  integrated shape change   {dshape[:, -1]}')
    denom = np.abs(dS_pred).max()
    print(f'  max difference {np.abs(dshape[:, -1] - dS_pred).max():.3e} '
          f'({np.abs(dshape[:, -1] - dS_pred).max() / denom * 100:.2f}% '
          'of the effect)')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        z = (np.arange(N) + 0.5) * HEIGHT / N
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.2),
                               constrained_layout=True)
        ax[0].plot(S0, z, 'o-')
        ax[0].set_xlabel('$S_0$ (fraction of fission source)')
        ax[0].set_ylabel('z (cm)')
        ax[0].set_title('base axial source shape')
        ax[0].grid(alpha=0.3)
        # alpha is per ZONE, so draw it as a step across each zone's extent,
        # with the propagated uncertainty, rather than as a per-segment curve
        for i, segs in enumerate(ZONES):
            z0 = segs[0] * HEIGHT / N
            z1 = (segs[-1] + 1) * HEIGHT / N
            ax[1].plot([alpha[i] * 1e5] * 2, [z0, z1], lw=3,
                       label=ZONE_NAMES[i])
            if sig is not None:
                ax[1].fill_betweenx(
                    [z0, z1], (alpha[i] - sig[i]) * 1e5,
                    (alpha[i] + sig[i]) * 1e5, alpha=0.2)
        ax[1].axvline(0.0, color='k', lw=0.6, ls=':')
        ax[1].set_xlabel(r'$\alpha$ (pcm/K)')
        ax[1].set_title(r'zone temperature coefficient, $\pm1\sigma$')
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        ax[2].semilogx(t[1:], total[1:], 'k-', lw=2, label='MPK total')
        ax[2].semilogx(t[1:], pk[1:], 'r--', label='frozen-shape PK')
        ax[2].set_xlabel('t (s)')
        ax[2].set_ylabel('fission source')
        ax[2].set_title('amplitude: the part PK gets right')
        ax[2].legend()
        ax[2].grid(alpha=0.3)
        # The part PK cannot represent at all, plus the independent
        # first-order prediction of the same quantity
        ax[3].plot(dshape[:, -1], z, 'o-', label='integrated, t = 60 s')
        ax[3].plot(dS_pred, z, 'x--', label='bordered, first order')
        ax[3].axvline(0.0, color='k', lw=0.6, ls=':')
        ax[3].set_xlabel(r'change in $S_m$ / total')
        ax[3].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax[3].locator_params(axis='x', nbins=5)
        ax[3].set_title('shape: the part PK cannot represent')
        ax[3].legend()
        ax[3].grid(alpha=0.3)
        fig.savefig('ten_region.png', dpi=130)
        print('\nwrote ten_region.png')


if __name__ == '__main__':
    main()
