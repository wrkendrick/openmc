#!/usr/bin/env python
"""
Self-consistency checks for mpk.py. No OpenMC run needed.

Everything here is exact algebra on synthetic parameters, so any failure is a
regression in the module rather than a statistical fluctuation. Run it after
editing mpk.py:

    python selftest.py

The numbered validation checks against real tallies live in the two example
scripts; the synthetic invariants of the formulation itself are locked in
tests/test_mpk_invariants.py at the top of the repository.
"""
import numpy as np

import mpk

D = 6
LAM = np.log(2) / np.array([55.72, 22.72, 6.22, 2.30, 0.610, 0.230])
BOOST = 1.05        # uniform delayed importance built into the synthetic set


def synth(N, rng, scale=None):
    """A plausible N-region parameter set with a known delayed importance."""
    x = np.arange(N)
    Kp = np.exp(-np.abs(x[:, None] - x[None, :]) / 1.8)
    Kp = Kp * (1.0 + 0.02 * rng.standard_normal((N, N)))
    Kp = Kp / np.abs(np.linalg.eigvals(Kp)).max()
    if scale is not None:
        Kp = Kp * scale[None, :]
    Kd = np.stack([Kp * BOOST for _ in range(D)])
    w = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
    w = w / w.sum()
    beta = np.outer(7.0e-3 * np.ones(N), w)
    p = mpk.MPKParams(Kp=Kp, Kd=Kd, beta=beta, ell=2.0e-5 * np.ones(N),
                      lam=LAM)
    p.kij = p.total_matrix()
    p.kij_sd = np.full((N, N), 1.0e-4)
    return p


def main():
    rng = np.random.default_rng(3)
    t = np.concatenate([[0.0], np.logspace(-9, 1, 200)])

    for N in (1, 2, 4, 10):
        p = synth(N, rng)
        k0, S0, Sadj = p.fundamental()
        cases = [synth(N, rng, 1.0 - 0.01 * np.eye(N)[q]) for q in range(N)]

        # combine_cases with no temperature change must be the identity, and
        # with one full-strength change must reproduce that case exactly
        assert np.array_equal(
            mpk.combine_cases(p, cases, 300.0, np.zeros(N)).total_matrix(),
            p.total_matrix())
        # combine_cases works on the total matrix, so the invariant is that
        # it reproduces that total -- not that it reproduces the case's own
        # Kp, which it redistributes
        assert np.allclose(
            mpk.combine_cases(p, cases, 300.0,
                              300.0 * np.eye(N)[0]).total_matrix(),
            cases[0].total_matrix())
        # a combination must stay non-negative for weights inside [0, 1]
        full = mpk.combine_cases(p, cases, 300.0, 300.0 * np.ones(N))
        assert np.all(full.Kp >= 0.0) and np.all(full.Kd >= 0.0)

        # interpolate_case must hit both endpoints
        assert np.allclose(mpk.interpolate_case(p, cases[0], 0.0).Kp, p.Kp)
        assert np.allclose(mpk.interpolate_case(p, cases[0], 1.0).Kp,
                           cases[0].Kp)

        # check 8: started from the fundamental of the assembled matrix,
        # renormalized to criticality, with no perturbation, nothing moves
        _, S, _ = p.step_transient(p, t, k_norm=k0)
        drift = np.max(np.abs(S / S0[:, None] - 1.0))
        assert drift < 1e-11, f'N={N} stationarity {drift:.2e}'

        # an unperturbed PK reference must have exactly zero reactivity
        pk, (_, rho, _) = p.pk_reference(p, t, k_norm=k0)
        assert abs(rho) < 1e-12, f'N={N} rho {rho:.2e}'
        pk_drift = np.max(np.abs(pk - 1.0))
        assert pk_drift < 1e-11, f'N={N} PK drift {pk_drift:.2e}'

        # the delayed importance ratio is measured against the TOTAL matrix,
        # so a uniform Kd = BOOST * Kp reads BOOST / (1 + (BOOST-1) B)
        ir = p.importance_ratio(S0, Sadj)
        expect = BOOST / (1.0 + (BOOST - 1.0) * p.B.mean())
        assert abs(ir - expect) < 1e-9, f'N={N} {ir} vs {expect}'

        # a uniform importance must come back as a uniform fitted ratio, and
        # the separable rebuild must not change the collapsed beta_eff
        sep, r = mpk.separable_delayed(p, S0, Sadj)
        assert np.ptp(r) < 1e-12, f'N={N} r_i spread {np.ptp(r):.1e}'
        assert abs(sep.importance_ratio(S0, Sadj) - ir) < 1e-9

        print(f'  N={N:2d}  stationarity {drift:.1e}  PK drift '
              f'{pk_drift:.1e}  importance {ir:.8f}  r_i spread '
              f'{np.ptp(r):.1e}')

    # shape validation must reject the orientation mistakes that matter
    p = synth(4, rng)
    for bad, why in (
            (dict(beta=p.beta.T), 'beta transposed to (D, N)'),
            (dict(Kd=np.moveaxis(p.Kd, 0, 2)), 'Kd left as (N, N, D)')):
        kwargs = dict(Kp=p.Kp, Kd=p.Kd, beta=p.beta, ell=p.ell, lam=p.lam)
        kwargs.update(bad)
        try:
            mpk.MPKParams(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'validate() accepted {why}')

    # and an i/j transposition, which no eigenvalue check can see
    k_fwd = p.fundamental()[0]
    flipped = mpk.MPKParams(Kp=p.Kp.T, Kd=np.moveaxis(p.Kd, 1, 2),
                            beta=p.beta, ell=p.ell, lam=p.lam)
    assert abs(flipped.fundamental()[0] - k_fwd) < 1e-12, \
        'a transposed K should keep its eigenvalue; if not, this test is void'
    try:
        mpk.MPKParams(Kp=p.Kp.T, Kd=np.moveaxis(p.Kd, 1, 2), beta=p.beta,
                      ell=p.ell, lam=p.lam, kij=p.kij, kij_sd=p.kij_sd)
    except ValueError:
        print('  i/j transposition rejected by the partition identity, '
              'as it must be')
    else:
        raise AssertionError('validate() accepted a transposed K')

    print('\nmpk.py self-consistency: all checks passed')


if __name__ == '__main__':
    main()
