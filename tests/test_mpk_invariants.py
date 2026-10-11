"""
MPK invariants, locked as tests. No OpenMC required: everything here runs on synthetic
parameters and checks algebra, orientation and solver behaviour.

Each test corresponds to a numbered check in docs/mpk_theory_manual.md, Section "Validation
hierarchy". Run after any change to parameter assembly, the MPK right-hand side, or the
sensitivity path.

    pytest tests/test_mpk_invariants.py -v
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

D = 6
LAM = np.log(2) / np.array([55.72, 22.72, 6.22, 2.30, 0.610, 0.230])
WEIGHTS = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
WEIGHTS = WEIGHTS / WEIGHTS.sum()


# ---------------------------------------------------------------------------
# helpers
def synth(N, seed=3, asym=0.08):
    """A plausible N-region parameter set. Kp normalized so k is near unity."""
    rng = np.random.default_rng(seed)
    x = np.arange(N)
    Kp = np.exp(-np.abs(x[:, None] - x[None, :]) / 1.8)
    Kp = Kp * (1.0 + 0.02 * rng.standard_normal((N, N)))
    Kp = Kp / np.abs(np.linalg.eigvals(Kp)).max()
    Kd = np.stack([Kp * (1.0 + asym + 0.01 * rng.standard_normal((N, N))) for _ in range(D)])
    beta = np.outer(7.0e-3 * (1.0 + 0.05 * rng.standard_normal(N)), WEIGHTS)
    ell = 2.0e-5 * (1.0 + 0.1 * rng.standard_normal(N))
    return Kp, Kd, beta, ell


def assemble(Kp, Kd, beta):
    """K_mn = (1-B_n) k^(p)_mn + sum_i beta_n^(i) k^(i)_mn. beta is (N, D), source-indexed."""
    B = beta.sum(axis=1)
    return (1.0 - B)[None, :] * Kp + np.einsum('ni,imn->mn', beta, Kd)


def fundamental(K):
    """Right and left fundamental eigenvectors, sum(S0)=1 and <Sadj, S0>=1."""
    w, V = np.linalg.eig(K)
    j = np.argmax(w.real)
    S0 = np.abs(V[:, j].real)
    wL, VL = np.linalg.eig(K.T)
    Sadj = np.abs(VL[:, np.argmax(wL.real)].real)
    S0 = S0 / S0.sum()
    return w[j].real, S0, Sadj / (Sadj @ S0)


def rhs(Kp, Kd, beta, ell, column_scaling=True):
    """MPK right-hand side. column_scaling=False reproduces the row-scaling bug."""
    N = ell.size
    B = beta.sum(axis=1)

    def f(t, y):
        S = y[:N]
        C = y[N:].reshape(D, N)
        delayed = np.einsum('i,imn,in->m', LAM, Kd, C)
        if column_scaling:
            prompt = Kp @ ((1.0 - B) * S)
        else:
            prompt = (1.0 - B) * (Kp @ S)
        dS = (prompt - S + delayed) / ell
        dC = beta.T * S[None, :] - LAM[:, None] * C
        return np.concatenate([dS, dC.ravel()])

    return f


def bordered(K0, dK):
    """First-order (dk, dS0) under the gauge 1^T dS = 0."""
    k0, S0, _ = fundamental(K0)
    N = K0.shape[0]
    M = np.zeros((N + 1, N + 1))
    M[:N, :N] = K0 - k0 * np.eye(N)
    M[:N, N] = -S0
    M[N, :N] = 1.0
    sol = np.linalg.solve(M, np.concatenate([-dK @ S0, [0.0]]))
    return sol[N], sol[:N]


def collapse(Kp, Kd, beta, ell, S0, Sadj):
    B = beta.sum(axis=1)
    P = (1.0 - B)[None, :] * Kp
    ell_eff = (Sadj * ell) @ S0 / (Sadj @ S0)
    p_eff = Sadj @ (P @ S0)
    btil = np.array([Sadj @ (Kd[i] @ (beta[:, i] * S0)) for i in range(D)])
    return ell_eff, p_eff, btil


# --------------------------------------------------------------- check 1:
# partition
@pytest.mark.parametrize("N", [1, 2, 4, 8])
def test_partition_identity_roundtrip(N):
    """Assembling K from the parts and decomposing it again must be exact."""
    Kp, Kd, beta, _ = synth(N)
    K = assemble(Kp, Kd, beta)
    delayed = np.einsum('ni,imn->mn', beta, Kd)
    prompt = (1.0 - beta.sum(axis=1))[None, :] * Kp
    assert np.allclose(K, prompt + delayed, rtol=0, atol=1e-15)


def test_partition_weights_sum_to_one():
    _, _, beta, _ = synth(4)
    B = beta.sum(axis=1)
    assert np.allclose((1.0 - B) + B, 1.0, atol=1e-14)


def test_transposition_is_detected_by_partition_not_by_eigenvalue():
    """A transposed K keeps its fundamental eigenvalue; only the partition catches it."""
    Kp, Kd, beta, _ = synth(4)
    K = assemble(Kp, Kd, beta)
    k_fwd, _, _ = fundamental(K)
    k_rev, _, _ = fundamental(K.T)
    assert abs(k_fwd - k_rev) < 1e-12                      # eigenvalue check is blind
    bad = assemble(Kp.T, np.moveaxis(Kd, 1, 2), beta)      # i/j swapped at extraction
    assert np.abs(bad - K).max() > 1e-4                    # partition check is not


def test_mixed_factorization_is_wrong_by_the_predicted_amount():
    """(1-B)*k_mn differs from the correct prompt term by B*(kbar_d - k_mn)."""
    Kp, Kd, beta, _ = synth(1)
    K = assemble(Kp, Kd, beta)
    B = beta.sum(axis=1)
    correct = (1.0 - B)[None, :] * Kp
    mixed = (1.0 - B)[None, :] * K
    kbar_d = np.einsum('ni,imn->mn', beta, Kd) / B[None, :]
    predicted = B[None, :] * (kbar_d - K)
    assert np.allclose(mixed - correct, predicted, rtol=1e-10)


# ------------------------------------------------- check 6: N=1 reduction to
# point kinetics
def test_single_region_mpk_equals_pk():
    Kp, Kd, beta, ell = synth(1)
    K = assemble(Kp, Kd, beta)
    k = K[0, 0]
    kp, kd, b, L = Kp[0, 0], Kd[:, 0, 0], beta[0], ell[0]

    Lambda = L / k
    rho = (k - 1.0) / k
    beta_eff = (kd / k) * b

    def mpk(t, y):
        S, C = y[0], y[1:]
        return np.concatenate([[((1 - b.sum()) * kp - 1.0) * S / L + (LAM * kd) @ C / L],
                               b * S - LAM * C])

    def pk(t, y):
        S, C = y[0], y[1:]
        return np.concatenate([[(rho - beta_eff.sum()) * S / Lambda + LAM @ C],
                               beta_eff * S / Lambda - LAM * C])

    T, kw = 60.0, dict(method='Radau', t_eval=np.linspace(0, 60, 2000),
                       rtol=1e-11, atol=1e-15)
    a = solve_ivp(mpk, (0, T), np.concatenate([[1.0], b / LAM]), **kw)
    c = solve_ivp(pk, (0, T), np.concatenate([[1.0], beta_eff / (Lambda * LAM)]), **kw)
    assert np.max(np.abs(a.y[0] - c.y[0]) / c.y[0]) < 1e-10
    # precursors map as Ctilde = (k^(i)/ell) C
    assert np.max(np.abs((kd[:, None] / L) * a.y[1:] - c.y[1:]) / c.y[1:]) < 1e-10


# ----------------------------------------------------- check 7: adjoint
# collapse identity
@pytest.mark.parametrize("N", [2, 4, 8])
def test_collapse_identity(N):
    """p_eff + sum(btil) == k, since both sides are <S+, K S0>."""
    Kp, Kd, beta, ell = synth(N)
    K = assemble(Kp, Kd, beta)
    k, S0, Sadj = fundamental(K)
    _, p_eff, btil = collapse(Kp, Kd, beta, ell, S0, Sadj)
    assert abs(p_eff + btil.sum() - k) < 1e-14


@pytest.mark.parametrize("N", [2, 4, 8])
def test_collapse_identity_collapses_to_the_rayleigh_quotient(N):
    """
    The collapse identity is exact against the RAYLEIGH QUOTIENT of whatever
    matrix is being contracted, not against that matrix's eigenvalue.

    Both sides of  p_eff + sum_i btil^(i) = <S+, K S0> / <S+, S0>  are the
    same projection, so the identity is exact by construction. At the
    unperturbed state S0 and S+ are K's own eigenvectors and the Rayleigh
    quotient equals the eigenvalue, which is why the check reads as
    "p_eff + sum(btil) == k" there. Once a perturbation is applied and the
    shape is held frozen at the base S0/S+ -- which is exactly what the
    frozen-shape PK reference of the multi-region comparison does -- the two
    part company at second order in the perturbation.

    A frozen-shape PK reference must therefore take rho from the Rayleigh
    quotient. Using the perturbed matrix's true eigenvalue instead puts a
    second-order error into the reference, and that error then shows up in
    the MPK-minus-PK residual, where it is indistinguishable from the
    spatial effect the comparison exists to measure.
    """
    Kp, Kd, beta, ell = synth(N)
    K0 = assemble(Kp, Kd, beta)
    k0, S0, Sadj = fundamental(K0)

    # a rod-like column perturbation of the critical-renormalized matrices
    Kp2, Kd2 = Kp / k0, Kd / k0
    Kp2[:, 0] *= 0.995
    Kd2[:, :, 0] *= 0.995
    K2 = assemble(Kp2, Kd2, beta)

    _, p_eff, btil = collapse(Kp2, Kd2, beta, ell, S0, Sadj)
    rayleigh = (Sadj @ (K2 @ S0)) / (Sadj @ S0)
    k2 = fundamental(K2)[0]

    # exact against the Rayleigh quotient
    assert abs(p_eff + btil.sum() - rayleigh) < 1e-14
    # and measurably different from the true eigenvalue, so the two are not
    # interchangeable in the reference
    assert k2 - rayleigh > 1e-9, (
        "the perturbation is too small to separate the Rayleigh quotient "
        "from the eigenvalue; this test is then vacuous")


# ------------------------------------------------------- check 4: reactivity
# scale
@pytest.mark.parametrize("N", [1, 2, 4])
def test_reactivity_stays_inside_one_dollar(N):
    """
    rho/beta_eff must be well inside one dollar for a transient comparison to
    mean anything. With Lambda near 1e-5 s a prompt-supercritical step makes
    both MPK and the PK reference overflow within milliseconds, at which
    point sub-percent parameter inconsistencies are amplified beyond
    recognition and any agreement figure is meaningless. Print rho/beta_eff
    before interpreting any transient result.
    """
    Kp, Kd, beta, ell = synth(N)
    K0 = assemble(Kp, Kd, beta)
    k0, S0, Sadj = fundamental(K0)

    Kp2, Kd2 = Kp / k0, Kd / k0
    Kp2[:, 0] *= 0.995
    Kd2[:, :, 0] *= 0.995
    K2 = assemble(Kp2, Kd2, beta)
    k2 = fundamental(K2)[0]

    _, _, btil = collapse(Kp2, Kd2, beta, ell, S0, Sadj)
    beta_eff = btil.sum() / k2
    rho = (k2 - 1.0) / k2
    assert abs(rho / beta_eff) < 1.0, (
        f"rho/beta_eff = {rho / beta_eff:.3f} dollars is not inside one "
        "dollar; reduce the perturbation before comparing transients")


# ------------------------------------------- ell enters the timescale, not the
# eigenvalue
def test_ell_does_not_affect_k_or_the_shape():
    """
    ell appears nowhere in K, so no error in ell can be caught by the
    partition identity, by the eigenvalue, or by the fundamental shape. It
    sets the prompt timescale only.

    This is why ell needs its own estimator-level validation rather than
    riding on the matrix checks: scaling one region's ell by 100 leaves k
    and S0 bit-identical while completely changing that region's prompt
    response.
    """
    N = 4
    Kp, Kd, beta, ell = synth(N)
    K = assemble(Kp, Kd, beta)
    k_a, S0_a, _ = fundamental(K)

    ell_bad = ell.copy()
    ell_bad[0] *= 100.0
    # K is built from Kp, Kd and beta only -- ell is not an input to it
    k_b, S0_b, _ = fundamental(assemble(Kp, Kd, beta))
    assert k_a == k_b
    assert np.array_equal(S0_a, S0_b)

    # but the stationary decay rate per region does change
    rate_a = (k_a - 1.0) / ell
    rate_b = (k_a - 1.0) / ell_bad
    assert abs(rate_b[0] / rate_a[0] - 0.01) < 1e-12


# ------------------------------------------------------------- check 8:
# stationarity
@pytest.mark.parametrize("N", [2, 4])
def test_fundamental_start_is_stationary(N):
    """Critical-renormalized K, started from S0, must not move."""
    Kp, Kd, beta, ell = synth(N)
    k, S0, _ = fundamental(assemble(Kp, Kd, beta))
    Kp_c, Kd_c = Kp / k, Kd / k
    C0 = (beta.T * S0[None, :]) / LAM[:, None]
    y0 = np.concatenate([S0, C0.ravel()])
    s = solve_ivp(rhs(Kp_c, Kd_c, beta, ell), (0, 10.0), y0, method='Radau',
                  t_eval=np.concatenate([[0.0], np.logspace(-9, 1, 400)]),
                  rtol=1e-11, atol=1e-16)
    assert np.max(np.abs(s.y[:N] / S0[:, None] - 1.0)) < 1e-12


def test_row_scaling_bug_breaks_stationarity():
    """(1-B) applied to the matvec output instead of to S is detectably wrong."""
    N = 4
    Kp, Kd, beta, ell = synth(N)
    k, S0, _ = fundamental(assemble(Kp, Kd, beta))
    Kp_c, Kd_c = Kp / k, Kd / k
    y0 = np.concatenate([S0, ((beta.T * S0[None, :]) / LAM[:, None]).ravel()])
    kw = dict(method='Radau', t_eval=np.concatenate([[0.0], np.logspace(-9, 1, 400)]),
              rtol=1e-11, atol=1e-16)
    good = solve_ivp(rhs(Kp_c, Kd_c, beta, ell, True), (0, 10.0), y0, **kw)
    bad = solve_ivp(rhs(Kp_c, Kd_c, beta, ell, False), (0, 10.0), y0, **kw)
    assert np.max(np.abs(good.y[:N] / S0[:, None] - 1.0)) < 1e-12
    assert np.max(np.abs(bad.y[:N] / S0[:, None] - 1.0)) > 1e-6


# --------------------------------------------- checks 9, 13: gauge and
# Rayleigh agreement
@pytest.mark.parametrize("N", [2, 4, 8])
def test_bordered_gauge_and_rayleigh(N):
    Kp, Kd, beta, _ = synth(N)
    K0 = assemble(Kp, Kd, beta)
    k0, S0, Sadj = fundamental(K0)
    rng = np.random.default_rng(11)
    dK = 1e-3 * K0 * rng.standard_normal((N, N))

    dk, dS = bordered(K0, dK)
    assert abs(dS.sum()) < 1e-15                                   # gauge enforced
    assert abs(dk - Sadj @ dK @ S0 / (Sadj @ S0)) < 1e-15          # same projection


def test_adding_eps_S0_leaves_the_first_order_equation_satisfied():
    """Gauge freedom: dS is defined only up to a multiple of S0."""
    N = 4
    Kp, Kd, beta, _ = synth(N)
    K0 = assemble(Kp, Kd, beta)
    k0, S0, _ = fundamental(K0)
    rng = np.random.default_rng(5)
    dK = 1e-3 * K0 * rng.standard_normal((N, N))
    dk, dS = bordered(K0, dK)
    rhs_vec = dk * S0 - dK @ S0
    for eps in (0.0, 0.01, -0.5, 7.0):
        res = (K0 - k0 * np.eye(N)) @ (dS + eps * S0) - rhs_vec
        assert np.linalg.norm(res) < 1e-12


# --------------------------------------- checks 10, 12: first-order adequacy
# and convergence
def test_shape_prediction_converges_at_second_order():
    """Halving the perturbation must quarter the error, or the residual is a bug."""
    N = 4
    Kp, Kd, beta, _ = synth(N)
    K0 = assemble(Kp, Kd, beta)
    _, S0, _ = fundamental(K0)
    rng = np.random.default_rng(7)
    direction = K0 * rng.standard_normal((N, N))

    errs = []
    for scale in (8e-3, 4e-3, 2e-3, 1e-3):
        dK = scale * direction
        _, dS = bordered(K0, dK)
        _, S0p, _ = fundamental(K0 + dK)
        errs.append(np.max(np.abs(dS - (S0p - S0))))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(3.4 < r < 4.6 for r in ratios), f"not second order: {ratios}"


def test_alpha_additivity():
    """Sum of per-region coefficients reproduces a uniform perturbation."""
    N = 4
    Kp, Kd, beta, _ = synth(N)
    K0 = assemble(Kp, Kd, beta)
    k0, S0, Sadj = fundamental(K0)

    def K_of(scales):                      # column scaling: a source-region perturbation
        return assemble(Kp * scales[None, :], Kd * scales[None, None, :], beta)

    eps = 2e-3
    alpha = np.zeros(N)
    for p in range(N):
        s = np.ones(N)
        s[p] = 1.0 - eps
        dK = K_of(s) - K0
        alpha[p] = (Sadj @ dK @ S0) / (k0 ** 2 * (Sadj @ S0))

    k_uni, _, _ = fundamental(K_of(np.full(N, 1.0 - eps)))
    rho_uni = (1 - 1 / k_uni) - (1 - 1 / k0)
    assert abs(alpha.sum() - rho_uni) / abs(rho_uni) < 0.02


# ----------------------------------------------------- kdij bin convention
# classifier
def test_kdij_convention_classifier():
    """Guard the bin reading: bin 0 ~ kij, delayed bins sum to ~D*kij, never to beta*kij."""
    Kp, Kd, beta, _ = synth(1)
    kij = assemble(Kp, Kd, beta)[0, 0]
    kdij = np.concatenate([[Kp[0, 0]], Kd[:, 0, 0]])

    assert abs(kdij[0] / kij - 1.0) < 0.01, "bin 0 is not k^(p)"
    assert 0.5 * D < kdij[1:].sum() / kij < 2.0 * D, "delayed bins are not bare k^(i)"
    assert kdij.sum() / kij > 2.0, "array looks like a partition, not coupling coefficients"


def test_einsum_orientation_against_hand_computed_values():
    """
    Pin the index semantics with numbers checkable by eye. With
        K[m,n] = (1-B_n) Kp[m,n] + sum_i beta[n,i] Kd[i,m,n]
    and beta indexed by the SOURCE region n:

        K[0,1] = 0.93*0.3 + 0.03*2.0 + 0.04*6.0 = 0.579
        K[1,0] = 0.97*0.4 + 0.01*3.0 + 0.02*7.0 = 0.558

    Reading beta as (D, N) instead of (N, D) changes both and is silent whenever N == D.
    """
    Kp = np.array([[0.6, 0.3], [0.4, 0.7]])
    Kd = np.array([[[1.0, 2.0], [3.0, 4.0]],
                   [[5.0, 6.0], [7.0, 8.0]]])
    beta = np.array([[0.01, 0.02],        # source region 0, families 0 and 1
                     [0.03, 0.04]])       # source region 1
    K = (1.0 - beta.sum(axis=1))[None, :] * Kp + np.einsum('ni,imn->mn', beta, Kd)
    assert K[0, 1] == pytest.approx(0.579, abs=1e-12)
    assert K[1, 0] == pytest.approx(0.558, abs=1e-12)

    wrong = (1.0 - beta.sum(axis=1))[None, :] * Kp + np.einsum('in,imn->mn', beta, Kd)
    assert not np.allclose(K, wrong), "transposed beta must change the result"

    row = (1.0 - beta.sum(axis=1))[:, None] * Kp + np.einsum('ni,imn->mn', beta, Kd)
    assert not np.allclose(K, row), "row vs column scaling must change the result"
