# Multipoint kinetics examples

Two worked examples of Kobayashi multipoint kinetics (MPK) built on OpenMC's
fission-matrix tallies: a two-region coupled-core problem and a ten-region
axially segmented pin. Each one tallies the MPK parameter set, measures a
temperature-driven reactivity, and integrates the kinetics equations in time
against a point-kinetics reference derived from the same parameters.

The derivation, the index conventions and the open issues are in
[`docs/mpk_theory_manual.md`](../../docs/mpk_theory_manual.md). This document
covers what the examples do, what to look at in their output, and where the
method's practical limits sit.

| file | what it is |
|---|---|
| `mpk.py` | the implementation: deck construction, parameter extraction, validation, solver, sensitivity |
| `two_region.py` | two coupled cores separated by an absorber |
| `ten_region.py` | a PWR pin cell split into ten axial segments |
| `selftest.py` | exact algebra checks on `mpk.py`, no OpenMC run — run after editing it |

```
python two_region.py          # 4 transport runs
python ten_region.py          # 5 transport runs
```

Both take of order ten minutes at their default statistics on eight cores, and
both cache their statepoints in a working directory and reuse them, so a second
invocation only redoes the postprocessing. Both take `--particles`,
`--batches` and `--plot`. They need an `openmc` executable built from this
fork — the fission matrix, `k_dij` and `ell_m` do not exist in upstream OpenMC.

## Why multipoint kinetics

Point kinetics collapses a reactor onto one amplitude and a set of precursor
concentrations. That is exact when the flux shape does not move. When it does
move — a rod on one side, a coolant temperature profile up a channel, two
weakly coupled cores — point kinetics has no variable in which to express the
redistribution, and the error shows up as a wrong amplitude for reasons that
cannot be diagnosed from inside the model.

MPK partitions the system into `N` fissile regions and carries the regional
fission source rates `S_m(t)` as the unknowns. The equations are the exact
kinetics equations projected onto `N` region-wise weight functions, and for
`N = 1` they reduce algebraically to point kinetics. Everything is driven by
one object, the **coupling matrix**

```
K[m, n] = expected next-generation fission production in region m
          per neutron produced in region n
```

which OpenMC tallies directly during an ordinary eigenvalue calculation.

### Index conventions

Fixed, and the source of most errors in this kind of work:

| index | meaning | OpenMC filter |
|---|---|---|
| `m` | **response** region, where next-generation neutrons are produced | `fission_matrix_i_filter`, a `CellFilter` |
| `n` | **source** region, where the current-generation neutron was born | `fission_matrix_j_filter`, a `CellBornFilter` |
| `i` | delayed precursor family | `fission_matrix_d_filter`, a `DelayedGroupBornFilter` |

`K[m, n] == kij[m, n]`: the first axis is the response region. A transposed `K`
has the *same* fundamental eigenvalue, so no eigenvalue-based check will catch
a swap. Only the partition identity below will.

### The parameter set

| parameter | source | meaning |
|---|---|---|
| `Kp` `(N,N)` | `sp.kdij_mean[:, :, 0]` | prompt coupling `k^(p)_mn` |
| `Kd` `(D,N,N)` | `moveaxis(sp.kdij_mean[:, :, 1:], 2, 0)` | delayed coupling `k^(i)_mn`, per family |
| `beta` `(N,D)` | `delayed-nu-fission / nu-fission` | raw delayed fraction, indexed by **source** region |
| `ell` `(N,)` | `sp.ell_mean` | Kobayashi generation time per region |
| `lam` `(D,)` | `decay-rate / delayed-nu-fission` | precursor decay constants |
| `kij` `(N,N)` | `sp.kij_mean` | tallied total matrix — **validation only** |

The total matrix is assembled from the fission-source partition: every neutron
produced in region `n` is either prompt or a member of exactly one delayed
family, so

```python
K = (1 - beta.sum(axis=1))[None, :] * Kp + np.einsum('ni,imn->mn', beta, Kd)
```

`(1 - B_n)` and `beta_n^(i)` are **column** weights indexed by the source
region. Applied as row weights they are silently wrong.

`kij` is deliberately not an input. It is tallied independently of `kdij` and
of the beta tallies, which is what makes comparing the two a real check rather
than a tautology. `MPKParams` enforces it on construction and refuses to build
a parameter set that fails.

Two traps worth stating explicitly, both of which cost real time to find:

- **`kdij` bin 0 is `k^(p)`, the prompt coupling coefficient — not `(1-B)k`.**
  Bins `1..D` are bare `k^(i)`, each "production per neutron of its own
  family", so they do **not** sum to `kij`; they sum to roughly `D` times it.
  Reading the array as a partition inflates the precursor source by a factor
  of about `D` and overflows any integrator in milliseconds.
- **`ell_m` is not the IFP generation time.** `ell_m` is weighted by the
  single-generation importance `G_m+`; `Lambda_eff` from an IFP tally is
  weighted by the fundamental adjoint. They differ by far more than the factor
  of `k` relating `ell` and `Lambda` — about 10% on a water-reflected sphere —
  and no setting of `ifp_n_generation` makes an IFP score reproduce `G_m+`,
  because at least one generation of lag is unavoidable. Use `sp.ell_mean` for
  MPK and `Lambda_eff` for conventional point kinetics; never mix them in one
  parameter set.

## Example 1 — two coupled cores

```
   water   |  core 1  |  B4C  |  core 2  |   water
 reflector |          |  gap  |          | reflector
 ----------+----------+-------+----------+----------
     20 cm |  17.2 cm | 2 cm  |  17.2 cm |   20 cm
      vacuum                                vacuum
```

One-dimensional in `x`, transversely infinite (reflective `y` and `z`). Both
cores are the same homogeneous UO2/water mixture at 4.5 wt% enrichment; the
boron carbide sheet between them absorbs most of what tries to cross.

The model is symmetric, and that is the point. At the reference temperature
`S0 = [1/2, 1/2]` exactly and the coupling matrix has equal diagonals. Heating
one core alone breaks the symmetry, and the symmetry gives three things for
free:

- `alpha_1` must equal `alpha_2`. Their measured difference is an
  **assumption-free estimate of the statistical error** on the coefficient,
  which the script cross-checks against the uncertainty propagated from
  `kij_std_dev`.
- `S0` must be `[0.5, 0.5]`. Its departure measures the error on the matrix.
- The shape change must be antisymmetric.

Typical base-case output:

```
kij = [[0.9121 0.0887]      off-diagonal ~10% of diagonal
       [0.0884 0.9098]]     diagonals equal to 0.25%, as symmetry requires
k = 0.99956                 base case very close to critical
dominance ratio 0.8228      so the first harmonic decays slowly
ell = [4.044e-05, 4.035e-05] s          equal to 0.2%
B   = [688.47, 688.61] pcm              equal to 0.02%
delayed importance ratio 1.058
S0  = [0.50368, 0.49632]    symmetric to 0.7%, which measures the noise
```

at 20,000 particles x 250 batches. The symmetric quantities agreeing to a
fraction of a percent while `S0` is off by 0.7% is itself informative: the
diagonal of `K` is much better converged than the eigenvector, because the
eigenvector is sensitive to the small off-diagonal difference.

For a symmetric two-core system `K = [[d, c], [c, d]]`, so the eigenvalues are
`d + c` and `d - c` and the dominance ratio is `(d-c)/(d+c)`. The closer that
is to 1, the longer a tilt survives — which is precisely when MPK earns its
cost. The B4C gap is there to push it up.

**What to look at.** Step 3 reports the amplitude and the shape change
separately. The amplitude is what point kinetics gets, and it gets it to a few
percent. The shape change is what it cannot produce at all: the fission source
moves out of the heated core and into the cold one, settles within about a
millisecond — the prompt redistribution onto the new fundamental, on the `ell`
timescale — and then stays, while the amplitude goes on decaying on the delayed
timescale.

The script closes with a cross-check that ties its two halves together: the
asymptotic shape change of the integrated ODE solution against the first-order
bordered sensitivity from step 2. These are completely different computations —
an eigenvector perturbation and a stiff linear solve — and they agree to about
0.1% of the effect.

## Example 2 — ten axial regions

A single BEAVRS-specification PWR pin cell, radially infinite (reflective pitch
boundaries), axially finite with vacuum at both ends, active height 56 cm, fuel
divided into ten 5.6 cm segments. The height is chosen so axial leakage brings
`k` to about 1, which makes the fission source a cosine-like axial shape rather
than a flat one.

Three things make this harder and more informative than the two-region case.

**The matrix is banded, not diagonal.** The axial migration length is
comparable to one segment, so each region couples strongly to its neighbours
and weakly to everything else. That structure is what a many-region MPK model
has to exploit to be affordable.

**Identical regions are not equivalent.** Every segment has the same material,
so all the variation is geometric:

```
 seg        S0     ell (s)   B (pcm)
   0   0.03572  2.5711e-05    729.83
   1   0.07668  2.7201e-05    718.69
   ...
   4   0.14311  2.7086e-05    717.46
   5   0.14425  2.7456e-05    717.09
   ...
   9   0.03693  2.5465e-05    729.57
```

`S0` is a clean cosine, symmetric to better than 1%. `ell_m` is about 6%
shorter in the end segments, because a neutron near a vacuum boundary is more
likely to leak than to survive long enough to cause a late fission. Even `B_n`
is 1.7% higher at the ends, where the spectrum is harder. And the zone
coefficients come out `[-0.50, -1.82, -0.54]` pcm/K: the middle carries most of
the feedback, both because it holds four segments and because it sits where the
adjoint-weighted source peaks. A single bulk coefficient of −2.86 pcm/K hides
all of it.

**Mirror symmetry gives a free error bar.** The bottom and top of the pin are
the same perturbation reflected, so their coefficients must agree. No model
assumption enters that, which makes the difference the most trustworthy
uncertainty estimate available — and the one that reveals when the statistics
are not good enough.

### The perturbation basis is coarser than the region basis

This is the most practically important thing in the example, and it was not
the original design. Perturbing each of the ten segments in turn does not
work at any statistics this example can afford. Heating one segment of ten
inserts only about 200 pcm, and measured at 10^6 histories per case the result
was:

```
alpha_p (pcm/K)  [-0.68  0.37  0.07 -0.90 -0.70 -0.70 -0.90  0.07  0.36 -0.68]
1 sigma          [ 0.65  0.64  0.65  0.65  0.66  0.66  0.65  0.65  0.64  0.65]
```

Relative uncertainties from 95% to 900%, alternating signs, and a symmetry
check that failed outright: `alpha_0 = -0.68` against `alpha_9 = +0.37`, which
must be equal. That is noise with a plausible-looking shape, and without the
symmetry check it would have been easy to interpret as an axial profile.

So the kinetics keeps all ten regions — `K`, `S0`, `ell_m` and the transient
are all 10-region — while the *sensitivity* is tabulated on three axial zones:
segments 0–2, 3–6, 7–9. That triples the signal per case and cuts the run
matrix from eight cases to five, so each coefficient is several times better
determined. `Gamma` then has shape `(10, 10, 3)`.

This is the theory manual's own advice for the regime where statistics bind,
and it generalizes: the perturbation basis need not match the region basis.
Radial rings, axial levels, or assembly groups all work the same way.

### What `Gamma` buys you

### What `Gamma` buys you

```
Gamma[m, n, p] = dK[m, n] / dT_p          shape (N, N, P),  P = 3 zones here
```

Once the perturbed runs exist, **any** temperature profile costs a matrix
combination rather than another transport calculation. Step 3 uses this: it
applies a coolant temperature rise that is coldest at the bottom and hottest
at the top, built from `Gamma` alone, and integrates the result. A different
profile — a hot spot, a flat rise, a reversed tilt — is free.

`mpk.combine_cases()` does that combination on the **total** matrix and
distributes the change back onto `Kp` and `Kd` as an elementwise scaling, so
the partition identity survives exactly. Combining `Kp` and `Kd` separately
does not work: a difference between two runs' `k^(i)_mn` is mostly noise, and
summing several such differences drives the result negative. (`MPKParams`
rejects that outright, which is how the problem was found.) The price is one
stated assumption — that temperature scales prompt and delayed coupling by the
same elementwise factor — and it is far milder than trusting the elementwise
delayed differences.

The script also measures how *local* `Gamma` is: the fraction of `|dK/dT_p|`
in the rows and columns of the perturbed zone's own segments, compared against
the share of the matrix those rows and columns occupy. Well above that share
would mean the sensitivity is concentrated where the heat was applied, which is
what would let a many-region model store `Gamma` sparsely. Close to it means
`Gamma` is spread out — or that noise is, which at these statistics is the more
likely reading. This is the open question that decides whether the method
scales past a handful of regions, and it needs a converged `Gamma` to answer;
the example sets it up and reports it honestly rather than claiming to settle
it.

## Validation

`mpk.MPKParams.report()` runs every check that applies to a single parameter
set, and `mpk.check_first_order()` the ones that need perturbed runs. The
numbered checks follow the theory manual's hierarchy.

Measured on the two examples at their default statistics:

| | two-region | ten-region |
|---|---|---|
| partition identity | 1.4 pcm, 0.02σ | 25 pcm, 0.3σ |
| collapse identity vs Rayleigh | 0.0 | 0.0 |
| gauge, `sum(dS)` | 0.0 | 0.0 |
| bordered `dk` vs Rayleigh | 1.7e-18 | 6.5e-19 |
| check 10, worst nonlinearity at 300 K | 111 pcm on 1371 | 2.5 pcm on 547 |
| check 11, additivity | 168 pcm on 3096 (5%) | 154 pcm on 704 (22%, 1σ) |
| symmetry check on `alpha` | 0.3σ | 0.1σ |
| `rho / beta_eff` of the transient | −0.298 $ | −0.299 $ |
| MPK vs frozen-shape PK | 1.3e-02 | 3.9e-03 |
| bordered `dS` vs integrated shape change | 1.0% | 3.2% |

The last row is the one that ties the two halves of each script together: an
eigenvector perturbation and a stiff linear solve, computed by entirely
different routes, agreeing to a few percent of the effect.

| # | check | catches |
|---|---|---|
| 1 | partition identity, `K` vs `kij` elementwise in sigma | i/j transposition, beta on the wrong axis, mixed factorization |
| 2 | coupling signs, magnitudes, `kdij` bin convention | filter misconfiguration, bin misread |
| 3 | fundamental eigenvector single-signed | wrong eigenvalue selected, negative entries from noise |
| 4 | `rho / beta_eff` well inside one dollar | a prompt-supercritical test case, where nothing is measurable |
| 7 | collapse identity, `p_eff + sum(btil)` vs the Rayleigh quotient | wrong adjoint contraction |
| — | delayed importance ratio within ~10% of 1 | biased or unconverged `k^(i)_mn` |
| 8 | stationarity from the fundamental, `K` renormalized | wrong initialization, `(1-B)` applied as a row scaling |
| 9 | `sum(dS)` under the `1^T` gauge | gauge not enforced |
| 10 | `alpha_p * dT` vs an exact eigenvalue re-solve | `dT` too large for linearity |
| 11 | `sum_p alpha_p` vs one uniform-temperature run | systematically wrong `Gamma` |
| 13 | bordered `dk` vs the adjoint formula | inconsistent eigenvector normalization |

Checks 6 and 12 — the `N = 1` reduction to point kinetics, and second-order
convergence of the shape prediction — need either one region or a halved
perturbation, so they live in
[`tests/test_mpk_invariants.py`](../../tests/test_mpk_invariants.py), which runs
on synthetic parameters without OpenMC.

Three of these deserve emphasis.

**Check 1 is the backbone, but only for orientation.** It is the only check
sensitive to an i/j transposition. It is *not* a test of the delayed
coefficients: the delayed half of the partition enters weighted by
`beta ~ 1e-3`, so the reconstruction is dominated by the prompt term and the
identity stays well inside one sigma even when individual `k^(i)_mn` are badly
under-converged.

**The delayed importance ratio is the sharpest test of the delayed data.**
`MPKParams.importance_ratio()` returns `beta_eff` over the same adjoint
collapse with every `k^(i)_mn` replaced by the tallied `k_mn`. Weighting both
sides identically is what makes it meaningful when `B` varies between regions:
on the two-core Aufiero benchmark `B_n` is 386 and 984 pcm, and a plain
`beta_eff / mean(B)` reads 0.61 where the properly weighted ratio is 0.84.

It should sit within about 10% of 1. A missing delayed count can only pull it
*down*, so a value well below 1 is either real physics or a bias in
`k^(i)_mn`. Check 1 cannot tell them apart, because the deficit simply moves
into the prompt term: `K` stays right and the prompt/delayed split goes wrong.
That is the worst kind of error — invisible to every eigenvalue check, and
`beta_eff` is the parameter a transient is most sensitive to.

This is not hypothetical. Building these examples turned up exactly such a bias
in the `k_dij` estimator itself; see below.

**Check 7 is exact against the Rayleigh quotient, not the eigenvalue.** At the
unperturbed state the two coincide. Once a perturbation is applied and the
shape is frozen at the base `S0`/`S+` — which is exactly what the point-kinetics
reference does — they part company at second order. A frozen-shape reference
must take `rho` from the Rayleigh quotient, or that second-order difference
lands in the MPK-minus-PK residual, where it is indistinguishable from the
spatial effect the comparison exists to measure.

## Statistics: the real constraint

Monte Carlo noise, not solver cost, is what limits this method. At `N = 200`
the matrices are about 12 MB and the spectral solve takes a couple of seconds.
Converging the matrices is the problem.

**Temperature coefficients.** `alpha_p` is a small difference of two large
numbers. The examples use a 300 K perturbation deliberately — at 50 K the
signal would be buried at any statistics they could reasonably use — and then
*check* linearity with check 10 rather than assuming it. Sharing a seed between
the base and perturbed runs helps less than one might hope: once the cross
sections differ the histories diverge immediately, so the two runs are largely
uncorrelated. Both scripts propagate `sigma(alpha_p)` from `kij_std_dev` and
print it next to the coefficient. Halving it costs 4x the histories.

**Delayed coupling coefficients are the binding constraint.** There are
`N^2 * D` of them, and the far off-diagonal ones — a delayed neutron born in
`n` causing a fission in distant `m` — are the rarest scores in the problem.
An exactly zero coefficient means no history in the entire run had such an
event. `ten_region.py` reports the zero fraction, the importance ratio, and a
cross-check against `mpk.separable_delayed()`, which rebuilds `Kd` as
`r_i * kij` under the assumption that the delayed importance ratio is
family-dependent but region-independent, fitting each `r_i` only on elements
that have counts. If that independent estimate of `beta_eff` agrees with the
tallied one, the sparsity is not biting.

`k_ij`, `k^(p)_mn`, `ell_m` and `alpha_p` are all orders of magnitude better
converged than `Kd` and are usable long before it is.

### A generation-averaging bias, found while writing these examples

`k_dij` is estimated as the across-generation mean of a per-generation ratio
`p/s`, where `s` is the starting weight of neutrons of birth type `d` born in
region `n`. The original implementation set that ratio to **zero** for any
generation in which `s` was empty, and averaged those zeros in with the real
estimates — biasing the result toward zero in proportion to how often the bin
was empty.

With `N` regions and `D` families a single `(n, d)` source bin holds roughly
`beta^(d) / N` of the bank. On the ten-region pin at 10,000 particles per
generation the sparsest family has about 0.2 neutrons per generation per
region, so its bin was empty roughly 80% of the time and its coefficients came
back about five times too low.

Two things made it hard to see:

- **It scales with particles per *generation*, not total histories.** Running
  more batches does not reduce it, so the result looks converged and its
  standard deviation looks small.
- **The partition identity does not catch it.** The delayed half of the
  partition is weighted by `beta ~ 1e-3`, so a large relative deficit in
  `k^(i)_mn` is a small absolute deficit in `K` — and it moves into the prompt
  term, leaving `K` right.

Measured on the ten-region model, before and after excluding empty generations
from the mean:

| configuration | `beta_eff` before | `beta_eff` after |
|---|---|---|
| 5,000 particles x 440 batches | — | 727.1 pcm |
| 20,000 particles x 110 batches | — | 735.8 pcm |
| 10,000 particles x 120 batches | 576 pcm | — |
| 20,000 particles x 220 batches | 673 pcm | — |

Before the fix, two runs of nearly equal total histories disagreed by 17%
purely because of their generation size. After it, they agree to 1.2%, which
is statistics. The fitted per-family importance ratios went from
`[0.29, 0.78, 0.71, 1.01, 0.78, 0.50]` to `[1.07, 1.01, 1.00, 1.03, 1.06, 1.04]`.

Single-region and two-region models were never affected: their `(n, d)` bins
are populated in essentially every generation, which is why the bias survived
until a ten-region case existed. The fix excludes empty generations from both
the mean and the variance, counted per element, and
[`tests/regression_tests/fission_matrix_delayed/`](../../tests/regression_tests/fission_matrix_delayed/)
locks it with a four-region model at a deliberately small generation size.

## Limits of these examples

- Both are **illustrative statistics by default**. The two-region coefficients
  come out to about 5% at the default setting; the ten-region zone
  coefficients are considerably worse, and the example reports both the
  propagated uncertainty and the symmetry check so the reader can see it. A
  publication-grade coefficient on either needs one to two orders of magnitude
  more histories — see the table above.
- The `locality(Gamma)` result is **not an answer** to the manual's open
  question about whether `Gamma` is local. At these statistics the noise floor
  dominates it. The example provides the measurement and the right comparison
  baseline; a converged answer needs the history counts above.
- Temperature is taken to move only the coupling matrices. The scripts measure
  `ell_m` and `B_n` at the perturbed temperature and print the fractional
  change being neglected, which is under 2% here. Neither appears in `K`, so
  the assumption cannot bias `alpha_p` or the shape sensitivity; it affects
  only the integrated transient, where `ell` sets the prompt timescale and
  `beta` the delayed.
- The transient steps are **scaled** from the measured 300 K cases to stay
  inside a dollar. That is a linear extrapolation in temperature, bounded by
  check 10, and the scripts report both the scale factor and the equivalent
  temperature step.
- `K(t)` is constant within a transient. No thermal-hydraulic feedback, and no
  time-varying coupling matrix.
- Neither example addresses the manual's remaining open questions about
  absolute `ell` magnitudes on strongly reflected systems, beyond reporting the
  numbers.
