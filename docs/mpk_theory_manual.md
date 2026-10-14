# Multipoint Kinetics: Theory and OpenMC Implementation

W. Reed Kendrick · MIT CRPG · as of 2026-10-08

> Source of record: the Claude doc this was exported from. Edit there and re-export,
> or edit here and treat this file as the source. Do not maintain both.

## Scope and status

This manual documents the multipoint kinetics (MPK) implementation as it currently stands: the Kobayashi formulation, its verified reduction to point kinetics, the regional temperature-reactivity formulation, and the OpenMC tallies and postprocessing that supply the parameters.

The theory reference is Tommasi (2019). Section 2.1 gives the Kobayashi derivation used throughout, 2.3.3 the single-region reduction to point kinetics, and the Appendix the conventional PK derivation. Two defects in that text matter for implementation and are documented here: an algebra slip in 2.3.3 (Section 5) and a subscript error in Eq.(16) (Section 3).

Verification status:

- Single-region MPK/PK identity: verified to 2.9e-15 relative on the amplitude over a 60 s transient, every parameter drawn from one statepoint. The earlier 1.6e-11 figure was limited by the ODE integrator, not the algebra; the spectral solver removed that floor.
- Fission-source partition identity: verified against the tallied coupling matrix at 0.02 to 0.77 sigma on three independent geometries.
- Multi-region solver and temperature-coefficient formulation: verified on real OpenMC parameter sets, not only synthetic ones. The two-region and ten-region examples in `examples/mpk/` run the full chain end to end.
- `ell_m` is now tallied directly by the transport code at zero generation lag (Section 13). The IFP route it replaced is not equivalent, and the difference is not small.
- Four of the five open issues in the first revision are resolved; see Section 18.

Not covered: Avery's MPK formulation (Tommasi 2.2), thermal-hydraulic feedback coupling, and time-varying K(t) within a transient.

## Part I — Theory

## Notation and index conventions

Every index error in this implementation has been a confusion between the response region and the source region, so the convention is fixed here and used without exception.

| index | range | meaning | OpenMC filter |
| --- | --- | --- | --- |
| m | 1..N | response region: where next-generation neutrons are produced | `fission_matrix_i_filter` (CellFilter) |
| n | 1..N | source region: where the current-generation neutron was born | `fission_matrix_j_filter` (CellBornFilter) |
| i | 1..D | delayed precursor family | `fission_matrix_d_filter` (DelayedGroupBornFilter) |
| p | 1..P | perturbed region, for sensitivity work | — |

K\[m, n\] = k\_mn, so the first axis is the response region. A transposed K has the same fundamental eigenvalue, so an eigenvalue check will not catch a swap; only the partition identity of Section 3 will.

| symbol | shape | meaning |
| --- | --- | --- |
| K | (N, N) | total coupling matrix, assembled per Section 3 |
| Kp | (N, N) | prompt coupling coefficients k^(p)\_mn |
| Kd | (D, N, N) | delayed coupling coefficients k^(i)\_mn |
| beta | (N, D) | raw delayed fractions, indexed by SOURCE region n |
| B | (N,) | raw total delayed fraction per source region, sum over i |
| ell | (N,) | Kobayashi generation times |
| lam | (D,) | precursor decay constants |
| S | (N,) | regional fission source rates, the MPK unknown |
| C | (D, N) | precursor concentrations |
| S0, Sadj | (N,) | fundamental right and left eigenvectors of K |

Two naming hazards worth stating explicitly. First, `beta` carries the source-region index n, not m: it multiplies columns of K, never rows. Second, a subscript 0 in this manual means the base (unperturbed) state — K0, k0, S0 — and never an array index, whereas in code `kij[0, 0]` uses 0 as a region index.

The raw delayed fraction B\_n and the importance-weighted fraction beta\_mn are different quantities throughout. Where the distinction matters the manual says "raw" or "importance-weighted" rather than relying on the symbol.

## The Kobayashi MPK equations

The MPK system is the exact kinetics equations projected onto N region-wise weight functions, with no approximation introduced until Eq.(18). Tommasi 2.1 carries the derivation; the structure is as follows.

The system is partitioned into N fissile regions. The precursor equation is split into N equations by multiplication by the region indicator, giving one flux equation plus ND precursor equations (Eq.8). The flux equation is then projected onto a weight function W\_m chosen for each region. Kobayashi's choice is the function G\_m+ solving the single-generation adjoint source problem

```latex
\mathbf{A}^{+} G_m^{+} = \nu \Sigma_{fm}
```

This choice makes the loss term collapse exactly onto the regional fission source, which becomes the MPK unknown:

```latex
\langle G_m^{+}, \mathbf{A}\Psi \rangle = \langle \mathbf{A}^{+}G_m^{+}, \Psi \rangle = \langle \nu\Sigma_{fm}\Psi \rangle = \langle \mathbf{F}_m\Psi \rangle = S_m
```

G\_m+ has a direct physical reading: the expected number of neutrons produced at the next generation in region m, per current-generation neutron placed at a given point in phase space. It is a one-generation importance function, not the fundamental adjoint flux — a distinction that matters when sourcing parameters from OpenMC's IFP machinery (Section 13).

The resulting N(1+D) equations are

```latex
\ell_m \frac{dS_m}{dt} = \sum_{n=1}^{N} (1-\beta_{mn}) k_{mn} S_n - S_m + \sum_{i=1}^{D} \lambda^{(i)} \sum_{n=1}^{N} k^{(i)}_{mn} C^{(i)}_n
```

```latex
\frac{dC^{(i)}_m}{dt} = \beta^{(i)}_m S_m - \lambda^{(i)} C^{(i)}_m
```

Term by term in the amplitude equation: the first sum is prompt production in region m from every source region; the subtraction of S\_m is the loss term, which is why a critical system satisfies K S = S; the double sum is delayed production, carrying its own coupling coefficients because delayed neutrons are born with a different spectrum and therefore a different importance. The precursor equation is local — precursors in region m are produced by fissions in region m and decay there — which is what gives the generator matrix of Section 8 its arrow structure.

The coefficients are defined in terms of the unknown dynamic flux. Eq.(18) replaces it with the static flux of the associated critical problem, which is the approximation that makes the parameters computable from a steady-state Monte Carlo run.

## The fission-source partition

Every neutron produced in region n is either prompt or a member of one delayed family, so the total coupling coefficient decomposes exactly:

```latex
k_{mn} = (1 - B_n)\, k^{(p)}_{mn} + \sum_{i=1}^{D} \beta^{(i)}_n k^{(i)}_{mn}
```

The weights (1 - B\_n) and beta\_n^(i) form a partition of unity over neutron birth type; on the test dataset they sum to 1.000000000000. Because every weight is indexed by the source region n, each one scales a column of K, never a row.

This identity is the implementation's primary validation check, since `kij` is tallied independently of `kdij` and the raw beta. On a 10k-particle single-region run the reconstruction gave 0.9945265 against a tallied 0.9945730, a 4.6 pcm discrepancy against a reported 1-sigma of 65 pcm.

### Two factorizations of the prompt term

The prompt coefficient multiplying S\_n must be the ratio of importance-weighted prompt production to total neutrons born in n. That single ratio can be split at either of two intermediates, giving two correct and algebraically identical forms:

```latex
\frac{\langle G_m^{+}, \mathbf{F}_{pn}\Phi\rangle}{\langle \mathbf{F}_n\Phi\rangle} = (1-\beta_{mn})\, k_{mn} = (1 - B_n)\, k^{(p)}_{mn}
```

The left form is Tommasi's Eq.(12), with beta\_mn the importance-weighted fraction of Eq.(18) — a fraction of next-generation *production*, carrying both indices. The right form is what this implementation uses, with B\_n the raw fraction of *neutrons born*, from Eq.(15). The two betas are related by the delayed importance ratio:

```latex
\beta_{mn} = B_n \cdot \frac{\bar{k}^{(d)}_{mn}}{k_{mn}}, \qquad \bar{k}^{(d)}_{mn} = \frac{\sum_i \beta^{(i)}_n k^{(i)}_{mn}}{\sum_i \beta^{(i)}_n}
```

On the test data: B = 713.7 pcm, importance ratio 1.0597, beta\_mn = 756.3 pcm. Both forms give 0.987004611386 for the prompt coefficient, identical to the last digit.

```text
                        <F_n Phi> = 1.0000000
                        all neutrons, unweighted
                       /                        \
          x k_mn      /                          \   x (1 - B_n)
       add G+        /                            \  keep only prompt
                    v                              v
     <G_m+, F_n Phi> = 0.9945265        <F_pn Phi> = 0.9928626
     importance-weighted                 prompt neutrons only
                    \                              /
       x (1 - b_mn)  \                            /  x k^(p)_mn
       keep prompt    \                          /   add G+
                       v                        v
                   <G_m+, F_pn Phi> = 0.9870046
                   the prompt coefficient
```

Left path applies importance weighting first, right path restricts to prompt first; both land on 0.9870046. The mixed form takes the first step of each and never takes a second, giving 0.9874282 — both its factors carry the denominator ⟨F\_n Φ⟩, so nothing cancels.

### The mixed form, and why it fails

Pairing the raw fraction with the total coupling coefficient — (1 - B\_n) k\_mn — is wrong by an exactly expressible amount:

```latex
(1-B_n)k_{mn} - (1-\beta_{mn})k_{mn} = B_n \left( \bar{k}^{(d)}_{mn} - k_{mn} \right)
```

This is +42.3 pcm on the test data. It vanishes only if delayed neutrons carry the same importance as prompt ones. The structural statement is that both factors in the mixed product share the denominator ⟨F\_n Phi⟩, so no normalization cancels; the product is ⟨F\_pn⟩⟨G+,F\_n⟩/⟨F\_n⟩², which equals the target only when k^(p)\_mn = k\_mn. A reliable guard: in any product of a beta factor and a coupling coefficient, the two must not share a denominator.

### A defect in Eq.(16)

Eq.(12) derives the prompt term as a sum over n of (1 - beta\_mn) k\_mn S\_n, with beta\_mn carrying both indices and sitting inside the sum. Eq.(16), which assembles the system, prints it as (1 - beta\_m) times the sum — single subscript, factored outside. Since Eq.(16)'s precursor equation uses beta\_m^(i), which Eq.(15) defines as the raw fraction, the literal reading of (1 - beta\_m) is the raw total, which is precisely the mixed form above.

Two separate problems: the index is the response region rather than the source region, and factoring a region-dependent quantity out of the n-sum requires beta\_mn to be independent of n. The second is mild — on a synthetic two-region case beta\_mn varies only about 5 pcm across n, so factoring a row mean costs 2.5 pcm. The first dominates, at roughly 52 pcm.

This is read as a typographical compression rather than a deliberate approximation, because Eq.(41) in the Avery section writes the same term with the two-index beta inside the sum, matching Eq.(12). **Eq.(12) and Eq.(18) are the authoritative statement; Eq.(16) as printed should not be implemented literally.**

### Implementation choice

The code uses (1 - B\_n) k^(p)\_mn, taking `kdij[:, :, 0]` and the raw beta tallies as inputs and leaving `kij` unused. That keeps the partition identity available as an independent check. The alternative — defining the prompt matrix as `kij` minus the delayed sum — is algebraically identical and makes `kdij[:, :, 0]` the check instead. Choose between them on which estimator is better converged in the run at hand: compare `kij_std_dev` against `kdij_std_dev[:, :, 0]`.

## Parameter definitions

Every coupling coefficient is defined by its denominator, and the denominator says which neutron population the coefficient is "per". Getting a parameter from the wrong denominator is the failure mode Section 3 describes.

| parameter | definition (Eq.18) | reads as | source |
| --- | --- | --- | --- |
| k\_mn | ⟨G\_m+, F\_n Φ⟩ / ⟨F\_n Φ⟩ | next-gen production in m per neutron produced in n | `sp.kij_mean` |
| k^(p)\_mn | ⟨G\_m+, F\_pn Φ⟩ / ⟨F\_pn Φ⟩ | same, per **prompt** neutron produced in n | `sp.kdij_mean[:, :, 0]` |
| k^(i)\_mn | ⟨G\_m+, F\_dn^(i) Φ⟩ / ⟨F\_dn^(i) Φ⟩ | same, per **family-i delayed** neutron produced in n | `sp.kdij_mean[:, :, i]` |
| beta\_n^(i) | ⟨ν\_d^(i) Σ\_fn Φ⟩ / ⟨F\_n Φ⟩ | raw delayed fraction, family i, region n — no importance weighting | `openmc.mgxs.Beta` |
| beta\_mn | ⟨G\_m+, F\_dn Φ⟩ / ⟨G\_m+, F\_n Φ⟩ | importance-weighted delayed fraction of production | derived, not tallied |
| ell\_m | ⟨G\_m+, (1/v) Φ⟩ / ⟨F\_m Φ⟩ | mean time for neutrons to reach m and produce the next generation | `sp.ell_mean` (zero-lag estimator, Section 13) |

Four consequences worth keeping in view.

**The coupling coefficients are all approximately unity.** Each is "expected next-generation production per neutron of its own type," so none of them knows how many neutrons of that type exist. The population fractions live entirely in the partition weights. A delayed coefficient exceeding k\_mn means delayed neutrons are *more* important than average, not more numerous.

**Raw beta is a plain forward tally.** Eq.(18) has no adjoint in the beta\_n^(i) definition, so it is a ratio of reaction rates in region n and is available directly from `openmc.mgxs.Beta` with a one-group structure. It must not be replaced by OpenMC's IFP beta\_eff, which is adjoint-weighted and therefore an estimate of beta\_mn instead.

**ell\_m is the Kobayashi generation time, not Lambda.** Two separate distinctions hide here and only the first is small. The factor of k, ell = k · Lambda, is 0.01% at k = 1.003. The *weighting function* is the large one: ell uses the single-generation importance G\_m+, while OpenMC's Lambda\_eff uses the IFP chain, which approaches the fundamental adjoint Phi+ as the generation count grows. Measured on a water-reflected sphere those differ by 9.75%, and regionally by orders of magnitude. Section 13 gives the mechanism.

**G\_m+ is a one-generation importance function.** It solves A+ G\_m+ = νΣ\_fm, a source problem, whereas the fundamental adjoint flux Φ+ solves an eigenvalue problem. The first revision of this manual asserted that for a single region with one fission spectrum the two are proportional, so that ell = k·Lambda would hold for any IFP generation count. That is wrong. Proportionality needs ∫χΦ+ dE to be spatially flat over the fissile region, which it is not even in a bare homogeneous sphere. The generation count is therefore not a modelling choice but a correctness requirement, and only g = 0 — unreachable through IFP — gives G\_m+. Hence the dedicated estimator.

## Exact reduction to point kinetics, N = 1

For a single region the Kobayashi system is algebraically identical to the point kinetics equations, so any disagreement in a numerical test is a parameter-mapping error and never a modelling difference. This is the implementation's primary correctness test.

The single-region system is

```latex
\ell \frac{dS}{dt} = \left[(1-B)k^{(p)} - 1\right] S + \sum_i \lambda^{(i)} k^{(i)} C^{(i)}, \qquad \frac{dC^{(i)}}{dt} = \beta^{(i)} S - \lambda^{(i)} C^{(i)}
```

and the unique exact mapping to the PK form of Tommasi Eq.(81) is

| PK parameter | from MPK |
| --- | --- |
| Lambda | ell / k |
| rho | (k - 1) / k |
| beta\_eff^(i) | (k^(i) / k) · beta^(i) |
| Ctilde^(i) | (k^(i) / ell) · C^(i) |

Verified to 2.2e-11 relative on the amplitude and 5.1e-11 on the precursors over a 60 s transient with synthetic parameters, and to 1.6e-11 using parameters assembled from an OpenMC statepoint.

### Two typos in Tommasi 2.3.3

The printed derivation cannot be followed literally. Dividing the amplitude equation by k·ell leaves (1/k) dS/dt on the left, not dS/dt. And Eq.(44) as printed drops a factor of 1/ell in the precursor substitution. The mapping table above is the corrected result.

### Three mapping errors to avoid

Each of these was encountered during development and each has a characteristic signature.

**Using IFP beta\_eff in the precursor equation.** Eq.(16) wants the raw beta^(i). If the adjoint-weighted value is used there *and* k^(i) appears in the delayed source term, the importance ratio k^(i)/k is counted twice. With k^(i)/k in the range 1.02 to 1.08 this inflated P(60 s) by 178% on a synthetic test. The clean fix is a change of variable: define Chat^(i) = k^(i) C^(i), so the precursor source becomes k^(i)beta^(i) and no beta appears in the delayed term at all.

**Using Lambda where ell belongs.** Small at k near unity, but it caps the achievable agreement at about three digits.

**Comparing against an independently-run PK case.** Two OpenMC runs of the same geometry give independent estimates of k. In one development case `keff` from a PK run and the fundamental of `kij` from an MPK run differed by 234 pcm, or 0.30 dollars, which is a different transient regardless of whether the algebra is right. The identity test must draw every parameter from one statepoint.

### Test design

For the identity test specifically, the parameter values are irrelevant — the reduction is exact for any of them. For the transient to be *meaningful*, keep the reactivity well inside one dollar: rho is inserted as a step at t = 0, and with Lambda near 1e-5 s a prompt-supercritical step makes both solutions overflow within milliseconds, at which point sub-percent parameter inconsistencies are amplified beyond recognition. Print rho/beta\_eff before interpreting any result.

## Multi-region: the adjoint-collapsed reference

For N greater than 1 there is no exact MPK/PK identity — Tommasi proves the reduction only for a single region — so the verification target has to change. The right reference is a point kinetics system *derived from the same parameter set*, obtained by contracting the MPK system with the adjoint eigenvector at frozen shape.

With S+ the left eigenvector of K and S0 the right one, define

```latex
\ell_{\text{eff}} = \frac{\langle S^{+}, \mathbf{L} S_0\rangle}{\langle S^{+}, S_0\rangle}, \quad p_{\text{eff}} = \langle S^{+}, \mathbf{P} S_0 \rangle, \quad \tilde{\beta}^{(i)} = \langle S^{+}, \mathbf{K}^{(i)} (\beta^{(i)} \odot S_0) \rangle
```

where L = diag(ell\_m) and P\_mn = (1 - B\_n) k^(p)\_mn. The collapsed system is then the single-region Kobayashi form, so Section 5's mapping applies unchanged: Lambda = ell\_eff/k, rho = (k-1)/k, beta\_eff^(i) = btil^(i)/k.

The collapse carries a free check. Both sides of

```latex
p_{\text{eff}} + \sum_i \tilde{\beta}^{(i)} = k
```

are ⟨S+, K S0⟩, so the identity is exact by construction and holds to 1e-16 in practice. If it fails, the contraction is wrong, not the physics.

### What the comparison measures

Once the reference is built this way, the MPK-minus-PK residual is a clean measurement of the *shape effect* rather than an accumulation of parameter inconsistencies. Measured on synthetic two-region parameters:

| case | max relative deviation | shape drift |
| --- | --- | --- |
| N = 1 (regression) | 1.7e-14 | 0 |
| N = 2, identical regions | 9.1e-09 | 1.0e-06 |
| N = 2, asymmetric (S0 = \[0.11, 0.89\], ell differing 30%) | 2.5e-07 | 1.0e-04 |
| N = 2, split fuel (S0 = \[0.028, 0.972\], ell within 2%) | 3.6e-08 | 1.5e-06 |

The degenerate case still reduces, as it must. On the split-fuel geometry the residual is around 1e-8, which is far below Monte Carlo noise on the off-diagonal coefficients — so for that geometry PK and MPK should agree to within statistics, and a visible disagreement indicates a parameter error rather than a spatial effect.

### When MPK earns its cost

A strongly coupled fuel region split in two gives almost no spatial signature: the only mechanism is ell\_1 differing from ell\_2. To see a real shape effect the regions must be genuinely decoupled — a moderator gap between fuel regions, or a perturbation that breaks symmetry such as a rod-like scaling of one column of K and Kd.

For the amplitude total, the plain sum over regions and the adjoint-weighted total differ by 4e-08 on the split-fuel case, so `S.sum(axis=0)` is an adequate scalar for comparison against PK and S+ is not needed for that purpose. S+ is needed for the collapse above and for the reactivity coefficients of Section 10.

## Steady state, normalization and initialization

The only non-transient starting condition is the fundamental eigenvector of the *assembled* coupling matrix. Any other start excites higher spatial harmonics that relax on the ell timescale and produce an artifact easily mistaken for physics.

### Why the eigenvector

Set the precursors to equilibrium, C^(i)\_n = beta\_n^(i) S\_n / lambda\_i, and substitute into the delayed source term:

```latex
\sum_i \lambda^{(i)} \sum_n k^{(i)}_{mn} C^{(i)}_n = \sum_n \left[\sum_i \beta^{(i)}_n k^{(i)}_{mn}\right] S_n
```

That bracket is exactly the delayed half of the partition in Section 3. Combined with the prompt half, the amplitude equation collapses to

```latex
\ell_m \frac{dS_m}{dt} = (\mathbf{K}S)_m - S_m
```

so K S0 = k S0 makes dS\_m/dt = (k-1)S\_m/ell\_m — pure decay with no shape transient. The eigenproblem reads componentwise as: propagate the source distribution through one fission generation, and the production landing in region m is proportional to what region m started with, by the same constant k for every m.

### Use the reconstructed K, not the tallied kij

The matrix whose eigenvector is stationary is the one the ODE integrates, which is the assembled K = (1-B\_n)Kp + sum\_i beta\_n^(i) Kd^(i). The tallied `kij` estimates the same physical matrix but is a different estimator with its own noise, so its eigenvector is not the null vector of the integrated operator. Seeding it excites the harmonics.

Quantified on a synthetic two-region case with `kij` perturbed from the reconstruction at a realistic 3e-4 level:

| initialized from | shape change over first 1 ms |
| --- | --- |
| reconstructed K | 2.9e-06 |
| tallied kij | 2.5e-04 |

Two orders of magnitude of purely numerical shape transient. It hides well, because the total amplitude is barely affected — both drop 15.3%, the physical prompt drop for the inserted reactivity — and the artifact appears only in the region split during the first millisecond.

A four-region synthetic case starting from a flat source converged to the fundamental shape within about 1 ms and produced a 0.2% amplitude overshoot with no reactivity inserted at all. Correct initialization held stationary to 2.4e-13.

### Normalization

K S = S has no solution unless the system is exactly critical, so renormalize Kp and Kd by k before integrating a reference transient. Then S0 is exactly stationary and perturbations are applied relative to a genuinely critical reference state.

S0 itself is defined only up to a scalar (Section 11). The code normalizes to sum(S0) = 1 so the MPK total starts at unity and is directly comparable to a PK amplitude of 1; the precursor initial condition is computed from the same S0, so the pair stays self-consistent.

One residual effect is physical, not an artifact: because ell\_m differs between regions, dS\_m/dt = (k-1)S\_m/ell\_m has a region-dependent relative rate, so a subcritical system drifts slightly in shape even from the fundamental. That is the 1e-06 level drift in the table above. It vanishes only when K is renormalized to exact criticality.

For N above a few hundred, use `scipy.sparse.linalg.eigs(k=1, which='LM')` rather than a dense decomposition. K is nonnegative, so Perron-Frobenius guarantees a positive fundamental and makes "largest real part" and "largest modulus" the same selection. Mixed signs in the selected eigenvector mean either the wrong eigenvalue was picked or K has negative entries from poor statistics; assert on it.

## Numerical solution

Between parameter updates the MPK system is linear with constant coefficients, so it should be solved spectrally rather than handed to a generic stiff integrator. The system is also considerably stiffer than PK, and gets stiffer as the region mesh is refined.

### State ordering and the generator

Fix the state vector as y = \[S\_1..S\_N, C^(1)\_1..C^(1)\_N, ..., C^(D)\_1..C^(D)\_N\], dimension N(1+D), and write the system as dy/dt = A y with

| block | value | structure |
| --- | --- | --- |
| A\_SS | L^-1 (P - I), P\_mn = (1-B\_n) k^(p)\_mn | dense N×N |
| A\_SC^(i) | L^-1 lambda\_i K^(i) | dense N×N |
| A\_C^(i)S | diag(beta^(i)) | diagonal |
| A\_C^(i)C^(i) | -lambda\_i I | diagonal |

where L = diag(ell\_m). Only the first block row and first block column are dense; everything else is diagonal. That arrow structure is what the solver should exploit.

Note that (1 - B\_n) multiplies S *inside* the matrix-vector product, not the result — it is a column scaling indexed by the source region. Applying it as a row scaling on the output is a silent error that shows up as a small drift on what should be a stationary solution, at the 4e-04 level in testing.

### Solve spectrally

One eigendecomposition of A gives y(t) = V exp(Λt) V^-1 y0 at every output time. Measured against Radau with the analytic Jacobian on a rod-insertion test, agreeing to 3e-11:

| N | states | spectral | Radau + analytic jac | Radau + FD jac |
| --- | --- | --- | --- | --- |
| 10 | 70 | 1.6 ms | 80 ms | 65 ms |
| 50 | 350 | 76 ms | 225 ms | 565 ms |
| 200 | 1400 | 2.5 s | 4.3 s | — |

Stiffness ratios for the same cases are 5.2e+06, 9.4e+06 and 3.7e+07: each spatial harmonic of P contributes a fast mode at (mu\_j - 1)/ell, so refining the region mesh makes the problem stiffer. The spectral solution is indifferent to this.

Two cautions. Never let `solve_ivp` build the Jacobian by finite differences — it costs N(1+D) extra RHS evaluations to rediscover a matrix available in closed form, and the penalty grows with N. And do not substitute `expm_multiply`: with ell near 2e-5 s and a 10 s window the norm of At reaches about 5e+05, forcing an enormous scaling-and-squaring count. It took 16 seconds at N = 10, ten thousand times slower than the eigendecomposition.

### Implicit stepping, when needed

For feedback or time-varying K, each implicit solve (I - hA)x = b can be reduced by Schur complement, because the precursor blocks are diagonal and can be eliminated analytically:

```latex
\hat{S} = A_{SS} + \sum_i A_{SC^{(i)}} \operatorname{diag}\!\left(\frac{h\, A_{C^{(i)}S}}{1 - h\, A_{C^{(i)}C^{(i)}}}\right)
```

This turns each factorization into an N×N problem instead of N(1+D)×N(1+D). Measured speedups: 4.9x at N = 50, 20.6x at N = 200, 27.9x at N = 500. The asymptotic factor is (1+D)^3 = 343 for six delayed groups; BLAS efficiency keeps it well short of that, but the gain is free.

### Scale the state

At the reference state C^(i)\_m = beta\_m^(i) S\_m / lambda\_i, so precursor magnitudes span roughly two decades across families while S is order unity. Divide each component by its equilibrium value — a similarity transform costing nothing — and y0 becomes the all-ones vector. A single `rtol` then means the same thing for every component, and `atol` stops being a per-family guess. For a verification report this is what makes one stated tolerance defensible.

## Part II — Reactivity Equivalents

## The perturbation framework

A regional temperature perturbation changes the coupling matrix, and the reactivity equivalent is extracted from that change. The governing rule is the normalization:

> Renormalize every case by the **base** eigenvalue k0. Never by its own eigenvalue.

If the base case is normalized K0 → K0/k0 and the perturbed case K\_pert → K\_pert/k\_pert, both have eigenvalue 1 and their difference contains no reactivity at all — the quantity being measured has been divided out. Normalizing both by k0 leaves K0/k0 as the critical reference state and K\_pert/k0 with eigenvalue k\_pert/k0, carrying exactly the inserted reactivity.

The perturbation is applied to the raw tallied matrices; k0 is the only normalizer that ever appears.

### Working assumption

This implementation takes temperature to affect only the coupling matrices:

```latex
\frac{\partial \beta^{(i)}_n}{\partial T} = 0, \qquad \frac{\partial \ell_m}{\partial T} = 0
```

This costs nothing for the coefficients themselves. Both alpha\_p and the shape sensitivity are functions of K alone — beta and ell appear nowhere in K's eigenvalue or eigenvector — so the assumption cannot bias them. It enters only when a transient is integrated, where ell\_m sets the prompt timescale and beta the delayed.

Two caveats to document rather than ignore. A spectrum shift alters the U-238 / U-235 fission ratio, and raw beta differs by roughly a factor of four between those nuclides, so d(beta)/dT is not negligible in a hard-spectrum region. And ell responds to spectrum hardening directly. Since the Beta and IFP tallies are already in the input deck, record ell\_m and beta\_n^(i) at the temperature extreme and quote the fractional change being neglected; under one percent across the range makes the assumption defensible in print rather than merely convenient.

### What is being differentiated

Perturbing one region changes the whole matrix, so the sensitivity object is a rank-3 tensor

```latex
\Gamma_{mn,p} = \frac{\partial K_{mn}}{\partial T_p}
```

of shape (N, N, P). Heating region p changes the spectrum of neutrons born there (column p), the fission rate there (row p), and the transport of everything passing through it (all elements), so Gamma is not confined to a row or column a priori. Whether it is *approximately* local is an empirical question that decides whether the method scales — see Section 17.

## Reactivity coefficient and contribution matrix

The sensitivity tensor collapses to a conventional scalar coefficient through the adjoint eigenvector, and the collapse decomposes elementwise into a map of which coupling pathways carry the feedback. That decomposition is the MPK-native product; the scalar alone is what a PK analysis would report.

### The scalar coefficient

First-order eigenvalue perturbation theory for a simple eigenvalue gives dk = ⟨S+, dK S0⟩ / ⟨S+, S0⟩. With rho = 1 - 1/k, so drho/dk = 1/k²:

```latex
\alpha_p = \frac{\partial \rho}{\partial T_p} = \frac{1}{k_0^2}\, \frac{\langle S^{+},\, \Gamma_{:,:,p}\, S_0\rangle}{\langle S^{+}, S_0\rangle}
```

This is in units of reactivity per Kelvin, reported as pcm/K, and is directly comparable to a conventional temperature coefficient.

### The contribution matrix

The same expression written before summation is

```latex
W_{mn,p} = \frac{S^{+}_m\, \Gamma_{mn,p}\, S_{0,n}}{k_0^2 \langle S^{+}, S_0\rangle}, \qquad \sum_{m,n} W_{mn,p} = \alpha_p
```

W\[:, :, p\] is an N×N map in pcm/K whose entries say how much of region p's feedback flows through each source-to-response pathway. Dividing by alpha\_p gives fractional contributions. This is the information a scalar coefficient discards, and it is what the formulation exists to produce.

### Units and reading

alpha\_p answers "what reactivity does heating region p alone insert, per Kelvin." The sum over p is the uniform-temperature coefficient, provided the perturbations are additive; Section 12 makes that a test rather than an assumption. For a multiphysics coupling, alpha\_p is the per-region feedback gain, and W tells you whether that gain is local or carried by cross-region coupling.

### Fit against the square root of temperature

Doppler reactivity varies roughly as the square root of T, so alpha\_T goes as T^(-1/2). Tabulating Gamma against sqrt(T) rather than T makes the interpolation nearly linear and lets three or four points per region cover a wide range. Verify linearity in sqrt(T) rather than assuming it.

### Worked magnitudes

On a four-region synthetic model with Doppler-like scaling, base k0 = 1.000400 and S0 = \[0.2169, 0.2853, 0.2793, 0.2185\]:

| perturbed region | alpha\_p (pcm/K) | exact rho at +50 K | first-order | error |
| --- | --- | --- | --- | --- |
| 0 | -1.092 | -53.80 pcm | -54.61 pcm | 0.81 pcm |
| 1 | -1.775 | -87.51 pcm | -88.74 pcm | 1.24 pcm |
| 2 | -1.743 | -85.93 pcm | -87.14 pcm | 1.22 pcm |
| 3 | -1.057 | -52.05 pcm | -52.84 pcm | 0.79 pcm |
| sum | -5.667 | -280.09 pcm (uniform) | -283.33 pcm | 3.24 pcm |

First-order agreement is about 1.4%, which locates where second-order terms begin to matter and sets the maximum useful perturbation size.

## Shape sensitivity, gauge, and the bordered solve

The eigenvector moves as well as the eigenvalue, and point kinetics has no way to represent that. The shape sensitivity is the second deliverable of the perturbation framework.

### The first-order equation and its deficit

Substituting the perturbed eigenpair, expanding, and cancelling K0 S0 = k0 S0 gives

```latex
(\mathbf{K}_0 - k_0 \mathbf{I})\, \delta S = \delta k\, S_0 - \delta \mathbf{K}\, S_0
```

N equations, N+1 unknowns. Two structural facts: the matrix on the left is singular by construction, since S0 spans its null space, so it cannot be inverted; and the deficit of one equation is gauge freedom, not a missing physical relation.

### Gauge freedom

An eigenvector is defined only up to a scalar — physically, a critical reactor has a definite source shape but no definite magnitude, since the same shape describes it at 1 W or 1 GW. Consequently, for any epsilon,

```latex
(\mathbf{K}_0 - k_0\mathbf{I})(\delta S + \epsilon S_0) = (\mathbf{K}_0 - k_0\mathbf{I})\,\delta S + \epsilon \cdot 0
```

so the solution set is a one-parameter family. Verified at epsilon = 0, 0.01, -0.5 and 7.0: the residual stays at machine zero while sum(dS) walks from 0 to 7.0. The epsilon direction is pure amplitude, and amplitude belongs to S(t), not to the eigenvector.

Fixing the gauge means adding one scalar equation, and the choice must match how S0 itself was normalized:

| gauge condition | preserves | dS on the test case |
| --- | --- | --- |
| 1^T dS = 0 | sum(S0) = 1 | \[-2.5e-05, -2.50e-04, 1.03e-04, 1.72e-04\] |
| S0^T dS = 0 | 2-norm of S0 | \[-1.6e-05, -2.38e-04, 1.14e-04, 1.80e-04\] |
| S+^T dS = 0 | biorthogonality (what a mode sum returns) | \[-1.7e-05, -2.39e-04, 1.13e-04, 1.80e-04\] |

All three are correct answers to different questions, and they differ by up to 36% on an individual component. **dk is identical across all three to 1e-19, and so is the renormalized perturbed shape**, so the physics is gauge-independent; only dS reported componentwise is not. A tabulated dS without a stated gauge is not comparable to anything.

This implementation uses 1^T dS = 0, which makes dS pure shape change with no amplitude content under the sum(S0) = 1 normalization already in use.

### The bordered solve

Rather than expanding in modes, solve the system directly with the gauge appended as an extra row and dk promoted to an unknown:

```latex
\begin{bmatrix} \mathbf{K}_0 - k_0\mathbf{I} & -S_0 \\ \mathbf{1}^{T} & 0 \end{bmatrix} \begin{bmatrix} \delta S \\ \delta k \end{bmatrix} = \begin{bmatrix} -\delta \mathbf{K}\, S_0 \\ 0 \end{bmatrix}
```

The extra column supplies the rank the singular block lacks; the extra row supplies the missing equation. The result is (N+1)×(N+1), nonsingular, and solved in one call. "Bordered" is the standard term — a singular matrix regularized by one added row and column, the workhorse of numerical continuation.

Advantages over the mode sum: no left eigenvectors, no eigenvalue pairing, no complex arithmetic, no small (k0 - k\_j) denominators, and the gauge is enforced as an equation so it cannot be silently violated (the solve returned sum(dS) = 2.7e-20). The bordered matrix depends only on K0, not on dK, so one `scipy.linalg.lu_factor` serves all P perturbations with only the right-hand side changing.

Measured accuracy: error 1.46e-07 against a shape effect of 2.50e-04, with the residual falling by ratios 3.80, 3.89, 3.94 as dT halves from 200 K — converging on 4, confirming a clean second-order residual rather than a bookkeeping error. Its dk agrees with the Rayleigh formula of Section 10 to 2e-19, as it must since they are the same projection.

### Why the mode sum is not recommended

The modal expansion dS = sum over j of x\_j ⟨y\_j, dK x\_0⟩/(k0 - k\_j) is correct mathematics and was implemented first. It failed, returning an error equal to the effect size. The cause was *not* eigenvalue pairing, which was healthy on the test case (eigenvalues real and well separated, |y\_j . x\_j| from 0.993 to 0.9998). It was two bookkeeping mismatches: the sum was built on LAPACK's unit-2-norm x0, whose components sum to 1.9834 rather than 1, and the biorthogonal gauge is not the sum-preserving gauge. Correcting both reproduces the bordered answer to the last digit.

The lesson is that a mode sum requires three independent conventions to align — x0's normalization, the biorthonormalization of y\_j, and the gauge — with no equation enforcing any of them. Pairing degeneracy and small denominators remain genuine hazards at larger N, but they were not the failure here.

## Validation hierarchy

Run these in order. Each is cheap, each catches a distinct class of error, and the early ones catch errors that make the later ones meaningless.

| # | check | expression | expected | catches |
| --- | --- | --- | --- | --- |
| 1 | Partition identity | reconstructed K vs tallied kij, elementwise in sigma | within 3 sigma | i/j transposition, beta on the wrong axis, mixed factorization |
| 1b | Delayed importance ratio | beta\_eff / adjoint-weighted raw beta | within ~10% of 1 | unconverged or biased k^(i)\_mn, which check 1 cannot see |
| 2 | Coupling sign and magnitude | all Kp, Kd non-negative; all near unity | positive | filter misconfiguration, bin misread |
| 3 | Fundamental positivity | S0 all one sign | uniform sign | wrong eigenvalue selected, negative K entries from noise |
| 4 | Reactivity scale | rho / beta\_eff | well inside 1 dollar | prompt-supercritical test case |
| 5 | ell consistency | production-weighted mean of ell\_m vs the unfiltered value | equal | region binning applied at the wrong event |
| 6 | N = 1 regression | MPK vs mapped PK | 1e-11 relative | any mapping regression |
| 7 | Collapse identity | p\_eff + sum(btil) vs the **Rayleigh quotient** | 1e-15 | wrong adjoint contraction |
| 8 | Stationarity | S(t)/S0 - 1 from the fundamental, K renormalized | 1e-13 | wrong initialization, (1-B) applied as a row scaling |
| 9 | Gauge | sum(dS) under the 1^T gauge | 1e-19 | gauge not enforced |
| 10 | First-order adequacy | alpha\_p · dT vs exact eigenvalue re-solve | a few % | dT too large for linearity |
| 11 | Additivity | sum over p of alpha\_p vs one uniform-T run | a few % | systematically wrong Gamma |
| 12 | Second-order convergence | shape error vs dT, halving dT | ratio → 4 | bookkeeping error masquerading as truncation |
| 13 | Rayleigh agreement | bordered dk vs adjoint formula | 1e-19 | inconsistent eigenvector normalization |

### Notes on the ones that matter most

**Check 1 is the backbone for orientation, and only for orientation.** It is
the only check sensitive to an i/j transposition, because a transposed K has
the same fundamental eigenvalue and passes every eigenvalue-based test. But
it is *not* a test of the delayed coefficients. The delayed half of the
partition enters weighted by beta ~ 1e-3, so a large relative deficit in
k^(i)\_mn is a small absolute deficit in K — and the deficit simply migrates
into the prompt term, leaving K right and the prompt/delayed split wrong.
Measured: the identity held to 0.4 sigma on a run whose delayed coefficients
were simultaneously 40% low. Make it fail loudly at parameter-load time, but
do not read it as a statement about Kd.

**Check 1b is what covers that gap.** Form beta\_eff from the delayed
coupling coefficients, then form the same adjoint collapse with every
k^(i)\_mn replaced by the tallied k\_mn, and take the ratio. Weighting both
sides identically is what makes it meaningful when B varies between regions:
on a two-core benchmark with B\_n of 386 and 984 pcm, a naive
beta\_eff / mean(B) reads 0.61 where the properly weighted ratio is 0.84.
The ratio should sit within about 10% of 1. A missing delayed count can only
pull it down, so a value well below 1 is either real physics or a bias in
k^(i)\_mn — and beta\_eff is the parameter a transient is most sensitive to.

**Check 5 is now a property of the estimator, not an assumption.** With the
zero-lag `ell_m` the region binning and the time integral come from the same
event by construction. The IFP version of this check was verified to machine
precision anyway (filtered and unfiltered agreed bit-for-bit), which is how
the real cause of the ell discrepancy was identified as the generation count.

**Check 7 is exact against the Rayleigh quotient, not against the
eigenvalue.** Both sides of p\_eff + sum\_i btil^(i) = ⟨S+, K S0⟩/⟨S+, S0⟩
are the same projection, so the identity is exact by construction. At the
unperturbed state S0 and S+ are K's own eigenvectors and the Rayleigh
quotient equals the eigenvalue, which is why the check reads as
"p\_eff + sum(btil) == k" there. Once a perturbation is applied and the shape
is held frozen at the base S0/S+ — exactly what the frozen-shape PK reference
of Section 6 does — the two part company at second order in the perturbation.
Measured: 3.3e-07 and 2.4e-06 on two real parameter sets, against 1e-16
for the Rayleigh form. **A frozen-shape PK reference must take rho from the
Rayleigh quotient**, or that second-order error lands in the MPK-minus-PK
residual, where it is indistinguishable from the spatial effect the
comparison exists to measure.

**Check 11 is the cheapest guard on the whole perturbation framework.** A
single uniform-temperature run compared against the sum of per-region
coefficients tests extraction, normalization and the adjoint contraction in
one number.

**Check 12 distinguishes truncation from bugs.** A first-order prediction
should have error scaling as dT squared. If halving dT does not quarter the
error, the residual is not the second-order term. This is what exposed the
mode-sum gauge mismatch.

### A check the hierarchy did not have

**S0 against an independently tallied fission source.** S0 is the
fundamental eigenvector of the assembled coupling matrix; an ordinary
`nu-fission` tally on the same regions is a completely different estimator
reached by a different code path — the fission-matrix accumulators in the
transport loop versus the normal tally framework. Agreement tests that K is
built *and oriented* correctly, and unlike the eigenvalue it is sensitive to
a transposition. Measured at 2.7% on a six-region HTGR parameter set. It
costs one extra tally and is worth having.

## Part III — OpenMC Implementation

## OpenMC implementation

Three groups of tallies supply the parameter set: the fission matrix settings give the coupling matrices, `openmc.mgxs.Beta` gives the raw delayed fractions, and an IFP score pair gives the generation times.

```text
  fission_matrix filters  --\
  (kdij bins 0, 1..D; kij)   \
                              \
  openmc.mgxs.Beta  ----------->  PARAMETER SET  --->  Transient
  (raw beta, per region)      /   Kp, Kd, beta, ell     (generator A, spectral solve)
                             /    K assembled from     \
  IFP scores, N_gen = 1  ---/     these                 --->  Sensitivity
  (ell_m, per region)                 :                       (Gamma, alpha_p, W)
                                      : (dashed)
                                      v
                           Partition identity
                           K vs kij, within 3 sigma
```

kij is deliberately not an input: it is tallied independently of kdij and the beta tallies, which is what makes the partition identity a real check rather than a tautology.

### Coupling matrices

```python
settings.fission_matrix_i_filter = openmc.CellFilter(bins=[c1, c2])      # m, response
settings.fission_matrix_j_filter = openmc.CellBornFilter(bins=[c1, c2])  # n, source
settings.fission_matrix_d_filter = openmc.DelayedGroupBornFilter(range(7))
```

The filter bin order fixes the region indexing for everything downstream; keep one list and reuse it.

**The kdij bin convention, confirmed from output.** Bin 0 is the prompt coupling coefficient k^(p)\_mn and bins 1..D are the delayed coefficients k^(i)\_mn. Each is a bare coupling coefficient — "expected production per neutron of that type" — so none is weighted by a delayed fraction and **they do not sum to kij**. The diagnostic that settles it:

| quantity | value | reading |
| --- | --- | --- |
| `kdij[1:].sum() / kij` | 6.52 | about D, so each bin is a bare k^(i); if the array were a partition this would be about beta\_eff |
| `kdij[0] / kij` | 0.999524, i.e. -47.6 pcm | k^(p), not (1-B)k, which would sit 714 pcm low |
| `kdij[1:] / kij` | 1.106, 1.110, 1.059, 1.071, 1.059, 1.111 | delayed importance ratios, all above unity |

Misreading bin 0 as (1-B)k is the 47 pcm error of Section 3. Misreading the array as a partition makes the precursor source about 6.5·S per second instead of 0.007·S, and Radau overflows within about 5 ms.

### Raw delayed fractions

```python
one_group = openmc.mgxs.EnergyGroups([0.0, 20.0e6])
for cell in fissile_cells:
    betas = openmc.mgxs.Beta(domain=cell, domain_type='cell',
                             energy_groups=one_group,
                             delayed_groups=list(range(1, 7)))
    for key in betas.tallies:
        tallies.append(betas.tallies[key], merge=False)
```

Note this is a **cell** filter — where the neutron is produced — not cellborn. Hand-rolling it as `delayed-nu-fission` with a `DelayedGroupFilter` over `nu-fission` with a `CellFilter` is equivalent.

This tally validates independently: it gave B = 713.7 pcm against 709.7 pcm back-derived from IFP beta\_eff through the importance ratio, on separate runs.

### Generation times

`ell_m` is tallied directly by the fork, in the same fission-site loop that
builds `kij`, and is read from `StatePoint.ell_mean`. No tally setup and no
settings are required: if `<fission_matrix>` is active, `ell_m` is reported.

```python
with openmc.StatePoint(path) as sp:
    ell = sp.ell_mean        # shape (N,), seconds
```

At each banked fission site the estimator accumulates the fissioning
particle's own 1/v track integral into the numerator of the i-bin the site
was produced in, and the site's weight into the denominator:

```
ell_m = sum over fission sites in m of [ tau_p * w ]   /   sum of w
```

where `tau_p` is `Particle::lifetime()`, the elapsed time since that
particle's birth. Both factors belong to the *same particle* and the *same
event*, which is precisely what makes the ratio
⟨G\_m+, (1/v)Φ⟩ / ⟨F\_m Φ⟩.

#### Why not IFP, and why the difference is large

The first revision of this manual built `ell_m` from an IFP score pair with
`ifp_n_generation = 1` and a `CellFilter`. That recipe is wrong, for a reason
that is structural rather than statistical.

`ifp()` fires at the *parent's* fission and stores the parent's lifetime on
the child's bank entry. `lifetimes[0]` is the oldest entry in the chain. At
scoring time the spatial filter is evaluated where the *child* fissions. So
at generation count g the weighting function is "expected production in
region m at generation g+1 per neutron at this phase-space point". Kobayashi's
G\_m+ is the g = 0 weight, and `ifp_n_generation >= 1` is enforced, so no
setting of the IFP machinery can produce it.

Two measurements settle this, both on one deck with one seed:

- A `CellFilter` on an IFP score *is* evaluated at the fission site: filtered
  and unfiltered IFP tallies on a single-fuel-region model came back
  bit-identical (numerator 1.713828e-05 both), and a non-fissionable cell
  scored exactly zero. The filter was never the problem.
- The generation count is. On the same model `ell` was 1.954473e-05 s at
  g = 1 and 1.780942e-05 s at g = 10 — a **9.75%** spread, which is the whole
  of the discrepancy that the first revision recorded as an open issue.

The regional contrast is worse than the global one. On a two-region
split-fuel sphere the ratio ell\_2/ell\_1 fell as 3.24, 1.64, 1.21, 1.04,
1.00 for g = 1, 2, 3, 5, 20: each extra generation multiplies the regional
signal by roughly the system's dominance ratio, and by g = 20 the weighting
is region-independent and all the spatial information is gone. The zero-lag
estimator on the same model gives 832 — the end blocks and the centre of a
reflected sphere differ by nearly three orders of magnitude in `ell`, because
the inner region is fed only by fast neutrons while the outer shell also
collects ~1e-4 s reflector returns.

That the estimator is measuring this and not an artefact was checked by
removing the reflector: on a bare sphere, where there is one fast population
everywhere, the contrast is **0.96**, and a spectrum tally confirms 0.0% of
production below 1 eV in the inner region against 21.5% in the outer when the
reflector is present.

**Use `ell_m` for MPK and `Lambda_eff` for conventional point kinetics, and
never mix them in one parameter set.** They are weighted by different
functions, not merely scaled by `k`.

### Settings hygiene

The MPK parameter set needs no IFP tally at all. If you add one anyway, to
produce a conventional point-kinetics reference for comparison, use the
*default* generation count: the conventional PK derivation weights with
Phi+, not G+. `ifp_n_generation` must be at least 1 and no greater than the
inactive batch count, and belongs with the other settings rather than inside
a tally loop.

The statepoint now records `ifp_n_generation` whenever IFP is active. It
previously did not, and that omission is the direct cause of the three
parameter-level "discrepancies" in the first revision: there was no way to
tell from a statepoint which weighting its IFP numbers carried.

## Statistics and the run matrix

Monte Carlo noise, not solver cost, is the binding constraint on this method. The coupling matrix has N² elements, each needing good statistics, and the delayed off-diagonals are the sparsest scores in the problem.

### Two constraints, satisfied by different knobs

**Total histories** set the uncertainty on k\_mn, on k-effective, and hence
on every reactivity difference. These scale as 1/sqrt(N) and are helped
equally by more particles or more batches.

**Particles per generation** set whether the delayed coefficients exist at
all. This is the constraint that is easy to miss and impossible to fix after
the fact. See the new open issue in Section 18: a generation whose (region,
family) source bin is empty carries no information about that element, and
the sparsest such bin is of order 1e-5 of the source bank. More batches does
not help; only more particles per batch does.

Keep at least ~200 **active** batches whatever the particle count, because
k\_ij and ell\_m take their variance from the spread across generations. A
handful of very large batches gives a good mean with a badly determined
uncertainty.

### Measured uncertainties

From a 25,000-particle, 180-active-batch run (4.5e+06 histories), single
region:

| quantity | 1-sigma |
| --- | --- |
| kij | 6.5e-04, i.e. 65 pcm on rho |
| kdij delayed bins, relative | 1.2% to 4.2% |

On a harder problem — an eleven-region axial partition of a prismatic HTGR
with explicit TRISO, 2.6e+05 histories at 2,000 particles per batch — the
same quantities read:

| quantity | value |
| --- | --- |
| max relative sigma on k\_mn in the tridiagonal band | 3.8% |
| sigma(k-effective) | 174 pcm |
| delayed coefficients exactly zero | 64% |
| median relative sigma on the non-zero ones | 52% |

The zero fraction is the particles-per-generation constraint biting, not the
history count. Scaling that run to 3e+07 histories at 100,000 particles per
batch projects the band to 0.35%, sigma(k) to 16 pcm, the sparsest delayed
bin to 1.8 neutrons per generation (84% of generations contributing), and the
band delayed coefficients to a median 4.8%.

### History requirement for a temperature coefficient

A temperature coefficient is a small difference of two large numbers. Scaling from the measured sigma above, for independent differencing of two runs at 10% precision on the coefficient:

| signal | needed sigma(k) | histories per run |
| --- | --- | --- |
| 50 pcm | 3.5 pcm | 1.5e+09 |
| 150 pcm | 10.6 pcm | 1.7e+08 |
| 500 pcm | 35 pcm | 1.5e+07 |

Multiply by two runs, by P regions, by the number of temperature points. A typical fuel coefficient of -3 pcm/K over a 50 K perturbation is a 150 pcm signal, putting a bare differencing approach at roughly 1.7e+08 histories per case.

### Three ways to make it affordable

**Correlated sampling.** Use the same seed and the same source distribution for base and perturbed runs so the difference variance falls well below sqrt(2)·sigma(k). This is close to mandatory at these signal sizes and is the standard approach for small reactivity differences.

**Large perturbations.** Use 300 to 500 K rather than fighting noise at 50 K, then verify linearity with validation checks 10 and 12 rather than assuming it. Tabulate against sqrt(T), where Doppler is nearly linear, so three or four points cover a wide range.

**Differential tallies, with caveats.** OpenMC supports [`TallyDerivative`](https://docs.openmc.org/en/latest/pythonapi/generated/openmc.TallyDerivative.html) with `variable='temperature'`, which would be far better statistically than differencing. Two things to verify before relying on it: whether it hooks into the fork's fission-matrix accumulation at all, since kij and kdij are not ordinary tallies; and whether the differential-operator implementation captures the perturbation's effect on the fission source distribution, which for an eigenvalue problem is a large part of the quantity wanted. If it only returns the direct cross-section term, it is a validation tool for alpha\_p rather than a route to Gamma.

### Choosing the region count

Plan the partition around statistical reachability, not solver cost. At N = 200 the matrices are about 12 MB and the spectral solve takes 2.5 s — trivial. Converging `kdij_std_dev` to a few percent on 240,000 delayed elements is not. The practical sequence is: run two or three regions on the real geometry, inspect the sigma map on the delayed coefficients and the locality of Gamma, and only then commit to a region count.

Exploit geometric symmetry to reduce independent regions, and consider perturbing groups of regions — radial rings, axial levels — rather than every cell.

Sources: [OpenMC kinetics parameters](https://docs.openmc.org/en/latest/usersguide/kinetics.html), [openmc.TallyDerivative](https://docs.openmc.org/en/latest/pythonapi/generated/openmc.TallyDerivative.html)

## Part IV — Postprocessing and Examples

## Postprocessing

### Never index sp.tallies by position

`sp.tallies` is keyed by tally ID, not position, and the iteration order of `betas.tallies` is an implementation detail. With two Beta tallies plus one IFP tally per region, positional indexing silently grabs the wrong tally as soon as a region is added. Look tallies up by score set and filter bin, and raise if the match is not unique.

```python
def _find_tally(sp, scores, cell_id):
    hits = []
    for t in sp.tallies.values():
        if set(t.scores) != set(scores):
            continue
        cf = [f for f in t.filters if isinstance(f, openmc.CellFilter)]
        if cf and list(cf[0].bins) == [cell_id]:
            hits.append(t)
    if len(hits) != 1:
        raise LookupError(f"expected 1 tally {scores} on cell {cell_id}, found {len(hits)}")
    return hits[0]
```

### Extraction

```python
with openmc.StatePoint(path) as sp:
    kij, kij_sd = np.asarray(sp.kij_mean), np.asarray(sp.kij_std_dev)
    kdij, kdij_sd = np.asarray(sp.kdij_mean), np.asarray(sp.kdij_std_dev)

    beta_raw = np.zeros((N, D))          # [n, i]  SOURCE region
    lam_region = np.zeros((N, D))
    ell = np.asarray(sp.ell_mean)        # (N,) seconds -- no tally needed
    for a, cid in enumerate(CELL_IDS):
        nu  = _find_tally(sp, ['nu-fission'], cid).mean.ravel()[0]
        dnu = _find_tally(sp, ['delayed-nu-fission'], cid).mean.ravel()
        dec = _find_tally(sp, ['decay-rate'], cid).mean.ravel()
        beta_raw[a] = dnu / nu
        lam_region[a] = dec / dnu          # per-region decay constants

Kp = kdij[:, :, 0]                       # (N, N)
Kd = np.moveaxis(kdij[:, :, 1:], 2, 0)   # (D, N, N)
```

The `moveaxis` puts the delayed family on the leading axis so `Kd[i]` is an N×N matrix, which is what every downstream einsum expects.

### Parameter container with validation

Standardize on one container with fixed shapes, validated on construction, carrying `kij` and `kij_sd` for checking only. Enforced on construction: shapes, positivity of ell and beta, non-negativity of Kp and Kd, and the partition identity in units of sigma. Making this fail at load time rather than as a wrong transient is the single highest-value piece of defensive code in the pipeline.

```python
K = (1 - beta_raw.sum(axis=1))[None, :] * Kp + np.einsum('ni,imn->mn', beta_raw, Kd)
z = np.abs(K - kij) / np.maximum(kij_sd, 1e-30)
assert z.max() < 3.0, f"partition identity off by {z.max():.1f} sigma"
```

The einsum index string matters. `'ni,imn->mn'` sums over i only; both n indices are the same index, so nothing is contracted over regions. Writing `'in,imn->mn'` reads beta as (D, N) instead of (N, D) — it raises a shape error at N ≠ D, but becomes a silent bug at N = 6.

### The MPK right-hand side

```python
def mpk(t, y):
    S = y[:N]
    C = y[N:].reshape(D, N)
    dS = (Kp @ ((1 - B) * S) - S + np.einsum('i,imn,in->m', lam, Kd, C)) / ell
    dC = beta_raw.T * S[None, :] - lam[:, None] * C
    return np.concatenate([dS, dC.ravel()])
```

`(1 - B)` multiplies S inside the matrix-vector product, not the result. As a column scaling indexed by the source region this is correct; applied to the output it is a row scaling and wrong, with a signature of about 4e-04 drift on a solution that should be stationary.

### Pipeline order

1. Load the statepoint, extract parameters by tally lookup.
2. Assemble K; run validation checks 1 through 3.
3. Solve the fundamental eigenpair; check 4 for reactivity scale.
4. Renormalize Kp and Kd by k0 for a transient, or keep them raw for perturbation work.
5. Build the generator, scale the state, solve spectrally.
6. For perturbation cases, assemble Gamma from base and perturbed matrices, then run checks 10 through 13.

## Worked examples

### Example 1: single-region identity from a statepoint

Model: 16.1 cm UO2 sphere, 50/50 U-235/U-238 at 10.2 g/cm3, inside 32 cm of water, vacuum outer boundary. ENDF/B-VIII.1. 10,000 particles, 200 batches, 20 inactive.

Extracted parameters:

| quantity | value |
| --- | --- |
| kij | 0.9945730 |
| kdij | \[0.99410, 0.95517, 1.04175, 0.99406, 1.07749, 1.09453, 1.03978\] |
| beta\_raw by family (pcm) | 22.7, 121.6, 118.6, 275.9, 123.5, 51.4 |
| B total | 713.7 pcm |
| ell | 1.9635e-05 s |

Reconstruction and mapping:

| step | result |
| --- | --- |
| k reconstructed | 0.9945265, i.e. -4.6 pcm vs tallied kij |
| rho | -545.7 pcm = -0.765 dollars |
| Lambda = ell / k | 1.9744e-05 s |
| beta\_eff mapped | 756.3 pcm |
| P(60 s), MPK and collapsed PK | agree to 1.6e-11 relative |

The lesson from this case is in what *failed* first. Comparing against a separately-run PK statepoint gave P(60 s) = 0.1299 against 0.0807, a ratio of 0.62, because the two runs' estimates of k differed by 234 pcm. The identity only closes when every parameter comes from one statepoint and the PK parameters are derived through the Section 5 mapping.

### Example 2: two-region split fuel

Model: the same geometry with the fuel split at r = 5 cm into two cells, both the same material. 40,000 particles, 400 batches.

What to expect before running: the inner sphere is a small fraction of the fuel volume and strongly coupled to the outer shell, so S0 should be heavily weighted to region 2, ell\_1 and ell\_2 should sit within 10 to 20% of each other, and the shape effect should be small — plausibly below Monte Carlo noise on the off-diagonal delayed coefficients. On synthetic parameters matching this configuration (S0 = \[0.028, 0.972\], ell = \[1.93, 1.96\]e-05 s), MPK and the collapsed PK agreed to 3.6e-08 with 1.5e-06 shape drift.

The diagnostics to print, in order: the elementwise `(K - kij)` in pcm and in sigma; the fundamental k and S0; ell\_m per region; B per region; and the denominator-weighted mean of ell\_m for comparison against an unfiltered IFP tally.

Agreement much worse than about 1e-03 on this geometry points at parameters rather than at the region splitting, and ell is the first place to look.

### Example 3: temperature coefficients, four regions

Synthetic Doppler-like model, source-region scaling in sqrt(T), base 900 K, central differences at +/-50 K. Base k0 = 1.000400, S0 = \[0.2169, 0.2853, 0.2793, 0.2185\], S+ = \[0.8887, 1.0977, 1.1013, 0.8534\].

Output, with every validation check from Section 12 that applies:

```text
alpha_p (pcm/K)        [-1.0922 -1.7749 -1.7429 -1.0567]   sum -5.667

check 10  first-order vs exact eigenvalue re-solve, +50 K
  region 0  exact  -53.80  1st-order  -54.61   diff 0.81 pcm
  region 1  exact  -87.51  1st-order  -88.74   diff 1.24 pcm
  region 2  exact  -85.93  1st-order  -87.14   diff 1.22 pcm
  region 3  exact  -52.05  1st-order  -52.84   diff 0.79 pcm

check 11  uniform +50 K: exact -280.09 pcm   sum_p alpha_p*dT -283.33 pcm   diff 3.24 pcm

check 12  bordered 1st-order shape prediction
  dT = 200.0 K   error = 2.158e-06
  dT = 100.0 K   error = 5.680e-07   ratio 3.80
  dT =  50.0 K   error = 1.459e-07   ratio 3.89
  dT =  25.0 K   error = 3.699e-08   ratio 3.94

check 13  bordered dk vs Rayleigh      agree to 2.2e-19
check  9  sum(dS) under the 1^T gauge  2.7e-20
```

The convergence ratios approaching 4 are the key diagnostic: they confirm the residual is the genuine second-order term rather than a bookkeeping error. An earlier mode-sum implementation produced an error of 2.249e-04 against a 2.499e-04 effect and did *not* show second-order convergence, which is how the gauge mismatch was found.

Contribution matrix for p = 0, as fractions of alpha\_0:

```text
          n=0     n=1     n=2     n=3
  m=0   0.4558  0.0370  0.0194  0.0076
  m=1   0.2787  0.      0.      0.
  m=2   0.1432  0.      0.      0.
  m=3   0.0583  0.      0.      0.
```

This particular synthetic model is row/column-separable by construction, so the apparent locality is an artifact of the test and carries no physical information. The real question — whether Gamma is concentrated in row p and column p on an actual geometry — is open and decides whether the method scales past a handful of regions.

### Reference scripts

Everything lives in `examples/mpk/` in the fork and is importable as one
module rather than scattered across notebooks.

| script | purpose |
| --- | --- |
| `mpk.py` | the implementation: deck construction, extraction, validating `MPKParams`, spectral solver, bordered sensitivity, contribution matrix |
| `two_region.py` | two coupled cores either side of an absorber: the smallest model with a shape effect PK cannot represent |
| `ten_region.py` | axially segmented PWR pin: banded K, zone-grouped sensitivity, locality of Gamma |
| `selftest.py` | exact-algebra checks on `mpk.py`, no OpenMC needed |

The synthetic invariants of the formulation are locked in
`tests/test_mpk_invariants.py`, and the estimators themselves in
`tests/regression_tests/fission_matrix*/`.

### Regions by mesh rather than by cell

`settings.fission_matrix_{i,j}_filter` accepts `MeshFilter`/`MeshBornFilter`
as well as the cell pair. This matters whenever the fuel sits deep in a
universe hierarchy and every instance shares one universe object — a
prismatic block core, a pin lattice — where a `CellFilter` cannot separate
regions without cloning the geometry once per region. A mesh partitions by
position and needs no geometry change at all. A `RectilinearMesh` lets the
region boundaries follow whatever partition the study actually uses.

## Open issues

Of the five items in the first revision, four are resolved. They had a single
common cause, and it was not the one suspected.

### 1. ell / k vs Lambda\_IFP — RESOLVED

Recorded as a 9.1% gap with the `CellFilter` on the IFP scores as prime
suspect. The filter was innocent: filtered and unfiltered IFP tallies in one
deck came back **bit-identical**, and a non-fissionable cell scored exactly
zero. The entire gap is the IFP generation count. On one deck with one seed,
`ell` was 1.954473e-05 s at `ifp_n_generation = 1` and 1.780942e-05 s at 10,
a 9.75% spread. The comparison was never an identity: `ell` wants the
single-generation importance G\_m+ and Lambda\_eff wants the fundamental
adjoint, and no IFP generation count produces the former. Superseded by the
zero-lag estimator of Section 13.

### 2. beta\_mn vs beta\_IFP — RESOLVED, same cause

Recorded as 18 pcm apart. Measured on a single deck at matched weighting:
beta\_mn = 755.72 pcm against beta\_IFP = 746.69 pcm at `ifp_n_generation = 1`,
**1.2% apart**. At generation count 10 the IFP value is 693.80 pcm, 8.9%
away. The quantities agree when weighted the same way; the original gap was
an unrecorded generation-count mismatch plus noise.

### 3. beta\_eff / beta running "the wrong way" — RESOLVED, same cause

Recorded as 1.078, above unity, flagged as unusual for a water-moderated
system and tentatively attributed to U-238 threshold fission. It is an
artefact of the weighting. On the same deck the ratio is 1.0462 at
`ifp_n_generation = 1` and **0.9721 at 10** — below unity, the normal
direction for a thermal system. Do not publish the U-238 explanation.

### 4. Delayed-group half-lives for a mixed fuel — RESOLVED

Replaced by a `decay-rate` mgxs tally on the same regions, sharing the
`delayed-nu-fission` denominator with the raw beta tally. Measured on a
Pu-239/U-238 plus U-235 two-core model, the tallied constants differ from the
hard-coded ENDF U-235 six-group values by up to **22%** (group 5: 0.888 vs
1.136 s^-1), while agreeing between the two regions to 0.03-3.5%, which is
the genuine nuclide-mix difference.

### 5. Locality of Gamma — STILL OPEN

Unchanged in substance, and now measurable. The contribution matrix W and a
locality fraction — the share of |dK/dT\_p| carried by the perturbed region's
own rows and columns, compared against the share of the matrix those occupy —
are reported by the example scripts. On a ten-region pin at modest statistics
the fraction sat close to the baseline share, i.e. Gamma is *not* obviously
concentrated. But at those statistics the noise floor dominates it, and noise
spreads |Gamma| uniformly, so the measurement is an upper bound on
delocalization rather than an answer. It needs a converged Gamma.

### A new issue the first revision did not have

**k\_dij was biased by its own generation averaging.** `k_dij` is the
across-generation mean of a ratio p/s, where s is the starting weight of
neutrons of birth type i born in region n. The original implementation set
that ratio to **zero** for any generation whose (n, i) bin was empty and
averaged those zeros in, biasing the estimate toward zero in proportion to
how often the bin was empty.

Two properties made it hard to see:

- It scales with particles per **generation**, not with total histories.
  Running more batches does not reduce it, so the result looks converged and
  its reported standard deviation looks small.
- The partition identity does not catch it, for the reason given under
  check 1.

Measured on a ten-region pin: two runs of nearly equal total histories
disagreed by **17%** on beta\_eff purely because of their generation size
(576 pcm at 10,000 particles/batch against 673 at 20,000). After excluding
empty generations from the mean and the variance, counted per element, the
same comparison agrees to 1.2%, and the fitted per-family importance ratios
went from `[0.29, 0.78, 0.71, 1.01, 0.78, 0.50]` to
`[1.07, 1.01, 1.00, 1.03, 1.06, 1.04]`. Single- and two-region models were
never affected, which is why it survived until a ten-region case existed.

The practical consequence is a sizing rule, not just a bug fix: **the
particle count per batch must be large enough that the sparsest (region,
family) source bin is populated in most generations.** That bin is
beta^(i) x S0\_n of the bank — of order 1e-5 — so it sets a floor of tens of
thousands of particles per batch independent of how many batches you run.

### Secondary items

- Off-diagonal delayed coupling coefficients remain the sparsest scores in
  the problem and set the statistical floor on every multi-region result.
- The locality question above is the one that decides whether the method
  scales past a handful of regions.
