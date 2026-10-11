## OpenMC Codebase Tools

Read the FULL `AGENTS.md` in this directory before starting work. It contains
project context, coding conventions, and documentation of the RAG search tools
registered in `.mcp.json`.

### Claude Code-specific: first-call behavior

The first `openmc_rag_search` call of each session returns an index status
message instead of search results. When this happens, you MUST use the
`AskUserQuestion` tool to present the rebuild/use-existing choice to the user.
Do not ask conversationally — always use the widget. Do not skip this step even
if the index looks current — the user may have uncommitted changes that warrant
a rebuild.

# MPK (multipoint kinetics) — working rules

Kobayashi multipoint kinetics on top of the OpenMC fission-matrix fork.
Full derivation, numbers and open issues: `docs/mpk_theory_manual.md`. Read it before
changing anything in the parameter extraction or the perturbation path.

## Index conventions — fixed, no exceptions

- `m` = response region (where next-gen neutrons are produced) → `fission_matrix_i_filter`, a `CellFilter`
- `n` = source region (where the current-gen neutron was born) → `fission_matrix_j_filter`, a `CellBornFilter`
- `i` = delayed family → `fission_matrix_d_filter`, a `DelayedGroupBornFilter`
- `K[m, n] == kij[m, n]`. First axis is the response region.

Array shapes, enforced on construction:

| name | shape | note |
|---|---|---|
| `Kp` | (N, N) | `kdij[:, :, 0]` |
| `Kd` | (D, N, N) | `np.moveaxis(kdij[:, :, 1:], 2, 0)` |
| `beta_raw` | (N, D) | indexed by **source** region n |
| `ell`, `B` | (N,) | |
| `lam` | (D,) | |

A subscript 0 in the manual means the base/unperturbed state, never an array index.

## Facts about the tallies that are easy to get wrong

- **`kdij` bin 0 is `k^(p)`, the prompt coupling coefficient — not `(1-B)k`.** Bins 1..D are bare
  `k^(i)`. They are per-neutron-of-that-type, so they do **not** sum to `kij`. Sanity check:
  `kdij[1:].sum()/kij` ≈ D (about 6.5), and `kdij[0]/kij` is within ~50 pcm of 1.
- **Raw beta is a plain forward tally** (`openmc.mgxs.Beta`, a **cell** filter, not cellborn).
  Never substitute OpenMC's IFP `beta_eff` for it — that is adjoint-weighted and estimates
  `beta_mn` instead. They differ by the delayed importance ratio, about 6% here.
- **`ell = k * Lambda`, not `Lambda`.** OpenMC builds `Lambda_eff` as
  `ifp-time-numerator / ifp-denominator / keff`, so numerator/denominator *is* `ell`.
- **`ifp_n_generation = 1`** for MPK parameters (Kobayashi wants the one-generation importance
  G+, not the fundamental adjoint). A PK reference for comparison uses the default larger count.
  Set it with the other settings, never inside a tally loop.
- **Never index `sp.tallies` by position.** It is keyed by tally ID. Look up by score set plus
  CellFilter bin and raise if the match is not unique.

## Assembly and the one check that matters

```python
K = (1 - beta_raw.sum(axis=1))[None, :] * Kp + np.einsum('ni,imn->mn', beta_raw, Kd)
```

- `(1 - B)` is a **column** scaling indexed by the source region. In the RHS it multiplies `S`
  *inside* the matvec: `Kp @ ((1 - B) * S)`. Applied to the output it is a row scaling and wrong
  (signature: ~4e-4 drift on a solution that should be stationary).
- `'ni,imn->mn'` sums over `i` only. `'in,imn->mn'` reads beta transposed — raises at N≠D,
  silent at N=6.
- Assert `|K - kij| / kij_sd < 3` elementwise at load time. This is the only check that catches
  an i/j transposition; a transposed K has the same fundamental eigenvalue and passes everything else.

## Never mix the two factorizations

The prompt term is `(1-beta_mn)*k_mn` **or** `(1-B_n)*k^(p)_mn` — both exact, and identical.
`(1-B_n)*k_mn` is wrong by `B_n*(kbar_d - k_mn)`, about 42 pcm here. Guard: in any product of a
beta factor and a coupling coefficient, the two must not share a denominator.

Tommasi Eq.(16) prints the prompt term with a single-subscript beta factored outside the n-sum.
That is a typo; Eq.(12) and Eq.(41) are correct. **Do not implement Eq.(16) literally.**

## Transients

- Initialize from the fundamental eigenvector of the **assembled** K, not of the tallied `kij`.
  Seeding `kij`'s eigenvector excites harmonics — 2.5e-4 vs 2.9e-6 shape artifact in the first ms.
- Renormalize `Kp` and `Kd` by `k` so the reference state is exactly critical.
- Solve spectrally (one eigendecomposition of the generator). Do not hand it to `solve_ivp`
  without `jac=A`, and do not use `expm_multiply` (‖At‖ ~ 5e5 makes it ~1e4x slower).
- Scale the state by its equilibrium values so one `rtol` means the same thing everywhere.

## Perturbation / temperature coefficients

- **Renormalize every case by the base `k0`, never by its own eigenvalue.** Self-normalizing both
  cases divides out the reactivity being measured.
- Working assumption: temperature moves only the coupling matrices (`d(beta)/dT = d(ell)/dT = 0`).
  This cannot bias `alpha_p` or the shape sensitivity, which depend on K alone. Record `ell` and
  `beta` at the temperature extreme anyway and quote what is being neglected.
- `alpha_p = (S+ @ Gamma[:,:,p] @ S0) / (k0**2 * (S+ @ S0))`, in pcm/K.
- Shape sensitivity via the **bordered solve**, not a mode sum. Gauge `1^T dS = 0` to match the
  `sum(S0) = 1` normalization. A mode sum needs three conventions to align with nothing enforcing
  them; the bordered form makes the gauge a row of the matrix.
- Fit against `sqrt(T)`, not `T`.

## Validation

`pytest tests/test_mpk_invariants.py` — runs on synthetic parameters, no OpenMC needed, and locks
every invariant above. Run it after any change to assembly, the RHS, or the sensitivity path.

Expected tolerances: partition identity exact to float; N=1 MPK/PK ≤ 1e-10; collapse identity
≤ 1e-14; stationarity ≤ 1e-12; gauge ≤ 1e-15; bordered dk vs Rayleigh ≤ 1e-15; shape error
ratio → 4 as dT halves.

## Open issues — do not paper over these

1. `ell/k` vs `Lambda_IFP` differ by 9.1% on a single region, where they should be equal.
   Prime suspect is the `CellFilter` on the IFP scores. **Test: same tally with and without the
   filter.** Blocks all multi-region work.
2. `beta_mn` (756.3 pcm) vs `beta_IFP` (774.4 pcm), 18 pcm apart. Possibly the same cause as (1).
3. `beta_eff/beta = 1.078` — above unity, unusual for a water-moderated system. May be real
   (U-238 threshold fission) but is unconfirmed, and the mapping hinges on it.
4. Hard-coded U-235 six-group half-lives used for a U-235/U-238 mixture. Replace with a
   `decay-rate` mgxs tally on the same cell filter.
5. Locality of `Gamma` is unknown and decides whether the method scales past a few regions.

If a result disagrees with one of these, say so — do not adjust parameters to make it agree.
