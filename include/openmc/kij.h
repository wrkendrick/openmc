//! \file kij.h
//! \brief Region-to-region fission matrix tallies (k_ij / k_dij)
//!
//! Implements the stochastic fission-matrix estimators described by
//! Tommasi, Aufiero, and Vicoli for multipoint-kinetics analysis. k_ij is
//! the average number of fission neutrons produced in region i by a neutron
//! born in region j; k_dij further resolves that quantity by delayed
//! neutron precursor family d. Both are estimated as a ratio of
//! per-generation numerator/denominator sums, and the reported mean/std
//! dev are the across-generation sample mean/variance of that per-generation
//! ratio -- the same statistical pattern used for k-effective in
//! eigenvalue.h/eigenvalue.cpp.
//!
//! The same fission-site loop also estimates ell_i, the Kobayashi
//! generation time of region i,
//!
//!     ell_i = <G_i+, (1/v) Phi> / <F_i Phi>,
//!
//! where G_i+ is the single-generation importance function solving
//! A+ G_i+ = nu Sigma_fi -- the expected production in region i at the
//! NEXT generation per neutron at a given point in phase space. Because
//! both the weighting and the time integral belong to the same (parent)
//! particle, ell_i is accumulated at zero generation lag: at each banked
//! fission site, the fissioning particle's own 1/v track integral
//! (Particle::lifetime()) is scored into the numerator of the i-bin the
//! site was produced in, and the site's weight into the denominator.
//!
//! This is deliberately NOT routed through OpenMC's IFP machinery. The IFP
//! scores (SCORE_IFP_TIME_NUM / SCORE_IFP_DENOM) attribute an ANCESTOR's
//! lifetime to a descendant's fission site, so a spatial filter on them
//! selects where the descendant fissions -- one fission generation later
//! than where the lifetime-carrying particle's own progeny were born. The
//! weighting that results is "production in region i at generation
//! ifp_n_generation + 1", which tends to the region-independent
//! fundamental adjoint as the generation count grows, and
//! ifp_n_generation >= 1 is enforced, so the zero-lag weight G_i+ cannot
//! be expressed that way. Measured on a two-region split-fuel sphere, the
//! regional contrast ell_2/ell_1 fell as 3.24, 1.64, 1.21, 1.04, 1.00 for
//! ifp_n_generation = 1, 2, 3, 5, 20 -- decaying per generation by roughly
//! the system's dominance ratio.

#ifndef OPENMC_KIJ_H
#define OPENMC_KIJ_H

#include <hdf5.h>

#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/tensor.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

// Per-generation accumulators (cleared at the start of each generation)
extern tensor::Tensor<double> kij_p;  //!< p_ij^(n), shape (n_i, n_j)
extern tensor::Tensor<double> kij_s;  //!< s_j^(n), shape (n_j)
extern tensor::Tensor<double> kdij_p; //!< p_dij^(n), shape (n_i, n_j, n_d)
extern tensor::Tensor<double> kdij_s; //!< s_dj^(n), shape (n_j, n_d)
extern tensor::Tensor<double> ell_p;  //!< <G_i+, (1/v)Phi>^(n), shape (n_i)
extern tensor::Tensor<double> ell_s;  //!< <F_i Phi>^(n), shape (n_i)

// Current generation's point estimate (recomputed every generation, not
// retained as a history)
extern tensor::Tensor<double> kij_generation;  //!< k_ij^(n), shape (n_i, n_j)
extern tensor::Tensor<double> kdij_generation; //!< k_dij^(n)
extern tensor::Tensor<double> ell_generation;  //!< ell_i^(n), shape (n_i)

// 1.0 where this generation produced a usable point estimate for that
// element, 0.0 where its denominator was empty. A generation in which no
// neutron of a given birth type was born in region j yields no information
// about k_dij(:, j, d); averaging a zero in its place would bias the
// estimate toward zero in proportion to how often that happens, which for
// a sparse (j, d) source bin is most of the time. See the comment in
// update_kij_stats().
extern tensor::Tensor<double> kij_valid;
extern tensor::Tensor<double> kdij_valid;
extern tensor::Tensor<double> ell_valid;

// Running sums for the across-generation Bessel-corrected mean/variance,
// mirroring simulation::k_sum for k-effective.
extern tensor::Tensor<double> kij_sum0;  //!< Sum of k_ij^(n) over generations
extern tensor::Tensor<double> kij_sum1;  //!< Sum of (k_ij^(n))^2
extern tensor::Tensor<double> kdij_sum0; //!< Sum of k_dij^(n) over generations
extern tensor::Tensor<double> kdij_sum1; //!< Sum of (k_dij^(n))^2
extern tensor::Tensor<double> ell_sum0;  //!< Sum of ell_i^(n) over generations
extern tensor::Tensor<double> ell_sum1;  //!< Sum of (ell_i^(n))^2

// Per-element count of generations that contributed a usable point
// estimate. Replaces the global generation count as the divisor in the
// mean and variance, since that count differs element by element.
extern tensor::Tensor<double> kij_count;
extern tensor::Tensor<double> kdij_count;
extern tensor::Tensor<double> ell_count;

// Final reported mean/std dev, set by calculate_average_kij()
extern tensor::Tensor<double> kij_mean;
extern tensor::Tensor<double> kij_std_dev;
extern tensor::Tensor<double> kdij_mean;
extern tensor::Tensor<double> kdij_std_dev;
extern tensor::Tensor<double> ell_mean;
extern tensor::Tensor<double> ell_std_dev;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

//! Allocate/zero the k_ij and k_dij accumulator tensors according to the
//! configured filters. Must be called once the i/j/d filters are known
//! (after settings.xml has been parsed) and before the first generation.
void init_kij_tallies();

//! Zero the per-generation numerator/denominator accumulators
void clear_kij_generation();

//! Score a newly-created fission site into the p_ij/p_dij numerator, and
//! the fissioning particle's 1/v track integral into the ell_i numerator.
//! \param[in] p the particle undergoing fission (the "parent")
//! \param[in] site the newly-created fission site
void accumulate_kij_fission_site(const Particle& p, const SourceSite& site);

//! Score a source particle's starting weight into the s_j/s_dj denominator.
//! Called once per primary source particle, at the point its birth cell
//! first becomes known.
//! \param[in] p the particle that was just sampled from the source bank
void accumulate_kij_source_particle(const Particle& p);

//! Reduce kij_p/kij_s (and kdij_p/kdij_s, ell_p/ell_s) across MPI processes
//! and form the current generation's point estimate k_ij^(n) (and
//! k_dij^(n), ell_i^(n))
void calculate_generation_kij();

//! Update the across-generation Bessel-corrected mean/variance of k_ij (and
//! k_dij, ell_i) from the current generation's point estimate. Only accumulates
//! statistics during active generations, mirroring calculate_average_keff().
void calculate_average_kij();

//! Write k_ij/k_dij/ell_i mean and standard deviation to the statepoint file
//! \param[in] group HDF5 group
void write_kij_hdf5(hid_t group);

//! Read k_ij/k_dij/ell_i mean and standard deviation from the statepoint file
//! \param[in] group HDF5 group
void read_kij_hdf5(hid_t group);

} // namespace openmc

#endif // OPENMC_KIJ_H
