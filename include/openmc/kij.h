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

// Current generation's point estimate (recomputed every generation, not
// retained as a history)
extern tensor::Tensor<double> kij_generation;  //!< k_ij^(n), shape (n_i, n_j)
extern tensor::Tensor<double> kdij_generation; //!< k_dij^(n)

// Running sums for the across-generation Bessel-corrected mean/variance,
// mirroring simulation::k_sum for k-effective.
extern tensor::Tensor<double> kij_sum0;  //!< Sum of k_ij^(n) over generations
extern tensor::Tensor<double> kij_sum1;  //!< Sum of (k_ij^(n))^2
extern tensor::Tensor<double> kdij_sum0; //!< Sum of k_dij^(n) over generations
extern tensor::Tensor<double> kdij_sum1; //!< Sum of (k_dij^(n))^2

// Final reported mean/std dev, set by calculate_average_kij()
extern tensor::Tensor<double> kij_mean;
extern tensor::Tensor<double> kij_std_dev;
extern tensor::Tensor<double> kdij_mean;
extern tensor::Tensor<double> kdij_std_dev;

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

//! Score a newly-created fission site into the p_ij/p_dij numerator.
//! \param[in] p the particle undergoing fission (the "parent")
//! \param[in] site the newly-created fission site
void accumulate_kij_fission_site(const Particle& p, const SourceSite& site);

//! Score a source particle's starting weight into the s_j/s_dj denominator.
//! Called once per primary source particle, at the point its birth cell
//! first becomes known.
//! \param[in] p the particle that was just sampled from the source bank
void accumulate_kij_source_particle(const Particle& p);

//! Reduce kij_p/kij_s (and kdij_p/kdij_s) across MPI processes and form the
//! current generation's point estimate k_ij^(n) (and k_dij^(n))
void calculate_generation_kij();

//! Update the across-generation Bessel-corrected mean/variance of k_ij (and
//! k_dij) from the current generation's point estimate. Only accumulates
//! statistics during active generations, mirroring calculate_average_keff().
void calculate_average_kij();

//! Write k_ij/k_dij mean and standard deviation to the statepoint file
//! \param[in] group HDF5 group
void write_kij_hdf5(hid_t group);

//! Read k_ij/k_dij mean and standard deviation from the statepoint file
//! \param[in] group HDF5 group
void read_kij_hdf5(hid_t group);

} // namespace openmc

#endif // OPENMC_KIJ_H
