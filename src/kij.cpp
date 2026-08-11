#include "openmc/kij.h"

#include <algorithm> // for max
#include <cmath>     // for sqrt, isfinite

#include "openmc/constants.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_match.h"
#include "openmc/tallies/tally.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

tensor::Tensor<double> kij_p;
tensor::Tensor<double> kij_s;
tensor::Tensor<double> kdij_p;
tensor::Tensor<double> kdij_s;

tensor::Tensor<double> kij_generation;
tensor::Tensor<double> kdij_generation;

tensor::Tensor<double> kij_sum0;
tensor::Tensor<double> kij_sum1;
tensor::Tensor<double> kdij_sum0;
tensor::Tensor<double> kdij_sum1;

tensor::Tensor<double> kij_mean;
tensor::Tensor<double> kij_std_dev;
tensor::Tensor<double> kdij_mean;
tensor::Tensor<double> kdij_std_dev;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void init_kij_tallies()
{
  if (!settings::kij_on)
    return;

  const auto& i_filter = *model::tally_filters[settings::kij_i_filter];
  const auto& j_filter = *model::tally_filters[settings::kij_j_filter];
  size_t n_i = static_cast<size_t>(i_filter.n_bins());
  size_t n_j = static_cast<size_t>(j_filter.n_bins());

  simulation::kij_p.resize({n_i, n_j});
  simulation::kij_s.resize({n_j});
  simulation::kij_generation.resize({n_i, n_j});
  simulation::kij_sum0.resize({n_i, n_j});
  simulation::kij_sum1.resize({n_i, n_j});
  simulation::kij_mean.resize({n_i, n_j});
  simulation::kij_std_dev.resize({n_i, n_j});
  simulation::kij_sum0.fill(0.0);
  simulation::kij_sum1.fill(0.0);
  simulation::kij_mean.fill(0.0);
  simulation::kij_std_dev.fill(0.0);

  if (settings::kij_d_filter >= 0) {
    const auto& d_filter = *model::tally_filters[settings::kij_d_filter];
    size_t n_d = static_cast<size_t>(d_filter.n_bins());

    simulation::kdij_p.resize({n_i, n_j, n_d});
    simulation::kdij_s.resize({n_j, n_d});
    simulation::kdij_generation.resize({n_i, n_j, n_d});
    simulation::kdij_sum0.resize({n_i, n_j, n_d});
    simulation::kdij_sum1.resize({n_i, n_j, n_d});
    simulation::kdij_mean.resize({n_i, n_j, n_d});
    simulation::kdij_std_dev.resize({n_i, n_j, n_d});
    simulation::kdij_sum0.fill(0.0);
    simulation::kdij_sum1.fill(0.0);
    simulation::kdij_mean.fill(0.0);
    simulation::kdij_std_dev.fill(0.0);
  }
}

void clear_kij_generation()
{
  if (!settings::kij_on)
    return;

  simulation::kij_p.fill(0.0);
  simulation::kij_s.fill(0.0);
  if (settings::kij_d_filter >= 0) {
    simulation::kdij_p.fill(0.0);
    simulation::kdij_s.fill(0.0);
  }
}

void accumulate_kij_fission_site(const Particle& p, const SourceSite& site)
{
  const auto& i_filter = *model::tally_filters[settings::kij_i_filter];
  const auto& j_filter = *model::tally_filters[settings::kij_j_filter];

  // i-bin: region the fission site was produced in (the parent's current
  // position, which equals site.r)
  FilterMatch match_i;
  i_filter.get_all_bins(p, TallyEstimator::COLLISION, match_i);
  if (match_i.bins_.empty())
    return;
  int i = match_i.bins_[0];

  // j-bin: region the parent particle was itself born in
  FilterMatch match_j;
  j_filter.get_all_bins(p, TallyEstimator::COLLISION, match_j);
  if (match_j.bins_.empty())
    return;
  int j = match_j.bins_[0];

#pragma omp atomic
  simulation::kij_p(i, j) += site.wgt;

  if (settings::kij_d_filter >= 0) {
    const auto& d_filter = *model::tally_filters[settings::kij_d_filter];
    FilterMatch match_d;
    d_filter.get_all_bins(p, TallyEstimator::COLLISION, match_d);
    if (!match_d.bins_.empty()) {
      int d = match_d.bins_[0];
#pragma omp atomic
      simulation::kdij_p(i, j, d) += site.wgt;
    }
  }
}

void accumulate_kij_source_particle(const Particle& p)
{
  const auto& j_filter = *model::tally_filters[settings::kij_j_filter];

  FilterMatch match_j;
  j_filter.get_all_bins(p, TallyEstimator::COLLISION, match_j);
  if (match_j.bins_.empty())
    return;
  int j = match_j.bins_[0];

#pragma omp atomic
  simulation::kij_s(j) += p.wgt();

  if (settings::kij_d_filter >= 0) {
    const auto& d_filter = *model::tally_filters[settings::kij_d_filter];
    FilterMatch match_d;
    d_filter.get_all_bins(p, TallyEstimator::COLLISION, match_d);
    if (!match_d.bins_.empty()) {
      int d = match_d.bins_[0];
#pragma omp atomic
      simulation::kdij_s(j, d) += p.wgt();
    }
  }
}

namespace {

//! MPI-reduce a per-generation tensor across all ranks, in place
void reduce_generation_tensor(tensor::Tensor<double>& t)
{
#ifdef OPENMC_MPI
  tensor::Tensor<double> reduced(t.shape());
  MPI_Allreduce(t.data(), reduced.data(), static_cast<int>(t.size()),
    MPI_DOUBLE, MPI_SUM, mpi::intracomm);
  t = reduced;
#endif
}

} // namespace

void calculate_generation_kij()
{
  reduce_generation_tensor(simulation::kij_p);
  reduce_generation_tensor(simulation::kij_s);

  size_t n_i = simulation::kij_p.shape(0);
  size_t n_j = simulation::kij_p.shape(1);
  for (size_t i = 0; i < n_i; ++i) {
    for (size_t j = 0; j < n_j; ++j) {
      double s = simulation::kij_s(j);
      simulation::kij_generation(i, j) =
        (s > 0.0) ? simulation::kij_p(i, j) / s : 0.0;
    }
  }

  if (settings::kij_d_filter >= 0) {
    reduce_generation_tensor(simulation::kdij_p);
    reduce_generation_tensor(simulation::kdij_s);

    size_t n_d = simulation::kdij_p.shape(2);
    for (size_t i = 0; i < n_i; ++i) {
      for (size_t j = 0; j < n_j; ++j) {
        for (size_t d = 0; d < n_d; ++d) {
          double s = simulation::kdij_s(j, d);
          simulation::kdij_generation(i, j, d) =
            (s > 0.0) ? simulation::kdij_p(i, j, d) / s : 0.0;
        }
      }
    }
  }
}

namespace {

//! Update the across-generation Bessel-corrected mean/std dev of a
//! per-generation ratio tensor, mirroring calculate_average_keff().
void update_kij_stats(int n, const tensor::Tensor<double>& generation,
  tensor::Tensor<double>& sum0, tensor::Tensor<double>& sum1,
  tensor::Tensor<double>& mean, tensor::Tensor<double>& std_dev)
{
  if (generation.empty())
    return;

  if (n <= 0) {
    // For inactive generations, use the current generation's value as the
    // running estimate (matches calculate_average_keff()'s treatment)
    mean = generation;
    return;
  }

  sum0 += generation;
  sum1 += generation * generation;

  mean = sum0;
  mean /= static_cast<double>(n);

  if (n > 1) {
    double t_value = 1.0;
    if (settings::confidence_intervals) {
      double alpha = 1.0 - CONFIDENCE_LEVEL;
      t_value = t_percentile(1.0 - alpha / 2.0, n - 1);
    }
    for (size_t k = 0; k < mean.size(); ++k) {
      double var = sum1.data()[k] / n - mean.data()[k] * mean.data()[k];
      double sd = t_value * std::sqrt(std::max(var, 0.0) / (n - 1));
      std_dev.data()[k] = std::isfinite(sd) ? sd : 0.0;
    }
  }
}

} // namespace

void calculate_average_kij()
{
  int n;
  if (simulation::current_batch > settings::n_inactive) {
    n = settings::gen_per_batch * simulation::n_realizations +
        simulation::current_gen;
  } else {
    n = 0;
  }

  update_kij_stats(n, simulation::kij_generation, simulation::kij_sum0,
    simulation::kij_sum1, simulation::kij_mean, simulation::kij_std_dev);

  if (settings::kij_d_filter >= 0) {
    update_kij_stats(n, simulation::kdij_generation, simulation::kdij_sum0,
      simulation::kdij_sum1, simulation::kdij_mean, simulation::kdij_std_dev);
  }
}

void write_kij_hdf5(hid_t group)
{
  write_dataset(group, "kij_mean", simulation::kij_mean);
  write_dataset(group, "kij_std_dev", simulation::kij_std_dev);
  if (settings::kij_d_filter >= 0) {
    write_dataset(group, "kdij_mean", simulation::kdij_mean);
    write_dataset(group, "kdij_std_dev", simulation::kdij_std_dev);
  }
}

void read_kij_hdf5(hid_t group)
{
  // Note: the running Bessel sums (kij_sum0/1) are not persisted, so a
  // restarted run resumes k_ij accumulation from zero rather than exactly
  // reproducing pre-restart statistics.
  read_dataset(group, "kij_mean", simulation::kij_mean);
  read_dataset(group, "kij_std_dev", simulation::kij_std_dev);
  if (settings::kij_d_filter >= 0) {
    read_dataset(group, "kdij_mean", simulation::kdij_mean);
    read_dataset(group, "kdij_std_dev", simulation::kdij_std_dev);
  }
}

} // namespace openmc
