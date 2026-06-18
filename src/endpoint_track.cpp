#include "openmc/endpoint_track.h"

#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/position.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/vector.h"

#include <fmt/core.h>
#include <hdf5.h>

#include <cstddef> // for size_t
#include <string>

namespace openmc {

//==============================================================================
// Global variables (file-local)
//==============================================================================

namespace {
hid_t endpoint_file;   //!< HDF5 identifier for the endpoint-track file
hid_t endpoint_dtype;  //!< HDF5 compound datatype for a TrackState
int n_endpoints_written {0}; //!< Number of source particles written (per rank)
} // namespace

//==============================================================================
// Helpers
//==============================================================================

namespace {

//! Is the currently-open history (if any) still accepting states?
bool has_open_history(Particle& p)
{
  return !p.endpoint_tracks().empty() && p.endpoint_tracks().back().open;
}

} // namespace

//==============================================================================
// Recording hooks
//==============================================================================

bool check_endpoint_criteria(const Particle& p)
{
  // Only the "record all (up to a cap)" mode is supported for now. Explicit
  // (batch, gen, particle) selection could be added here in the same way as
  // settings::track_identifiers if desired.
  if (!settings::write_all_endpoints)
    return false;

  // Do not start recording until the configured batch. Checked before the
  // counter is touched so skipped batches don't consume the max_tracks budget.
  // (current_batch is 1-indexed over all batches; for eigenvalue runs set
  // endpoint_start_batch = n_inactive + 1 to record active batches only.)
  if (simulation::current_batch < settings::endpoint_start_batch)
    return false;

  int n;
#pragma omp atomic capture
  n = n_endpoints_written++;
  return n < settings::max_endpoint_tracks;
}

void add_endpoint_track(Particle& p)
{
  // Only record for source particles that were selected, and only store
  // neutron primary/secondary particles.
  if (!p.write_endpoints())
    return;
  if (!p.type().is_neutron())
    return;

  // Open an *empty* history. The birth state is captured later, in
  // event_calculate_xs, once the particle has been located -- calling
  // get_track_state() here (before location) reads an unresolved cell/material
  // and segfaults.
  auto& h = p.endpoint_tracks().emplace_back();
  h.particle = p.type();
  h.fission_death = false;
  h.open = true;
}

void record_endpoint_birth(Particle& p)
{
  if (!p.write_endpoints())
    return;
  if (!has_open_history(p))
    return;
  auto& h = p.endpoint_tracks().back();
  // Birth is the first state; only capture it once, after location.
  if (!h.states.empty())
    return;
  h.states.push_back(p.get_track_state());
}

void record_endpoint_collision(Particle& p)
{
  if (!p.write_endpoints() || !settings::endpoint_collisions)
    return;
  if (!has_open_history(p))
    return;
  auto& h = p.endpoint_tracks().back();
  // Birth must be states[0]; never let a collision be recorded first. In the
  // normal event order (calculate_xs -> advance -> collide) birth is already
  // present, so this only matters as a safety invariant.
  if (h.states.empty())
    return;
  h.states.push_back(p.get_track_state());
}

void record_endpoint_terminal(Particle& p)
{
  if (!p.write_endpoints())
    return;
  if (!has_open_history(p))
    return;

  auto& h = p.endpoint_tracks().back();
  // Only append a death state if a birth was actually captured (i.e. the
  // particle was located). A history that is still empty here belongs to a
  // particle lost before its first cross-section lookup; leave it empty so
  // finalize drops it, and don't call get_track_state() on an unlocated
  // particle.
  if (!h.states.empty()) {
    h.states.push_back(p.get_track_state());
    // Latch how this sub-particle died.
    h.fission_death = p.fission_death();
  }
  // Clear so the flag cannot leak into the next revived secondary.
  p.fission_death() = false;
  h.open = false;
}

//==============================================================================
// File management
//==============================================================================

void open_endpoint_file()
{
  // Open file and write filetype/version. Under MPI with more than one rank,
  // each rank writes its own file (combine afterward, as with tracks_p#.h5).
#ifdef OPENMC_MPI
  std::string filename;
  if (mpi::n_procs > 1) {
    filename =
      fmt::format("{}endpoint_tracks_p{}.h5", settings::path_output, mpi::rank);
  } else {
    filename = fmt::format("{}endpoint_tracks.h5", settings::path_output);
  }
#else
  std::string filename =
    fmt::format("{}endpoint_tracks.h5", settings::path_output);
#endif

  endpoint_file = file_open(filename, 'w');
  write_attribute(endpoint_file, "filetype", "endpoint_track");
  write_attribute(endpoint_file, "version",
    vector<int> {VERSION_ENDPOINT_TRACK[0], VERSION_ENDPOINT_TRACK[1]});
  write_attribute(
    endpoint_file, "collisions", settings::endpoint_collisions ? 1 : 0);
  write_attribute(
    endpoint_file, "fission_only", settings::endpoint_fission_only ? 1 : 0);

  // Compound type for Position (identical layout to track_output.cpp).
  hid_t postype = H5Tcreate(H5T_COMPOUND, sizeof(struct Position));
  H5Tinsert(postype, "x", HOFFSET(Position, x), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "y", HOFFSET(Position, y), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "z", HOFFSET(Position, z), H5T_NATIVE_DOUBLE);

  // Compound type for TrackState (identical layout to track_output.cpp).
  endpoint_dtype = H5Tcreate(H5T_COMPOUND, sizeof(struct TrackState));
  H5Tinsert(endpoint_dtype, "r", HOFFSET(TrackState, r), postype);
  H5Tinsert(endpoint_dtype, "u", HOFFSET(TrackState, u), postype);
  H5Tinsert(endpoint_dtype, "E", HOFFSET(TrackState, E), H5T_NATIVE_DOUBLE);
  H5Tinsert(
    endpoint_dtype, "time", HOFFSET(TrackState, time), H5T_NATIVE_DOUBLE);
  H5Tinsert(endpoint_dtype, "wgt", HOFFSET(TrackState, wgt), H5T_NATIVE_DOUBLE);
  H5Tinsert(
    endpoint_dtype, "cell_id", HOFFSET(TrackState, cell_id), H5T_NATIVE_INT);
  H5Tinsert(endpoint_dtype, "cell_instance",
    HOFFSET(TrackState, cell_instance), H5T_NATIVE_INT);
  H5Tinsert(endpoint_dtype, "material_id", HOFFSET(TrackState, material_id),
    H5T_NATIVE_INT);

  H5Tclose(postype);
}

void close_endpoint_file()
{
  H5Tclose(endpoint_dtype);
  file_close(endpoint_file);
  n_endpoints_written = 0;
}

//==============================================================================
// Flush to disk
//==============================================================================

void finalize_endpoint_track(Particle& p)
{
  // Make sure the final sub-particle's death state is captured.
  record_endpoint_terminal(p);

  // Assemble flat arrays across all neutron histories for this source particle,
  // applying the fission-only filter if requested.
  vector<int> offsets;
  vector<int> particles;
  vector<int> fission;     // 1 if that sub-particle died via fission
  vector<TrackState> states;

  int offset = 0;
  for (auto& h : p.endpoint_tracks()) {
    // Drop histories with no birth (particle lost before location) and, when
    // requested, anything that did not end in fission.
    if (h.states.empty())
      continue;
    if (settings::endpoint_fission_only && !h.fission_death)
      continue;
    offsets.push_back(offset);
    particles.push_back(h.particle.pdg_number());
    fission.push_back(h.fission_death ? 1 : 0);
    offset += static_cast<int>(h.states.size());
    states.insert(states.end(), h.states.begin(), h.states.end());
  }
  offsets.push_back(offset);

  // Nothing survived the filter -> nothing to write for this source particle.
  if (states.empty()) {
    p.endpoint_tracks().clear();
    return;
  }

#pragma omp critical(FinalizeEndpointTrack)
  {
    std::string dset_name = fmt::format("endpoint_{}_{}_{}",
      simulation::current_batch, simulation::current_gen, p.id());

    hsize_t dims[] {static_cast<hsize_t>(states.size())};
    hid_t dspace = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate(endpoint_file, dset_name.c_str(), endpoint_dtype,
      dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(
      dset, endpoint_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, states.data());

    write_attribute(dset, "n_particles", static_cast<int>(particles.size()));
    write_attribute(dset, "offsets", offsets);
    write_attribute(dset, "particles", particles);
    write_attribute(dset, "fission_death", fission);

    H5Dclose(dset);
    H5Sclose(dspace);
  }

  p.endpoint_tracks().clear();
}

} // namespace openmc
