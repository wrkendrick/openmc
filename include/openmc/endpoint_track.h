#ifndef OPENMC_ENDPOINT_TRACK_H
#define OPENMC_ENDPOINT_TRACK_H

//! \file endpoint_track.h
//! \brief Birth/death (+ optional collision) recording for neutrons.
//!
//! This is a light-weight analogue of the full particle-track machinery in
//! track_output.{h,cpp}. Instead of recording *every* state along a particle's
//! history, it records only:
//!   * the birth state (first state of each primary/secondary neutron),
//!   * the death/terminal state (last state of that neutron), and
//!   * optionally every collision state in between.
//!
//! Output is written to "endpoint_tracks.h5" using the same compound TrackState
//! datatype as the regular track file, so existing readers/plotters that
//! understand the r/u/E/time/wgt/cell_id/cell_instance/material_id fields work
//! unchanged.

#include "openmc/particle.h"        // Particle, ParticleType
#include "openmc/particle_data.h"   // TrackState, EndpointHistory

namespace openmc {

//==============================================================================
// Constants
//==============================================================================

//! Revision of the endpoint-track file format. Bump if the schema changes.
constexpr int VERSION_ENDPOINT_TRACK[2] {1, 0};

//==============================================================================
// Non-member functions
//==============================================================================

//! Decide whether endpoint information should be recorded for a source
//! particle. Mirrors check_track_criteria(): honors settings::write_all_endpoints
//! and the settings::max_endpoint_tracks cap (counted per MPI rank).
bool check_endpoint_criteria(const Particle& p);

//! Begin a new (empty) endpoint history for the current primary or secondary
//! particle. Neutrons only; no-op for other particle types or for particles not
//! selected by check_endpoint_criteria(). The birth *state* is captured later,
//! once the particle has been located in the geometry (see
//! record_endpoint_birth) -- calling get_track_state() here would read an
//! unresolved cell/material and crash. Call everywhere add_particle_track() is
//! called.
void add_endpoint_track(Particle& p);

//! Capture the birth state into the open history the first time the particle is
//! located (i.e. when the history is still empty). Call from event_calculate_xs,
//! right beside write_particle_track(), so the geometry lookup is valid. No-op
//! after the birth state has been stored.
void record_endpoint_birth(Particle& p);

//! Record the current state as a collision point for the open history. No-op
//! unless settings::endpoint_collisions is enabled. Call from event_collide().
void record_endpoint_collision(Particle& p);

//! Close the currently-open history by appending the current state as its
//! death/terminal state, latching the fission-death flag, and marking it
//! closed. Call from event_revive_from_secondary() (before the next secondary
//! is revived) so each sub-particle gets its own terminal state.
void record_endpoint_terminal(Particle& p);

//! Open the endpoint output file ("endpoint_tracks.h5", or
//! "endpoint_tracks_p#.h5" per rank under MPI) and build the HDF5 datatypes.
void open_endpoint_file();

//! Close the endpoint output file and reset the per-rank write counter.
void close_endpoint_file();

//! Close the last open history, then flush all recorded neutron histories for
//! this source particle to the file (applying the fission-only filter when
//! settings::endpoint_fission_only is set) and clear the per-particle buffer.
//! Call from event_death().
void finalize_endpoint_track(Particle& p);

} // namespace openmc

#endif // OPENMC_ENDPOINT_TRACK_H
