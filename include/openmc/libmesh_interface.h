#ifndef OPENMC_LIBMESH_INTERFACE_H
#define OPENMC_LIBMESH_INTERFACE_H

#include "openmc/position.h"

#include <cstddef>
#include <string>
#include <vector>

namespace openmc {
namespace libmesh {

//============================================================================
// Solution loading
//============================================================================

void load_solution(const std::string& filepath,
                   const std::string& meshpath,
                   const std::string& system_name = "nl0",
                   const std::string& variable_name = "temp");

void unload_solution();

void reload_solution();

//============================================================================
// Runtime sampling
//============================================================================

std::size_t sample_along_ray(const std::vector<Position>& positions,
                             std::vector<double>& temperatures);

// Add this new overload for fixed-size arrays
template<std::size_t N>
std::size_t sample_along_ray(const std::array<Position, N>& positions,
                             std::array<double, N>& temperatures);

std::size_t sample_along_ray_fast(const Position& start, const Direction& dir,
                                   const std::array<double, 5>& distances,
                                   std::array<double, 5>& temperatures);

//============================================================================
// Query state
//============================================================================

bool has_solution();

} // namespace libmesh
} // namespace openmc

#endif