#ifndef OPENMC_PETSC_FE_SAMPLING_H
#define OPENMC_PETSC_FE_SAMPLING_H

#include <string>
#include <vector>
#include <array>
#include <cstddef>
#include <limits>
#include <petscsys.h>

namespace openmc {
namespace petsc_fe {

// Simple geometry types mirroring your libMesh interface
struct Position {
  double x;
  double y;
  double z;
};

struct Direction {
  double x;
  double y;
  double z;
};

// Initialize PETSc if needed, build DMPlex from mesh_file,
// load solution vector from sol_file, and prepare for sampling.
PetscErrorCode load_solution(const std::string& mesh_file,
                   const std::string& sol_file,
                   const std::string& variable_name);

// Destroy PETSc objects associated with the current solution.
void unload_solution();

// Reload the solution vector from sol_file, keeping the mesh / DM intact.
PetscErrorCode reload_solution(const std::string& sol_file);

// Return whether a solution is currently loaded.
bool has_solution();

//---------------------------------------------------------------------------
// Sampling interface
//---------------------------------------------------------------------------

// Sample the FE solution at arbitrary positions (batch).
//  positions  : list of sampling coordinates
//  values     : output, resized to positions.size()
//  return     : number of points that produced valid values
std::size_t sample_at_points(const std::vector<Position>& positions,
                             std::vector<double>& values);

// Sample the FE solution at points along a ray,
// given as explicit positions (same as libMesh version).
std::size_t sample_along_ray(const std::vector<Position>& positions,
                             std::vector<double>& temperatures);

// Templated version for fixed-size arrays (e.g. N=5),
// mirroring the libMesh templated API.
template<std::size_t N>
std::size_t sample_along_ray(const std::array<Position, N>& positions,
                             std::array<double, N>& temperatures)
{
  // Implemented in terms of the vector API.
  std::vector<Position> pos_vec(positions.begin(), positions.end());
  std::vector<double>   vals;

  std::size_t nvalid = sample_at_points(pos_vec, vals);

  const std::size_t ncopy = std::min<std::size_t>(N, vals.size());
  for (std::size_t i = 0; i < ncopy; ++i) {
    temperatures[i] = vals[i];
  }
  for (std::size_t i = ncopy; i < N; ++i) {
    temperatures[i] = std::numeric_limits<double>::quiet_NaN();
  }

  return nvalid;
}

} // namespace petsc_fe
} // namespace openmc

#endif // OPENMC_PETSC_FE_SAMPLING_H
