#include "openmc/petsc_fe_sampling.h"

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscviewerhdf5.h>

#include <iostream>
#include <limits>
#include <vector>
#include <cstdlib>

namespace openmc {
namespace petsc_fe {

namespace {

//---------------------------------------------------------------------------
// Module state
//---------------------------------------------------------------------------

bool petsc_initialized = false;
bool solution_loaded   = false;

// Core PETSc objects
DM  dm        = nullptr; // DMPlex mesh + FE
Vec u_global  = nullptr; // global FE solution

// Basic FE / geometry info
PetscInt g_dim = 0;  // spatial dimension
PetscInt g_dof = 1;  // number of dofs per point (scalar field)

// Simple fatal error helper; adapt to your own error system if desired.
[[noreturn]] void fatal_error(const std::string& msg)
{
  std::cerr << "FATAL (PETSc FE): " << msg << std::endl;
  std::abort();
}

void init_petsc_if_needed()
{
  if (petsc_initialized) return;

  int argc = 1;
  const char* argv_raw[] = {"openmc-petsc", nullptr};
  char** argv = const_cast<char**>(argv_raw);

  PetscErrorCode ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr) fatal_error("Failed to initialize PETSc.");

  petsc_initialized = true;
}

// Core routine: given a batch of positions, interpolate u_global at those
// points using DMInterpolation (hash-based point location inside DMPlex).
PetscErrorCode sample_at_points_impl(const std::vector<Position>& positions,
                                     std::vector<double>& values)
{
  PetscErrorCode ierr;

  const PetscInt N = static_cast<PetscInt>(positions.size());
  values.assign(N, std::numeric_limits<double>::quiet_NaN());
  if (N == 0) return PETSC_SUCCESS;

  // Flatten coordinates into a PETSc Real array [x0,y0,z0, x1,y1,z1, ...]
  std::vector<PetscReal> pts;
  pts.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(g_dim));
  for (PetscInt i = 0; i < N; ++i) {
    pts[i*g_dim + 0] = positions[i].x;
    if (g_dim > 1) pts[i*g_dim + 1] = positions[i].y;
    if (g_dim > 2) pts[i*g_dim + 2] = positions[i].z;
  }

  DMInterpolationInfo ctx = nullptr;
  ierr = DMInterpolationCreate(PetscObjectComm(reinterpret_cast<PetscObject>(dm)),
                               &ctx);CHKERRQ(ierr);
  ierr = DMInterpolationSetDim(ctx, g_dim);CHKERRQ(ierr);
  ierr = DMInterpolationSetDof(ctx, g_dof);CHKERRQ(ierr);

  // Add all points in one batch
  ierr = DMInterpolationAddPoints(ctx, N, pts.data());CHKERRQ(ierr);

  // For a DMPLEX, SetUp builds and uses a grid hash for point location.
  ierr = DMInterpolationSetUp(ctx, dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);

  // Create result Vec
  Vec v;
  ierr = DMInterpolationGetVector(ctx, &v);CHKERRQ(ierr);

  // Evaluate u_global at the given points
  ierr = DMInterpolationEvaluate(ctx, dm, u_global, v);CHKERRQ(ierr);

  // Copy results back
  const PetscScalar* a = nullptr;
  ierr = VecGetArrayRead(v, &a);CHKERRQ(ierr);
  for (PetscInt i = 0; i < N; ++i) {
    values[static_cast<std::size_t>(i)] = PetscRealPart(a[i]);
  }
  ierr = VecRestoreArrayRead(v, &a);CHKERRQ(ierr);

  // Cleanup
  ierr = DMInterpolationRestoreVector(ctx, &v);CHKERRQ(ierr);
  ierr = DMInterpolationDestroy(&ctx);CHKERRQ(ierr);

  return PETSC_SUCCESS;
}

} // anonymous namespace

//---------------------------------------------------------------------------
// Public interface
//---------------------------------------------------------------------------

PetscErrorCode load_solution(const std::string& mesh_file,
                   const std::string& sol_file,
                   const std::string& variable_name)
{
  (void)variable_name; // currently unused; kept for API similarity.

  init_petsc_if_needed();

  PetscErrorCode ierr;

  if (solution_loaded) {
    // Destroy previous state
    if (u_global) { ierr = VecDestroy(&u_global);CHKERRQ(ierr); u_global = nullptr; }
    if (dm)       { ierr = DMDestroy(&dm);CHKERRQ(ierr);        dm       = nullptr; }
    solution_loaded = false;
  }

  // 1) Build DMPlex from mesh file
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD,
                              mesh_file.c_str(),
                              nullptr,    // file type (auto-detect)
                              PETSC_TRUE, // interpolate faces/edges
                              &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = DMGetDimension(dm, &g_dim);CHKERRQ(ierr);

  // For now, assume scalar field; can be generalized via PetscDS/PetscFE.
  g_dof = 1;

  // 2) Create a global Vec and load solution from HDF5
  ierr = DMCreateGlobalVector(dm, &u_global);CHKERRQ(ierr);

  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,
                             sol_file.c_str(),
                             FILE_MODE_READ,
                             &viewer);CHKERRQ(ierr);

  // If file has multiple named Vecs, you can set the Vec name here:
  // PetscObjectSetName(reinterpret_cast<PetscObject>(u_global),
  //                    variable_name.c_str());

  ierr = VecLoad(u_global, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  solution_loaded = true;

  return PETSC_SUCCESS;
}

void unload_solution()
{
  if (!solution_loaded) return;

  PetscErrorCode ierr;

  if (u_global) { ierr = VecDestroy(&u_global);(void)ierr; u_global = nullptr; }
  if (dm)       { ierr = DMDestroy(&dm);(void)ierr;        dm       = nullptr; }

  solution_loaded = false;
}

PetscErrorCode reload_solution(const std::string& sol_file)
{
  if (!solution_loaded)
    fatal_error("No solution to reload.");

  PetscErrorCode ierr;
  PetscViewer viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,
                             sol_file.c_str(),
                             FILE_MODE_READ,
                             &viewer);CHKERRQ(ierr);
  ierr = VecLoad(u_global, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  return PETSC_SUCCESS;
}

bool has_solution()
{
  return solution_loaded;
}

//---------------------------------------------------------------------------
// Sampling
//---------------------------------------------------------------------------

std::size_t sample_at_points(const std::vector<Position>& positions,
                             std::vector<double>& values)
{
  if (!solution_loaded)
    fatal_error("No solution loaded. Call petsc_fe::load_solution() first.");

  PetscErrorCode ierr = sample_at_points_impl(positions, values);
  if (ierr) fatal_error("PETSc error in sample_at_points_impl().");

  // Count "valid" samples; for now, treat all as valid.
  // You can refine this if you want an explicit "outside mesh" check.
  std::size_t valid_count = 0;
  for (double v : values) {
    if (!std::isnan(v)) ++valid_count;
  }
  return valid_count;
}

std::size_t sample_along_ray(const std::vector<Position>& positions,
                             std::vector<double>& temperatures)
{
  return sample_at_points(positions, temperatures);
}

} // namespace petsc_fe
} // namespace openmc
