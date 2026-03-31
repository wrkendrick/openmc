#include "openmc/libmesh_interface.h"
#include "openmc/error.h"

#ifdef OPENMC_LIBMESH_ENABLED
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe_compute_data.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_type.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/point_locator_tree.h"
#include "libmesh/system.h"
#include "libmesh/mesh_tools.h"
#include "openmc/nanoflann.hpp"
#include <chrono>
#endif

#include <cmath>
#include <limits>

namespace openmc {
namespace libmesh {

#ifdef OPENMC_LIBMESH_ENABLED

namespace {

//============================================================================
// Module state
//============================================================================

std::unique_ptr<libMesh::LibMeshInit> libmesh_init;
std::unique_ptr<libMesh::Mesh> mesh;
std::unique_ptr<libMesh::EquationSystems> equation_systems;
std::unique_ptr<libMesh::PointLocatorTree> point_locator;

std::string solution_filepath;
std::string system_name;
std::string variable_name;

const libMesh::System* system_ptr = nullptr;
const libMesh::DofMap* dof_map_ptr = nullptr;
const libMesh::NumericVector<libMesh::Number>* solution_ptr = nullptr;
libMesh::FEType fe_type;
unsigned int variable_number = 0;

bool solution_loaded = false;

//============================================================================
// Nanoflann kd-tree for element centroid lookup
//============================================================================

// Stores element centroids and pointers for kd-tree queries
struct ElemCentroidCloud {
  struct CentroidEntry {
    double x, y, z;
    const libMesh::Elem* elem;
  };

  std::vector<CentroidEntry> entries;

  // nanoflann adaptor interface
  inline size_t kdtree_get_point_count() const { return entries.size(); }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const
  {
    if (dim == 0) return entries[idx].x;
    if (dim == 1) return entries[idx].y;
    return entries[idx].z;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

using ElemKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, ElemCentroidCloud>,
    ElemCentroidCloud,
    3,       // 3D
    size_t   // index type
>;

// Module-level kd-tree state
std::unique_ptr<ElemCentroidCloud> elem_cloud;
std::unique_ptr<ElemKDTree> elem_kdtree;

// How many nearest centroids to check for containment.
// For well-shaped hex/tet meshes, the correct element is almost always
// among the closest 1-3 centroids. We check a few extras for safety.
constexpr size_t KNN_CANDIDATES = 1;

// Build the kd-tree from the current mesh
void build_elem_kdtree()
{
  elem_cloud = std::make_unique<ElemCentroidCloud>();

  // Reserve space
  elem_cloud->entries.reserve(mesh->n_active_local_elem());

  for (const auto* elem : mesh->active_local_element_ptr_range()) {
    libMesh::Point c = elem->true_centroid();
    elem_cloud->entries.push_back({c(0), c(1), c(2), elem});
  }

  elem_kdtree = std::make_unique<ElemKDTree>(
      3 /* dim */, *elem_cloud,
      nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf size */));

  std::cout << "[nanoflann] Built kd-tree with "
            << elem_cloud->entries.size() << " element centroids.\n";
}

/// Locate the element containing point p using the nanoflann kd-tree.
/// Falls back to the libMesh PointLocatorTree if the kd-tree candidates
/// don't contain the point (e.g. near mesh boundaries or curved elements).
const libMesh::Elem* locate_elem_kdtree(const libMesh::Point& p)
{
  double query[3] = {p(0), p(1), p(2)};

  // knnSearch returns indices into elem_cloud->entries
  std::array<size_t, KNN_CANDIDATES> ret_indices;
  std::array<double, KNN_CANDIDATES> ret_dists_sq;

  nanoflann::KNNResultSet<double> result_set(KNN_CANDIDATES);
  result_set.init(ret_indices.data(), ret_dists_sq.data());
  elem_kdtree->findNeighbors(result_set, query);

  // Check each candidate element for actual containment
  for (size_t k = 0; k < result_set.size(); ++k) {
    const libMesh::Elem* candidate = elem_cloud->entries[ret_indices[k]].elem;
    if (candidate->contains_point(p)) {
      return candidate;
    }
  }

  // Fallback: nanoflann candidates didn't contain the point.
  // This can happen for points near mesh boundaries, in curved elements,
  // or in concave regions where the nearest centroid isn't the host element.
  return (*point_locator)(p);
}

//============================================================================
// Timing counters
//============================================================================
static double time_point_locate = 0;
static double time_inverse_map = 0;
static double time_dof_indices = 0;
static double time_fe_compute = 0;
static double time_dot_product = 0;
static int sample_count = 0;

// Nanoflann-specific stats
static int nanoflann_hits = 0;    // found via kd-tree candidates
static int nanoflann_misses = 0;  // fell back to point_locator

//============================================================================
// Core sampling implementation
//============================================================================

double sample_impl(const libMesh::Point& p)
{
  using Clock = std::chrono::high_resolution_clock;

  auto t0 = Clock::now();

  // --- Element location via nanoflann kd-tree ---
  const libMesh::Elem* elem = nullptr;

  if (elem_kdtree) {
    double query[3] = {p(0), p(1), p(2)};
    std::array<size_t, KNN_CANDIDATES> ret_indices;
    std::array<double, KNN_CANDIDATES> ret_dists_sq;

    nanoflann::KNNResultSet<double> result_set(KNN_CANDIDATES);
    result_set.init(ret_indices.data(), ret_dists_sq.data());
    elem_kdtree->findNeighbors(result_set, query);

    for (size_t k = 0; k < result_set.size(); ++k) {
      const libMesh::Elem* candidate =
          elem_cloud->entries[ret_indices[k]].elem;
      if (candidate->contains_point(p)) {
        elem = candidate;
        ++nanoflann_hits;
        break;
      }
    }

    if (!elem) {
      // Fallback to full tree search
      elem = (*point_locator)(p);
      ++nanoflann_misses;
    }
  } else {
    // No kd-tree built yet, use original locator
    elem = (*point_locator)(p);
  }

  auto t1 = Clock::now();
  time_point_locate += std::chrono::duration<double>(t1 - t0).count();

  if (!elem) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  auto t2 = Clock::now();
  libMesh::Point ref_point =
      libMesh::FEInterface::inverse_map(elem->dim(), fe_type, elem, p);
  auto t3 = Clock::now();
  time_inverse_map += std::chrono::duration<double>(t3 - t2).count();

  std::vector<libMesh::dof_id_type> dof_indices;
  dof_map_ptr->dof_indices(elem, dof_indices, variable_number);
  auto t4 = Clock::now();
  time_dof_indices += std::chrono::duration<double>(t4 - t3).count();

  libMesh::FEComputeData fe_data(*equation_systems, ref_point);
  libMesh::FEInterface::compute_data(elem->dim(), fe_type, elem, fe_data);
  auto t5 = Clock::now();
  time_fe_compute += std::chrono::duration<double>(t5 - t4).count();

  double value = 0.0;
  for (std::size_t i = 0; i < dof_indices.size(); ++i) {
    value += fe_data.shape[i] * (*solution_ptr)(dof_indices[i]);
  }
  auto t6 = Clock::now();
  time_dot_product += std::chrono::duration<double>(t6 - t5).count();

  sample_count++;
  if (sample_count % 50000 == 0) {
    std::cout << "===== LIBMESH SAMPLE_IMPL TIMING (" << sample_count
              << " calls) =====\n";
    std::cout << "Point locate:  "
              << (time_point_locate / sample_count) * 1e6 << " us\n";
    std::cout << "Inverse map:   "
              << (time_inverse_map / sample_count) * 1e6 << " us\n";
    std::cout << "DOF indices:   "
              << (time_dof_indices / sample_count) * 1e6 << " us\n";
    std::cout << "FE compute:    "
              << (time_fe_compute / sample_count) * 1e6 << " us\n";
    std::cout << "Dot product:   "
              << (time_dot_product / sample_count) * 1e6 << " us\n";
    std::cout << "Total:         "
              << ((time_point_locate + time_inverse_map + time_dof_indices +
                   time_fe_compute + time_dot_product) /
                  sample_count) *
                     1e6
              << " us\n";
    std::cout << "Nanoflann hits/misses: " << nanoflann_hits << "/"
              << nanoflann_misses << " ("
              << (100.0 * nanoflann_hits /
                  std::max(1, nanoflann_hits + nanoflann_misses))
              << "% hit rate)\n";
    std::cout << "=================================================\n";
  }

  return value;
}

} // anonymous namespace

//============================================================================
// Public interface
//============================================================================

void load_solution(const std::string& filepath, const std::string& sys_name,
                   const std::string& var_name)
{
  // Initialize libMesh if not already initialized
  if (!libmesh_init) {
    int argc = 1;
    const char* argv[] = {"openmc", nullptr};
    libmesh_init = std::make_unique<libMesh::LibMeshInit>(
        argc, const_cast<char**>(argv));
  }

  if (!libMesh::initialized()) {
    fatal_error("Failed to initialize libMesh.");
  }

  if (solution_loaded) {
    unload_solution();
  }

  solution_filepath = filepath;
  system_name = sys_name;
  variable_name = var_name;

  mesh = std::make_unique<libMesh::Mesh>(libmesh_init->comm());
  mesh->read("/home/william/Documents/PostDoc/Research/FE_coupled_sampling/"
             "moose_run/constant_900K_case/moose_xda_0000_mesh.xda");

  equation_systems = std::make_unique<libMesh::EquationSystems>(*mesh);
  equation_systems->read(filepath,
                         libMesh::EquationSystems::READ_HEADER |
                             libMesh::EquationSystems::READ_DATA |
                             libMesh::EquationSystems::READ_ADDITIONAL_DATA);

  system_ptr = &equation_systems->get_system(system_name);
  dof_map_ptr = &system_ptr->get_dof_map();
  solution_ptr = system_ptr->current_local_solution.get();
  variable_number = system_ptr->variable_number(variable_name);
  fe_type = dof_map_ptr->variable_type(variable_number);

  point_locator = std::make_unique<libMesh::PointLocatorTree>(*mesh);
  point_locator->enable_out_of_mesh_mode();

  // Build nanoflann kd-tree of element centroids
  build_elem_kdtree();

  // Debug check
  auto bbox = libMesh::MeshTools::create_bounding_box(*mesh);
  std::cout << "LibMesh bbox X: [" << bbox.min()(0) << ", " << bbox.max()(0)
            << "]\n";
  std::cout << "LibMesh bbox Y: [" << bbox.min()(1) << ", " << bbox.max()(1)
            << "]\n";
  std::cout << "LibMesh bbox Z: [" << bbox.min()(2) << ", " << bbox.max()(2)
            << "]\n";

  solution_loaded = true;
}

void unload_solution()
{
  if (!solution_loaded)
    return;

  system_ptr = nullptr;
  dof_map_ptr = nullptr;
  solution_ptr = nullptr;

  elem_kdtree.reset();
  elem_cloud.reset();
  point_locator.reset();
  equation_systems.reset();
  mesh.reset();

  solution_loaded = false;
}

void reload_solution()
{
  if (!solution_loaded) {
    fatal_error("No solution to reload.");
  }

  equation_systems->read(solution_filepath,
                         libMesh::EquationSystems::READ_DATA);
  solution_ptr = system_ptr->current_local_solution.get();

  // Note: kd-tree does NOT need rebuilding on solution reload —
  // the mesh topology and element centroids haven't changed,
  // only the solution DOF values have.
}

std::size_t sample_along_ray(const std::vector<Position>& positions,
                             std::vector<double>& temperatures)
{
  if (!solution_loaded) {
    fatal_error("No solution loaded. Call libmesh::load_solution() first.");
  }

  temperatures.resize(positions.size());
  std::size_t valid_count = 0;

  for (std::size_t i = 0; i < positions.size(); ++i) {
    temperatures[i] = sample_impl(
        libMesh::Point(positions[i].x, positions[i].y, positions[i].z));
    if (!std::isnan(temperatures[i])) {
      ++valid_count;
    }
  }
  return valid_count;
}

template <std::size_t N>
std::size_t sample_along_ray(const std::array<Position, N>& positions,
                             std::array<double, N>& temperatures)
{
  std::size_t valid_count = 0;
  for (std::size_t i = 0; i < N; ++i) {
    temperatures[i] = sample_impl(
        libMesh::Point(positions[i].x, positions[i].y, positions[i].z));
    if (!std::isnan(temperatures[i])) {
      ++valid_count;
    }
  }
  return valid_count;
}

// Explicit instantiation for N_SAMPLES = 5
template std::size_t sample_along_ray<5>(const std::array<Position, 5>&,
                                         std::array<double, 5>&);

bool has_solution() { return solution_loaded; }

// Fast ray-based sampling with neighbor walking
std::size_t sample_along_ray_fast(const Position& start, const Direction& dir,
                                  const std::array<double, 5>& distances,
                                  std::array<double, 5>& temperatures)
{
  if (!solution_loaded) {
    fatal_error("No solution loaded.");
  }

  // Find starting element once
  libMesh::Point p0(start.x, start.y, start.z);
  const libMesh::Elem* elem = elem_kdtree ? locate_elem_kdtree(p0)
                                           : (*point_locator)(p0);

  if (!elem) {
    for (int i = 0; i < 5; ++i)
      temperatures[i] = std::numeric_limits<double>::quiet_NaN();
    return 0;
  }

  std::size_t valid_count = 0;

  for (int i = 0; i < 5; ++i) {
    libMesh::Point p(start.x + distances[i] * dir.x,
                     start.y + distances[i] * dir.y,
                     start.z + distances[i] * dir.z);

    // Check if still in same element
    if (!elem->contains_point(p)) {
      // Search neighbors first
      bool found = false;
      for (unsigned int n = 0; n < elem->n_neighbors(); ++n) {
        const libMesh::Elem* neighbor = elem->neighbor_ptr(n);
        if (neighbor && neighbor->contains_point(p)) {
          elem = neighbor;
          found = true;
          break;
        }
      }
      // Fall back to kd-tree or full search
      if (!found) {
        elem = elem_kdtree ? locate_elem_kdtree(p) : (*point_locator)(p);
        if (!elem) {
          temperatures[i] = std::numeric_limits<double>::quiet_NaN();
          continue;
        }
      }
    }

    // Sample at this point
    libMesh::Point ref_point =
        libMesh::FEInterface::inverse_map(elem->dim(), fe_type, elem, p);

    std::vector<libMesh::dof_id_type> dof_indices;
    dof_map_ptr->dof_indices(elem, dof_indices, variable_number);

    libMesh::FEComputeData fe_data(*equation_systems, ref_point);
    libMesh::FEInterface::compute_data(elem->dim(), fe_type, elem, fe_data);

    double value = 0.0;
    for (std::size_t j = 0; j < dof_indices.size(); ++j) {
      value += fe_data.shape[j] * (*solution_ptr)(dof_indices[j]);
    }

    temperatures[i] = value;
    ++valid_count;
  }

  return valid_count;
}

#else // OPENMC_LIBMESH_ENABLED not defined

void load_solution(const std::string&, const std::string&, const std::string&)
{
  fatal_error("OpenMC was not built with libMesh support.");
}

void unload_solution() {}

void reload_solution()
{
  fatal_error("OpenMC was not built with libMesh support.");
}

std::size_t sample_along_ray(const std::vector<Position>&,
                             std::vector<double>&)
{
  fatal_error("OpenMC was not built with libMesh support.");
  return 0;
}

bool has_solution() { return false; }

#endif // OPENMC_LIBMESH_ENABLED

} // namespace libmesh
} // namespace openmc