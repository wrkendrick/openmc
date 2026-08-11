#ifndef OPENMC_TALLIES_FILTER_DELAYEDGROUPBORN_H
#define OPENMC_TALLIES_FILTER_DELAYEDGROUPBORN_H

#include <string>
#include <unordered_map>

#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Specifies which delayed group, if any, the particle itself was born into.
//!
//! Unlike DelayedGroupFilter (which bins newly-created fission progeny by
//! the family they are being born into), this filter bins the currently
//! transporting particle by the family it was itself born as -- bin 0
//! denotes a prompt (non-delayed) particle.
//==============================================================================

class DelayedGroupBornFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~DelayedGroupBornFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "delayedgroupborn"; }
  FilterType type() const override { return FilterType::DELAYED_GROUP_BORN; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<int>& groups() const { return groups_; }

  void set_groups(span<int> groups);

private:
  //----------------------------------------------------------------------------
  // Data members

  //! Delayed group values for each bin. 0 denotes prompt (non-delayed).
  vector<int> groups_;

  //! Map from delayed group value to filter bin index.
  std::unordered_map<int, int> map_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_DELAYEDGROUPBORN_H
