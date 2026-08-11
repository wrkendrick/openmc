#include "openmc/tallies/filter_delayedgroupborn.h"

#include "openmc/error.h"
#include "openmc/xml_interface.h"

namespace openmc {

void DelayedGroupBornFilter::from_xml(pugi::xml_node node)
{
  auto groups = get_node_array<int>(node, "bins");
  this->set_groups(groups);
}

void DelayedGroupBornFilter::set_groups(span<int> groups)
{
  // Clear existing groups
  groups_.clear();
  groups_.reserve(groups.size());
  map_.clear();

  // Make sure all the group index values are valid. Unlike
  // DelayedGroupFilter, 0 (prompt) is a valid bin here.
  for (auto group : groups) {
    if (group < 0) {
      throw std::invalid_argument {
        "Encountered delayedgroupborn bin with index " +
        std::to_string(group) + " which is less than 0"};
    } else if (group > MAX_DELAYED_GROUPS) {
      throw std::invalid_argument {"Encountered delayedgroupborn bin with "
                                    "index " +
                                    std::to_string(group) +
                                    " which is greater than "
                                    "MAX_DELAYED_GROUPS (" +
                                    std::to_string(MAX_DELAYED_GROUPS) + ")"};
    }
    map_[group] = groups_.size();
    groups_.push_back(group);
  }

  n_bins_ = groups_.size();
}

void DelayedGroupBornFilter::get_all_bins(
  const Particle& p, TallyEstimator estimator, FilterMatch& match) const
{
  auto search = map_.find(p.delayed_group());
  if (search != map_.end()) {
    match.bins_.push_back(search->second);
    match.weights_.push_back(1.0);
  }
}

void DelayedGroupBornFilter::to_statepoint(hid_t filter_group) const
{
  Filter::to_statepoint(filter_group);
  write_dataset(filter_group, "bins", groups_);
}

std::string DelayedGroupBornFilter::text_label(int bin) const
{
  if (groups_[bin] == 0)
    return "Born Prompt";
  return "Born Delayed Group " + std::to_string(groups_[bin]);
}

} // namespace openmc
