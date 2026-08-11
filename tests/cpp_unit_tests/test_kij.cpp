#include "openmc/particle.h"
#include "openmc/tallies/filter_delayedgroupborn.h"
#include "openmc/tallies/filter_match.h"
#include <catch2/catch_test_macros.hpp>

using namespace openmc;

TEST_CASE("Test DelayedGroupBornFilter bin matching")
{
  DelayedGroupBornFilter filter;
  vector<int> groups = {0, 1, 2, 3, 4, 5, 6};
  filter.set_groups(groups);

  Particle p;
  FilterMatch match;

  // Prompt (bin 0) should match
  p.delayed_group() = 0;
  filter.get_all_bins(p, TallyEstimator::ANALOG, match);
  REQUIRE(match.bins_.size() == 1);
  REQUIRE(match.bins_[0] == 0);
  REQUIRE(match.weights_[0] == 1.0);

  // A delayed family in the filter's bin list should match its bin index
  match.bins_.clear();
  match.weights_.clear();
  p.delayed_group() = 3;
  filter.get_all_bins(p, TallyEstimator::ANALOG, match);
  REQUIRE(match.bins_.size() == 1);
  REQUIRE(match.bins_[0] == 3);

  // A delayed group value outside the filter's configured bins -> no match
  match.bins_.clear();
  match.weights_.clear();
  p.delayed_group() = 7;
  filter.get_all_bins(p, TallyEstimator::ANALOG, match);
  REQUIRE(match.bins_.empty());
}

TEST_CASE("Test DelayedGroupBornFilter accepts prompt (0) but rejects "
          "out-of-range bins")
{
  DelayedGroupBornFilter filter;

  // 0 (prompt) is valid here, unlike DelayedGroupFilter
  vector<int> ok = {0, 1};
  REQUIRE_NOTHROW(filter.set_groups(ok));
  REQUIRE(filter.groups().size() == 2);

  vector<int> negative = {-1};
  REQUIRE_THROWS_AS(filter.set_groups(negative), std::invalid_argument);

  vector<int> too_big = {MAX_DELAYED_GROUPS + 1};
  REQUIRE_THROWS_AS(filter.set_groups(too_big), std::invalid_argument);
}
