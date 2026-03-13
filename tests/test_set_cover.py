from buildstock_query.tools.set_cover import SetCoverSolver
from buildstock_query.tools.upgrades_analyzer import UpgradesAnalyzer
import pytest
import pandas as pd


class TestSetCoverSolver:
    def test_greedy_hitting_set(self):
        # Test case: Basic functionality
        building_groups = [
            {1, 2, 3, 8},  # Group 1
            {2, 4, 5},  # Group 2
            {3, 5, 6},  # Group 3
            {6, 7, 8},  # Group 4
            {1, 8, 9},  # Group 5
        ]

        solver = SetCoverSolver(groups=building_groups)

        assert [8, 5] == solver.get_greedy_hitting_set()
        # The preffered items doesn't influence tie-breaking and the final result
        assert [8, 5] == solver.get_greedy_hitting_set(current_list=[1, 5, 7])

        # Test case: Empty input
        solver = SetCoverSolver(groups=[])
        assert solver.get_greedy_hitting_set() == []

        # Test case: Input with empty sets (empty sets are ignored)
        building_groups_with_empty = [set(), {1, 2}, set(), {3, 4}]
        solver = SetCoverSolver(groups=building_groups_with_empty)
        assert [4, 2] == solver.get_greedy_hitting_set()

        # Test case: Disjoint sets requiring multiple buildings
        disjoint_groups = [
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8},
            {9, 10},
        ]
        solver = SetCoverSolver(groups=disjoint_groups)
        assert [10, 8, 6, 4, 2] == solver.get_greedy_hitting_set()
        # Algorithm should honor the preferred items when feasible
        assert [10, 8, 6, 4, 2] == solver.get_greedy_hitting_set(current_list=[10, 8, 6, 4, 2])
        assert [10, 8, 6, 3, 1] == solver.get_greedy_hitting_set(current_list=[10, 8, 6, 3, 1])
        assert [1, 3, 5, 7, 9] == solver.get_greedy_hitting_set(current_list=[1, 3, 5, 7, 9])
        assert [1, 3, 5, 8, 10] == solver.get_greedy_hitting_set(current_list=[1, 3, 5, 8, 10])
        # Even if the preferred items are not feasible, the algorithm should still find a minimal hitting set
        assert [1, 3, 5, 10, 8] == solver.get_greedy_hitting_set(current_list=[1, 2, 3, 4, 5])

        # Test case where greedy algorithm is not optimal
        building_groups = [
            {1, 10, 4},
            {1, 9, 4},
            {1, 8},
            {2, 7},
            {2, 6, 4},
            {2, 5, 4},
        ]
        solver = SetCoverSolver(groups=building_groups)
        assert [4, 2, 1] == solver.get_greedy_hitting_set()  # [2, 1] would have been enough

    def test_refine_minimal_set(self):
        """"""
        groups = [
            {1, 2, 3, 8},  # Group 1
            {2, 4, 5},  # Group 2
            {3, 5, 6},  # Group 3
            {6, 7, 8},  # Group 4
            {1, 8, 9},  # Group 5
        ]
        solver = SetCoverSolver(groups=groups)
        # Starting list has extra items; algorithm should pare it down to the minimal hitting set
        assert [8, 5] == solver.refine_minimal_set(current_list=[8, 5, 4])

        # Starting list is already minimal – nothing should change
        assert [8, 5] == solver.refine_minimal_set(current_list=[8, 5])

        # Starting list covers only some groups; refine should add as few items as needed
        assert [8, 5] == solver.refine_minimal_set(current_list=[8])
        assert [2, 8, 6] == solver.refine_minimal_set(current_list=[2])
        assert [1, 7, 5] == solver.refine_minimal_set(current_list=[1, 7])
        assert [3, 8, 5] == solver.refine_minimal_set(current_list=[3, 8])
        assert [1, 2, 6] == solver.refine_minimal_set(current_list=[1, 2, 3, 6])

        # Empty starting list should raise an error
        with pytest.raises(ValueError):
            solver.refine_minimal_set([])

    def test_find_minimal_set(self):
        """Test finding minimal set using both greedy and refined algorithms."""
        # Test case where refine does better than greedy
        groups = [
            {1, 10, 4},
            {1, 9, 4},
            {1, 8},
            {2, 7},
            {2, 6, 4},
            {2, 5, 4},
        ]
        solver = SetCoverSolver(groups=groups)
        assert [4, 2, 1] == solver.get_greedy_hitting_set(current_list=[2])
        assert [2, 1] == solver.refine_minimal_set(current_list=[2])
        assert [4, 2, 1] == solver.find_minimal_set()
        assert [2, 1] == solver.find_minimal_set(current_list=[2])

        # Test case where greedy does better than refine
        # ------------------------------------------------
        # Here `current_list` starts with an item that does **not** appear in any group.  The
        # refine-based approach will therefore need to *add* every item found by the greedy
        # algorithm and will end up longer, so `find_minimal_set` must prefer the greedy
        # solution.
        groups = [{1}, {2}]
        solver = SetCoverSolver(groups=groups)
        # Sanity-check helpers return what we expect
        assert [2, 1] == solver.get_greedy_hitting_set()  # no preferred list
        assert [3, 2, 1] == solver.refine_minimal_set(current_list=[3])
        # `find_minimal_set` should therefore return the shorter greedy result
        assert [2, 1] == solver.find_minimal_set(current_list=[3])

        # Test case where both algorithms return the same *length* but different items
        # ---------------------------------------------------------------------------
        # There are multiple optimal 2-item covers.  Greedy picks one based on its own
        # heuristics; refine preserves the order of the supplied list.  Because the lengths
        # are the same, `find_minimal_set` should prefer the refined result.
        groups = [{1, 2}, {3, 4}]
        solver = SetCoverSolver(groups=groups)
        greedy_res = solver.get_greedy_hitting_set()  # expected [4, 2]
        refined_res = solver.refine_minimal_set(current_list=[1, 3])  # [1, 3]
        assert len(greedy_res) == len(refined_res) == 2  # both minimal-length covers
        # `find_minimal_set` should follow the refined result when lengths tie
        assert refined_res == solver.find_minimal_set(current_list=[1, 3])

        # Test case where no `current_list` is provided – defaults to greedy algorithm
        # ---------------------------------------------------------------------------
        groups = [{1, 2, 3}, {3, 4, 5}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        assert solver.find_minimal_set() == solver.get_greedy_hitting_set()

    # --- Additional edge-case tests ----------------------------------------------------

    def test_all_common_item(self):
        """All groups share a common item - minimal set should be that single item."""
        groups = [{1, 2}, {1, 3}, {1, 4}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set() == [1]
        assert solver.refine_minimal_set(current_list=[1]) == [1]

    def test_duplicate_items_in_group(self):
        """Duplicates inside an input group must not affect the result."""
        groups = [{1, 1, 2}, {2, 3}]  # duplicates collapse when turned into set
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set() == [2]

    def test_preferred_already_covers(self):
        """If preferred list already hits all groups, algorithm should immediately return from it."""
        groups = [{8, 4}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set(current_list=[8, 5]) == [8, 5]

    def test_preferred_with_nonexistent_items(self):
        """Non-existent preferred items should be ignored, algorithm still finds minimal set."""
        groups = [{8, 4}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set(current_list=[8, 44, 53]) == [8, 6]

    def test_single_large_group(self):
        """One large group should be solved with a single picked item."""
        universe = list(range(1, 101))
        groups = [set(universe)]
        solver = SetCoverSolver(groups=groups)
        res = solver.get_greedy_hitting_set()
        assert len(res) == 1 and res[0] in universe

    def test_minimal_set_empty_groups(self):
        """An empty list of groups should result in an empty minimal set regardless of method."""
        solver = SetCoverSolver(groups=[])
        assert solver.find_minimal_set() == []

    def test_minimal_set_duplicate_preferred(self):
        """`current_list` may contain duplicate items; they must be handled gracefully."""
        groups = [{1, 2}, {2, 3}]
        solver = SetCoverSolver(groups=groups)
        # The duplicates in preferred list should be ignored but 2 already covers both groups.
        assert [2] == solver.find_minimal_set(current_list=[2, 2, 2])

    def test_tie_break_stability(self):
        """When multiple items tie, algorithm still returns a single valid item deterministically."""
        groups = [[2, 1], [1, 2]]
        solver = SetCoverSolver(groups=groups)
        # Both 2 and 1 is valid, but due initial sorting of the groups, 2 should be returned
        assert [2] == solver.get_greedy_hitting_set()

    # --- Preference weights tests ---------------------------------------------------

    def test_preference_weights_basic(self):
        """With disjoint groups, preference_weights should steer toward higher-weighted items."""
        groups = [{1, 2}, {3, 4}]
        solver = SetCoverSolver(groups=groups)
        # Without preference_weights: default picks highest (2, 4) via popitem
        assert [4, 2] == solver.get_greedy_hitting_set()
        # With negative weights penalizing 2 and 4, should prefer 1 and 3
        weights = {2: (-1, 0), 4: (-1, 0)}
        result = solver.get_greedy_hitting_set(preference_weights=weights)
        assert 1 in result and 3 in result
        assert len(result) == 2

    def test_preference_weights_ranking(self):
        """Among penalized items, the one with the higher weight tuple should be preferred."""
        # All items in one group — only one pick needed
        groups = [{1, 2, 3}]
        solver = SetCoverSolver(groups=groups)
        # Item 1: low weight (important char match), item 2: slightly higher weight (less important char)
        # Item 3: no weight entry (implicit (0, 0) — highest)
        weights = {1: (-1, 0), 2: (-1, 1)}
        # Should pick item 3 (highest weight — no penalty)
        assert [3] == solver.get_greedy_hitting_set(preference_weights=weights)

        # If all items have weights, pick the highest-weighted one
        weights_all = {1: (-1, 0), 2: (-1, 1), 3: (-2, 1)}
        # Item 2 has weight (-1, 1) which is greater than item 1's (-1, 0), and item 3's (-2, 1)
        assert [2] == solver.get_greedy_hitting_set(preference_weights=weights_all)

    def test_preference_weights_preferred_wins(self):
        """Preferred items (current_list) should take priority over preference_weights."""
        groups = [{1, 2}, {3, 4}]
        solver = SetCoverSolver(groups=groups)
        # Item 2 is penalized but also preferred — preferred should win
        weights = {2: (-1, 0)}
        result = solver.get_greedy_hitting_set(current_list=[2, 3], preference_weights=weights)
        assert result == [2, 3]

    def test_preference_weights_no_effect_on_coverage(self):
        """Preference weights should never cause worse coverage."""
        groups = [{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {5, 6}]
        solver = SetCoverSolver(groups=groups)
        # Give item 3 a very low weight — it's the best greedy pick but should still be chosen
        # if no alternative covers as many groups
        weights = {3: (-3, 3)}
        result = solver.get_greedy_hitting_set(preference_weights=weights)
        # Verify all groups are covered
        result_set = set(result)
        for group in groups:
            assert result_set & group, f"Group {group} not covered by {result}"

    def test_preference_weights_refine_removal_order(self):
        """refine_minimal_set with preference_weights should try removing least-preferred items first."""
        groups = [{1, 2}, {2, 3}]
        solver = SetCoverSolver(groups=groups)
        # current_list covers all groups, has redundancy
        # Item 1 has low weight — should be tried for removal first
        weights = {1: (-1, 0)}
        result = solver.refine_minimal_set(current_list=[1, 2, 3], preference_weights=weights)
        # Item 2 alone covers both groups, so 1 and 3 can be removed
        assert result == [2]

    def test_preference_weights_refine_keeps_needed_penalized(self):
        """refine_minimal_set should keep low-weight items if they're essential for coverage."""
        groups = [{1}, {2}]
        solver = SetCoverSolver(groups=groups)
        # Both items are essential — low weights don't cause removal
        weights = {1: (-1, 0), 2: (-1, 0)}
        result = solver.refine_minimal_set(current_list=[1, 2], preference_weights=weights)
        assert set(result) == {1, 2}

    def test_preference_weights_none(self):
        """Passing preference_weights=None should produce identical results to not passing it."""
        groups = [{1, 2, 3, 8}, {2, 4, 5}, {3, 5, 6}, {6, 7, 8}, {1, 8, 9}]
        solver = SetCoverSolver(groups=groups)
        assert solver.get_greedy_hitting_set() == solver.get_greedy_hitting_set(preference_weights=None)
        assert solver.find_minimal_set() == solver.find_minimal_set(preference_weights=None)

    def test_preference_weights_find_minimal_set(self):
        """find_minimal_set should pass preference_weights through to both greedy and refine paths."""
        groups = [{1, 2}, {3, 4}]
        solver = SetCoverSolver(groups=groups)
        weights = {2: (-1, 0), 4: (-1, 0)}
        result = solver.find_minimal_set(preference_weights=weights)
        # Should prefer non-penalized items (higher weight)
        assert 1 in result and 3 in result


class TestComputePreferenceWeights:
    """Tests for UpgradesAnalyzer.compute_preference_weights using a minimal mock."""

    @pytest.fixture
    def analyzer(self):
        """Create a minimal UpgradesAnalyzer-like object with just buildstock_df."""
        # We only need buildstock_df for compute_preference_weights, so mock the rest
        obj = object.__new__(UpgradesAnalyzer)
        obj.buildstock_df = pd.DataFrame(
            {
                "heating fuel": ["Gas", "Wood", "Electric", "Wood", "Gas"],
                "geometry": ["SFD", "MH", "SFD", "MH", "MH"],
                "vintage": ["1980s", "1990s", "2000s", "1980s", "1980s"],
            },
            index=pd.Index([1, 2, 3, 4, 5], name="building_id"),
        )
        return obj

    def test_single_avoid(self, analyzer):
        """Single avoid char should produce weights for matching buildings only."""
        result = analyzer.compute_preference_weights([("heating fuel", "Wood")])
        # Buildings 2 and 4 have Wood heating
        assert set(result.keys()) == {2, 4}
        # 1 match at index 0 → (-1, 0)
        assert result[2] == (-1, 0)
        assert result[4] == (-1, 0)

    def test_multiple_avoids(self, analyzer):
        """Multiple avoid chars should accumulate match counts and index sums."""
        result = analyzer.compute_preference_weights([("heating fuel", "Wood"), ("geometry", "MH")])
        # Building 2: Wood (idx 0) + MH (idx 1) → (-2, 0+1) = (-2, 1)
        assert result[2] == (-2, 1)
        # Building 4: Wood (idx 0) + MH (idx 1) → (-2, 1)
        assert result[4] == (-2, 1)
        # Building 5: MH only (idx 1) → (-1, 1)
        assert result[5] == (-1, 1)
        # Buildings 1, 3: no match → not in dict
        assert 1 not in result
        assert 3 not in result

    def test_ordering_matters(self, analyzer):
        """First-listed avoid char should produce lower weight than later ones."""
        # Wood first (more important to avoid)
        result_wood_first = analyzer.compute_preference_weights([("heating fuel", "Wood"), ("geometry", "MH")])
        # MH first (more important to avoid)
        result_mh_first = analyzer.compute_preference_weights([("geometry", "MH"), ("heating fuel", "Wood")])
        # Building 5 only matches MH.
        # Wood-first: MH at idx 1 → (-1, 1). MH-first: MH at idx 0 → (-1, 0)
        # (-1, 1) > (-1, 0), so building 5 is MORE preferred when Wood is listed first
        assert result_wood_first[5] > result_mh_first[5]

    def test_no_matches(self, analyzer):
        """If no buildings match, return empty dict."""
        result = analyzer.compute_preference_weights([("heating fuel", "Oil")])
        assert result == {}

    def test_empty_avoid_list(self, analyzer):
        """Empty avoid list should return empty dict."""
        result = analyzer.compute_preference_weights([])
        assert result == {}
