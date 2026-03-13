# set_cover_solver.py

import logging
from collections import defaultdict
from typing import TypeVar, Generic, Protocol
from typing_extensions import Self
from abc import abstractmethod
from collections.abc import Iterable, Collection, Hashable, Sequence
import time

logger = logging.getLogger(__name__)


class HashableComparable(Hashable, Protocol):
    @abstractmethod
    def __lt__(self, other: Self, /) -> bool: ...


T = TypeVar("T", bound=HashableComparable)


class SetCoverSolver(Generic[T]):
    """
    Class to solve the hitting set problem.
    """

    def __init__(self, groups: Iterable[Collection[T]], verbose: bool = False):
        """
        Args:
            groups: Iterable[Collection[T]] - The list of groups to hit with the minimal set
            verbose: bool - Whether to print verbose output
        """
        self.verbose = verbose
        # Convert potentially non-sized iterables to lists so len() is safe
        group_list = [tuple(sorted(s)) for s in groups if len(s) > 0]
        self.vprint(f"Initializing SetCoverSolver with {len(group_list)} non-empty groups")
        self.groups_list = group_list
        self.groups_set = [set(s) for s in self.groups_list]
        self.group_sizes: list[int] = [len(g) for g in self.groups_list]
        self.vprint(f"Group sizes: {self.group_sizes}")

    def vprint(self, msg: str):
        if self.verbose:
            logger.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}")

    def find_minimal_set(
        self,
        current_list: list[T] | None = None,
        preference_weights: dict[T, tuple[int, int]] | None = None,
    ) -> list[T]:
        """Find a minimal hitting set using a hybrid greedy + refinement strategy.

        If current_list is provided, also tries refining it and returns whichever result is smaller.

        Args:
            current_list: Previous solution to refine from. If None, uses greedy only.
            preference_weights: Tie-breaking weights. See get_greedy_hitting_set for details.

        Returns:
            list[T]: The minimal hitting set.
        """
        self.vprint("Finding greedy minimal set")
        minimal_set = self.get_greedy_hitting_set(current_list=current_list, preference_weights=preference_weights)
        if current_list:
            self.vprint("Finding refined minimal set")
            refined_minimal_set = self.refine_minimal_set(current_list, preference_weights=preference_weights)
            self.vprint(f"Refined minimal set size: {len(refined_minimal_set)}")
            self.vprint(f"Greedy minimal set size: {len(minimal_set)}")
            if len(refined_minimal_set) <= len(minimal_set):
                self.vprint("Using refined minimal set")
                minimal_set = refined_minimal_set
            else:
                self.vprint("Using greedy minimal set")
        self.vprint(f"Minimal set size: {len(minimal_set)}")
        return minimal_set

    def refine_minimal_set(
        self,
        current_list: Sequence[T],
        preference_weights: dict[T, tuple[int, int]] | None = None,
    ) -> list[T]:
        """Refine current_list by removing redundant items or adding missing ones.

        If all groups are already covered, tries removing items (least-preferred first when
        preference_weights is provided). Otherwise, adds the minimum items needed via greedy."""
        if not current_list:
            raise ValueError("current_list cannot be empty")
        current_items_set = set(current_list)
        current_items_list = list(current_list)
        uncovered_groups = [group for group in self.groups_set if group.isdisjoint(current_items_set)]
        self.vprint(f"{len(uncovered_groups)} of {len(self.groups_set)} groups left uncovered by existing items")
        if len(uncovered_groups) == 0:
            self.vprint("Checking if we can remove any items from the existing set to still cover all groups")
            # Try removing least-preferred items first (lowest weight) to increase chance of dropping them
            if preference_weights:
                removal_order = sorted(current_items_list, key=lambda x: preference_weights.get(x, (0, 0)))
            else:
                removal_order = list(reversed(current_items_list))
            for current_item in removal_order:
                current_items_set.remove(current_item)
                uncovered_groups = [group for group in self.groups_set if group.isdisjoint(current_items_set)]
                if len(uncovered_groups) == 0:
                    self.vprint(f"Removed {current_item} from the existing set - remaining items still hit all groups")
                    continue
                else:  # Add it back because we can't cover all the groups without it
                    current_items_set.add(current_item)
            return [item for item in current_items_list if item in current_items_set]
        self.vprint("Finding the minimal extra items to cover the uncovered groups")
        minimal_extra_items = self.get_greedy_hitting_set(
            groups=uncovered_groups, preference_weights=preference_weights
        )
        return current_items_list + minimal_extra_items

    def get_greedy_hitting_set(
        self,
        groups: Iterable[Collection[T]] | None = None,
        current_list: Sequence[T] | None = None,
        preference_weights: dict[T, tuple[int, int]] | None = None,
    ) -> list[T]:
        """Greedy algorithm to find a minimal hitting set that hits all groups.
        At each step, the greedy algorithm picks an item that hits the most groups out of the remaining groups
        and removes these groups from the list of groups to hit. This process is repeated until all groups are hit.

        1. Initialize a bucket queue with index=hit_count and values=items with this hit_count.
        2. Track which groups are hit by the minimal set. Initially no groups are hit by the minimal set.
        3. Track how many groups each item hits (belongs to).
        4. While there are still groups to hit by the minimal set:
            4.1. Pick an item from the bucket queue with maximum hit_count. That is an item that hits the most groups.
            4.2. Add the item to the minimal set.
            4.3. Remove the item from the bucket for max_hit_count - this item is no longer available for selection.
            4.4. This item likely belonged to many groups. For each group it hit (belonged to):
                4.4.1. Mark that group as hit by the minimal set.
                4.4.2. For every item in that group
                    4.4.2.1. Decrease their hit count by 1 because the hit groups in minimal set don't count.
                             We are only interested in the number of groups that are not yet hit by minimal set when
                             greedily picking the next item to add to the minimal set.
                    4.4.2.2. Move them from higher bucket to lower bucket (because they now hit fewer groups).
        5. Return the minimal set.
        (Note: both item2hit_count mapping and bucket queue contains the same info, but each has its own use because
         of efficiency in information access and update. Specifically, bucket queue is used in step 4.1 to find the
         item with maximum hit count in O(1) time. item2hit_count mapping is used in step 4.4.2.2 to find the bucket
         each item in the currently hit group belongs to and move them one bucket down.)

        Args:
            groups: The groups to hit. If None, uses all groups in the solver.
            current_list: Items preferred during tie-breaking (e.g. a previous solution for stability).
            preference_weights: Soft tie-breaking bias. Maps item -> weight tuple (higher = more preferred).
                Items not in the dict get implicit weight (0, 0). Only affects selection among items
                with equal hit count; never overrides the greedy "pick max coverage" rule.

        Returns:
            A list of items forming a minimal hitting set.

        Examples:
            >>> groups = [{1, 2, 3, 8}, {2, 4, 5}, {3, 5, 6}, {6, 7, 8}, {1, 8, 9}]
            >>> solver = SetCoverSolver(items=list(range(1, 11)), groups=groups)
            >>> solver.get_greedy_hitting_set(groups=groups)
            [8, 5]
        """
        groups_list = list(groups) if groups else self.groups_list
        groups_set = [set(g) for g in groups_list] if groups else self.groups_set

        preferred_items_list = list(current_list) if current_list else []
        num_groups = len(groups_list)

        item2groups: dict[T, list[int]] = defaultdict(list)
        for group_index, group in enumerate(groups_list):
            for item_id in group:
                item2groups[item_id].append(group_index)

        item2hit: dict[T, int] = {item_id: len(groups) for item_id, groups in item2groups.items()}
        max_hit = max(item2hit.values()) if item2hit else 0
        # Bucket Queue to store items with same number of groups (count) they belong to.
        # We are using a dictionary instead of set because dictionary returns elements in repeatable order
        # Sets can return elements in arbitrary order and we want this algorithm to be deterministic
        buckets: list[dict[T, None]] = [{} for _ in range(max_hit + 1)]
        for item_id, hit_count in item2hit.items():
            buckets[hit_count][item_id] = None

        self.vprint(f"Bucket heap initialized with {len(item2hit)} elements, max count: {max_hit}")

        # Track which groups are already hit by the minimal set
        is_group_hit: list[bool] = [False] * num_groups
        remaining = num_groups
        minial_item_list: list[T] = []
        iteration = 0
        current_max_hit = max_hit
        while remaining:
            iteration += 1
            self.vprint(f"{remaining} sets remaining to hit, minimal set size: {len(minial_item_list)}")
            while current_max_hit > 0 and not buckets[current_max_hit]:
                current_max_hit -= 1
            if current_max_hit == 0:
                raise RuntimeError("No item left that fall in any remaining sets.")

            # Priority 1: prefer items from current_list (stability)
            chosen_item = None
            if preferred_items_list:
                candidate_set: set[T] = set(buckets[current_max_hit].keys())
                for c in preferred_items_list:
                    if c in candidate_set:
                        chosen_item = c
                        del buckets[current_max_hit][chosen_item]
                        self.vprint(f"Preferred item {chosen_item} chosen from bucket with {current_max_hit} hits")
                        break

            # Priority 2: among remaining candidates, pick the one with highest preference weight
            if chosen_item is None and preference_weights:
                chosen_item = max(buckets[current_max_hit], key=lambda x: preference_weights.get(x, (0, 0)))
                del buckets[current_max_hit][chosen_item]
                weight = preference_weights.get(chosen_item, (0, 0))
                self.vprint(
                    f"Item {chosen_item} (preference_weight={weight}) chosen from bucket with {current_max_hit} hits"
                )

            # Priority 3: fallback to popitem (last inserted, typically highest building_id)
            if chosen_item is None:
                chosen_item = buckets[current_max_hit].popitem()[0]
                self.vprint(f"Arbitrary item {chosen_item} chosen from bucket with {current_max_hit} hits")

            hit_count = item2hit[chosen_item]
            if hit_count == 0:
                raise RuntimeError("Counter reached zero but some sets remain uncovered.")

            self.vprint(f"Adding item: {chosen_item} to the minimal set")
            minial_item_list.append(chosen_item)

            if self.verbose:
                covered_sets = len([1 for indx in item2groups[chosen_item] if not is_group_hit[indx]])
                bldg_count = sum(
                    [len(groups_list[indx]) for indx in item2groups[chosen_item] if not is_group_hit[indx]]
                )
                self.vprint(
                    f"{covered_sets} sets covered by this item will be removed and ~{bldg_count}"
                    " count of items will be decreased."
                )

            for group_index in item2groups[chosen_item]:
                if is_group_hit[group_index]:
                    continue  # this set was already covered earlier
                is_group_hit[group_index] = True
                remaining -= 1

                for impacted_item in groups_list[group_index]:
                    if impacted_item == chosen_item:
                        continue  # we just chose this item; its counter will be discarded

                    # Update group count of these items and move them to lower bucket
                    current_hit_count = item2hit[impacted_item]
                    new_hit_count = current_hit_count - 1
                    item2hit[impacted_item] = new_hit_count

                    # Move item from higher bucket to new lower bucket
                    if impacted_item in buckets[current_hit_count]:
                        del buckets[current_hit_count][impacted_item]
                    if new_hit_count > 0:
                        buckets[new_hit_count][impacted_item] = None
        self.vprint(f"Finished! Found minimal set of size {len(minial_item_list)}")
        minimal_items_set = set(minial_item_list)
        if not all(s & minimal_items_set for s in groups_set):
            raise RuntimeError("Not every set was hit — bug in algorithm!")
        return minial_item_list
