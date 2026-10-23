"""Group-aware splits (one lesion never spans two splits) and leakage checks."""
from collections import Counter
from itertools import combinations
import re

import numpy as np

ISIC_ID = re.compile(r"ISIC_\d{7}")


def isic_id(name):
    """Extract the ISIC image ID from a file name or sample ID, ignoring suffixes like _v1 or _segmentation."""
    match = ISIC_ID.search(str(name))
    return match.group(0) if match else None


def grouped_split(groups, fractions, seed):
    """Assign whole groups to splits so that image counts follow ``fractions``.

    ``groups`` holds one group key (e.g. lesion_id) per image; ``fractions`` maps split name to
    its share of images, in assignment order. Returns one split name per image.
    """
    names = list(fractions)
    weights = np.array([fractions[name] for name in names], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {weights.sum()}")
    groups = [str(group) for group in groups]
    sizes = Counter(groups)
    unique = sorted(sizes)
    bounds = np.cumsum(weights) * len(groups)
    assigned = {}
    cumulative = 0
    for index in np.random.default_rng(seed).permutation(len(unique)):
        group = unique[index]
        split_index = min(int(np.searchsorted(bounds, cumulative, side="right")), len(names) - 1)
        assigned[group] = names[split_index]
        cumulative += sizes[group]
    return [assigned[group] for group in groups]


def group_leaks(groups, splits):
    """Groups that appear in more than one split, mapped to the splits they appear in."""
    seen = {}
    for group, split in zip(groups, splits):
        seen.setdefault(str(group), set()).add(split)
    return {group: sorted(found) for group, found in sorted(seen.items()) if len(found) > 1}


def id_overlaps(ids_by_split):
    """Normalized ISIC IDs shared between any two splits (or datasets)."""
    normalized = {
        split: {isic_id(name) for name in names} - {None} for split, names in ids_by_split.items()
    }
    overlaps = {}
    for first, second in combinations(sorted(normalized), 2):
        common = normalized[first] & normalized[second]
        if common:
            overlaps[(first, second)] = sorted(common)
    return overlaps
