from collections import Counter

import numpy as np
import pytest

from artigate.data.split import group_leaks, grouped_split, id_overlaps, isic_id

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def lesion_groups(seed=0, lesions=400):
    sizes = np.random.default_rng(seed).integers(1, 5, lesions)
    return [f"HAM_{lesion:07d}" for lesion, size in enumerate(sizes) for _ in range(size)]


def test_grouped_split_keeps_lesions_together_and_tracks_fractions():
    groups = lesion_groups()
    splits = grouped_split(groups, FRACTIONS, seed=0)
    assert group_leaks(groups, splits) == {}
    counts = Counter(splits)
    for split, fraction in FRACTIONS.items():
        assert counts[split] / len(groups) == pytest.approx(fraction, abs=0.03)


def test_grouped_split_is_seeded():
    groups = lesion_groups()
    assert grouped_split(groups, FRACTIONS, seed=1) == grouped_split(groups, FRACTIONS, seed=1)
    assert grouped_split(groups, FRACTIONS, seed=1) != grouped_split(groups, FRACTIONS, seed=2)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        grouped_split(["a", "b"], {"train": 0.8, "test": 0.1}, seed=0)


def test_group_leaks_reports_groups_in_several_splits():
    assert group_leaks(["a", "a", "b"], ["train", "test", "train"]) == {"a": ["test", "train"]}


def test_isic_ids_are_normalized_before_overlap_checks():
    assert isic_id("ISIC_0024306_v7.jpg") == "ISIC_0024306"
    assert isic_id("IMD002") is None
    overlaps = id_overlaps({"train": ["ISIC_0000001_v1", "ISIC_0000002_v7"], "test": ["ISIC_0000001_v1", "IMD002"]})
    assert overlaps == {("test", "train"): ["ISIC_0000001"]}
