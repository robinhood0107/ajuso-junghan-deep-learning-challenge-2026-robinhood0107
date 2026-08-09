from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import pytest

from deep_challenge.splits import (
    SplitManifest,
    SplitPartition,
    SplitValidationError,
    build_group_clusters,
    eligible_training_ids,
    eligible_validation_ids,
    expand_hard_group_exclusions,
    make_grouped_split_manifest,
)


def test_union_and_shared_labels_create_transitive_order_independent_clusters() -> None:
    ids = ["d", "c", "b", "a", "e"]
    labels = {"a": ["exact:1"], "b": ["template:7"], "c": ["template:7"]}
    pairs = [("a", "b"), ("d", "e")]

    forward = build_group_clusters(ids, union_pairs=pairs, group_labels=labels)
    reverse = build_group_clusters(reversed(ids), union_pairs=reversed(pairs), group_labels=labels)

    assert forward == reverse
    assert forward["a"] == forward["b"] == forward["c"] == "a"
    assert forward["d"] == forward["e"] == "d"


def test_group_builder_rejects_unknown_references_and_duplicate_ids() -> None:
    with pytest.raises(SplitValidationError, match="duplicates"):
        build_group_clusters(["a", "a"])
    with pytest.raises(SplitValidationError, match="unknown IDs"):
        build_group_clusters(["a", "b"], union_pairs=[("a", "missing")])
    with pytest.raises(SplitValidationError, match="unknown IDs"):
        build_group_clusters(["a", "b"], group_labels={"missing": "x"})


def _fixture_groups() -> tuple[list[str], dict[str, str]]:
    sizes = [5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    ids: list[str] = []
    groups: dict[str, str] = {}
    for group_index, size in enumerate(sizes):
        group_id = f"g{group_index:02d}"
        for member_index in range(size):
            record_id = f"{group_id}-r{member_index}"
            ids.append(record_id)
            groups[record_id] = group_id
    return ids, groups


def test_split_is_deterministic_versioned_and_group_safe() -> None:
    ids, groups = _fixture_groups()
    first = make_grouped_split_manifest(
        ids,
        groups,
        n_folds=4,
        holdout_fraction=0.2,
        seed=17,
        version="audit-v3",
    )
    second = make_grouped_split_manifest(
        reversed(ids),
        groups,
        n_folds=4,
        holdout_fraction=0.2,
        seed=17,
        version="audit-v3",
    )

    assert first == second
    assert first.sha256 == second.sha256
    assert first.version == "audit-v3"
    assert first.final_holdout_ids()
    assert all(first.fold_ids(fold) for fold in range(4))

    destinations_by_group: dict[str, set[tuple[SplitPartition, int | None]]] = defaultdict(set)
    for assignment in first.assignments:
        destinations_by_group[assignment.group_id].add((assignment.partition, assignment.fold))
    assert all(len(destinations) == 1 for destinations in destinations_by_group.values())

    serialized = json.loads(first.to_json())
    assert serialized["sha256"] == first.sha256
    assert serialized["algorithm"] == "locked-holdout-balanced-group-kfold-v2"
    assert serialized["actual_counts"] == first.actual_counts()
    assert sum(serialized["actual_counts"]["partition_records"].values()) == len(ids)
    legacy_identity = first.to_dict(include_sha256=False)
    legacy_identity.pop("actual_counts")
    legacy_payload = json.dumps(
        legacy_identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert first.sha256 == hashlib.sha256(legacy_payload).hexdigest()


def test_cv_fold_sizes_are_balanced_within_largest_indivisible_group() -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(
        ids,
        groups,
        n_folds=4,
        holdout_fraction=0.15,
        seed=101,
        version="balance-v1",
    )
    fold_sizes = [len(manifest.fold_ids(fold)) for fold in range(4)]
    largest_group = max(Counter(groups.values()).values())
    assert max(fold_sizes) - min(fold_sizes) <= largest_group


def test_holdout_hash_order_does_not_concentrate_all_larger_groups() -> None:
    ids: list[str] = []
    groups: dict[str, str] = {}
    multi_groups: set[str] = set()
    for group_index in range(50):
        group_id = f"multi-{group_index:02d}"
        multi_groups.add(group_id)
        for member_index in range(2):
            record_id = f"{group_id}-{member_index}"
            ids.append(record_id)
            groups[record_id] = group_id
    for group_index in range(50):
        record_id = f"single-{group_index:02d}"
        ids.append(record_id)
        groups[record_id] = record_id

    manifest = make_grouped_split_manifest(
        ids,
        groups,
        n_folds=5,
        holdout_fraction=0.2,
        seed=20260731,
        version="hash-order-v1",
    )
    holdout_groups = {
        assignment.group_id
        for assignment in manifest.assignments
        if assignment.partition is SplitPartition.FINAL_LOCKED_HOLDOUT
    }
    held_multi = holdout_groups & multi_groups
    assert held_multi
    assert held_multi != multi_groups
    assert holdout_groups - multi_groups


def test_final_holdout_is_locked_out_of_cv() -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(ids, groups, n_folds=3, holdout_fraction=0.25)
    holdout = set(manifest.final_holdout_ids())
    cv = {record_id for fold in range(3) for record_id in manifest.fold_ids(fold)}
    assert holdout
    assert holdout.isdisjoint(cv)
    assert holdout | cv == set(ids)
    assert all(
        assignment.fold is None
        for assignment in manifest.assignments
        if assignment.partition is SplitPartition.FINAL_LOCKED_HOLDOUT
    )


def test_training_ids_exclude_validation_fold_and_locked_holdout() -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(ids, groups, n_folds=4, holdout_fraction=0.2)
    holdout = set(manifest.final_holdout_ids())
    all_cv = {record_id for fold in range(4) for record_id in manifest.fold_ids(fold)}

    for fold in range(4):
        validation = set(manifest.fold_ids(fold))
        training = set(manifest.training_ids(fold))
        assert training == all_cv - validation
        assert training.isdisjoint(validation)
        assert training.isdisjoint(holdout)

    with pytest.raises(SplitValidationError, match="fold must be"):
        manifest.training_ids(4)


def test_manifest_hash_changes_when_version_changes() -> None:
    ids, groups = _fixture_groups()
    first = make_grouped_split_manifest(ids, groups, n_folds=3, version="v1")
    second = make_grouped_split_manifest(ids, groups, n_folds=3, version="v2")
    assert first.sha256 != second.sha256


def test_manifest_strict_dict_and_json_round_trip(tmp_path) -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(
        ids,
        groups,
        n_folds=4,
        holdout_fraction=0.2,
        seed=31,
        version="reuse-v1",
    )

    assert SplitManifest.from_dict(manifest.to_dict()) == manifest
    assert SplitManifest.from_json(manifest.to_json()) == manifest

    artifact = tmp_path / "split-artifact.json"
    artifact.write_text(
        json.dumps({"metadata": "opaque", "split": manifest.to_dict()}),
        encoding="utf-8",
    )
    assert SplitManifest.load_json(artifact, manifest_key="split") == manifest


def test_manifest_loader_rejects_tamper_and_schema_changes() -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(ids, groups, n_folds=3, holdout_fraction=0.2)

    stale_sha = manifest.to_dict()
    stale_sha["algorithm"] = "tampered-algorithm"
    with pytest.raises(SplitValidationError, match="sha256 mismatch"):
        SplitManifest.from_dict(stale_sha)

    stale_counts = manifest.to_dict()
    stale_counts["actual_counts"]["fold_records"]["0"] += 1
    with pytest.raises(SplitValidationError, match="actual_counts"):
        SplitManifest.from_dict(stale_counts)

    missing_key = manifest.to_dict()
    missing_key.pop("sha256")
    with pytest.raises(SplitValidationError, match="schema mismatch"):
        SplitManifest.from_dict(missing_key)

    extra_key = manifest.to_dict()
    extra_key["unexpected"] = None
    with pytest.raises(SplitValidationError, match="schema mismatch"):
        SplitManifest.from_dict(extra_key)

    with pytest.raises(SplitValidationError, match="duplicate JSON object key"):
        SplitManifest.from_json('{"version":"v1","version":"v2"}')


def test_manifest_loader_rejects_malformed_partition_fold_and_split_group() -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(ids, groups, n_folds=4, holdout_fraction=0.2)

    invalid_partition = manifest.to_dict()
    invalid_partition["assignments"][0]["partition"] = "leaderboard"
    with pytest.raises(SplitValidationError, match="unknown partition"):
        SplitManifest.from_dict(invalid_partition)

    cv_index = next(
        index
        for index, assignment in enumerate(manifest.assignments)
        if assignment.partition is SplitPartition.CROSS_VALIDATION
    )
    invalid_fold = manifest.to_dict()
    invalid_fold["assignments"][cv_index]["fold"] = manifest.n_folds
    with pytest.raises(SplitValidationError, match="invalid CV fold"):
        SplitManifest.from_dict(invalid_fold)

    members_by_group: dict[str, list[int]] = defaultdict(list)
    for index, assignment in enumerate(manifest.assignments):
        if assignment.partition is SplitPartition.CROSS_VALIDATION:
            members_by_group[assignment.group_id].append(index)
    group_indexes = next(indexes for indexes in members_by_group.values() if len(indexes) > 1)
    split_group = manifest.to_dict()
    first_fold = split_group["assignments"][group_indexes[0]]["fold"]
    split_group["assignments"][group_indexes[1]]["fold"] = (first_fold + 1) % manifest.n_folds
    with pytest.raises(SplitValidationError, match="is split across"):
        SplitManifest.from_dict(split_group)


def test_group_expanded_exclusions_keep_cv_train_validation_and_holdout_disjoint() -> None:
    ids, groups = _fixture_groups()
    manifest = make_grouped_split_manifest(
        ids,
        groups,
        n_folds=4,
        holdout_fraction=0.2,
        seed=77,
        version="eligible-v1",
    )
    members_by_group: dict[str, list[str]] = defaultdict(list)
    for assignment in manifest.assignments:
        if assignment.partition is SplitPartition.CROSS_VALIDATION:
            members_by_group[assignment.group_id].append(assignment.record_id)
    excluded_group, excluded_members = next(
        (group_id, members)
        for group_id, members in members_by_group.items()
        if len(members) > 1
    )
    direct_exclusion = excluded_members[-1]
    excluded_fold = manifest.assignment_by_id()[direct_exclusion].fold
    assert excluded_fold is not None

    expanded = expand_hard_group_exclusions(manifest, [direct_exclusion])
    assert expanded == tuple(
        assignment.record_id
        for assignment in manifest.assignments
        if assignment.group_id == excluded_group
    )
    assert set(expanded) == set(excluded_members)

    validation = set(eligible_validation_ids(manifest, excluded_fold, [direct_exclusion]))
    assert validation == set(manifest.fold_ids(excluded_fold)) - set(excluded_members)

    training_fold = (excluded_fold + 1) % manifest.n_folds
    training = set(eligible_training_ids(manifest, training_fold, [direct_exclusion]))
    assert training == set(manifest.training_ids(training_fold)) - set(excluded_members)
    training_validation = set(
        eligible_validation_ids(manifest, training_fold, [direct_exclusion])
    )

    holdout = set(manifest.final_holdout_ids())
    assert training.isdisjoint(training_validation)
    assert training.isdisjoint(holdout)
    assert validation.isdisjoint(holdout)
    assert training_validation.isdisjoint(holdout)
    assert set(eligible_training_ids(manifest, training_fold, ())) == set(
        manifest.training_ids(training_fold)
    )

    with pytest.raises(SplitValidationError, match="absent from split manifest"):
        eligible_training_ids(manifest, 0, ["unknown-id"])


@pytest.mark.parametrize(
    "kwargs,error_fragment",
    [
        ({"n_folds": 1}, "at least 2"),
        ({"holdout_fraction": 0.0}, "strictly between"),
        ({"holdout_fraction": 1.0}, "strictly between"),
        ({"version": ""}, "version must be non-empty"),
    ],
)
def test_split_rejects_invalid_parameters(kwargs, error_fragment) -> None:
    ids, groups = _fixture_groups()
    with pytest.raises(SplitValidationError, match=error_fragment):
        make_grouped_split_manifest(ids, groups, **kwargs)


def test_split_rejects_stale_or_insufficient_group_mapping() -> None:
    with pytest.raises(SplitValidationError, match="mapping mismatch"):
        make_grouped_split_manifest(["a", "b", "c"], {"a": "a", "b": "b"}, n_folds=2)
    with pytest.raises(SplitValidationError, match="at least 4 groups"):
        make_grouped_split_manifest(
            ["a", "b", "c"],
            {"a": "one", "b": "two", "c": "three"},
            n_folds=3,
        )
