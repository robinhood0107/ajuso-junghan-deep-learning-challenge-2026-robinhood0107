"""Leakage-resistant, deterministic grouped validation splits.

The final holdout is selected first and is never assigned a CV fold.  Remaining
groups are greedily balanced across folds by record count.  Every artifact is
versioned and content-addressable so an experiment can pin its exact split.

Important: union inputs are *hard equivalence assertions*.  Fuzzy similarity or
number-masked template matches can join semantically different problems and
must remain soft audit signals until a deterministic rule or human review has
confirmed that grouping them is appropriate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final


class SplitValidationError(ValueError):
    """Raised when clustering or split invariants cannot be satisfied."""


_SPLIT_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "version",
        "seed",
        "n_folds",
        "holdout_fraction",
        "algorithm",
        "source_groups_sha256",
        "actual_counts",
        "assignments",
        "sha256",
    }
)
_ASSIGNMENT_KEYS: Final[frozenset[str]] = frozenset(
    {"record_id", "group_id", "partition", "fold"}
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")


class SplitPartition(StrEnum):
    """Mutually exclusive evaluation partitions."""

    CROSS_VALIDATION = "cross_validation"
    FINAL_LOCKED_HOLDOUT = "final_locked_holdout"


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Partition and fold assignment for one record."""

    record_id: str
    group_id: str
    partition: SplitPartition
    fold: int | None


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Deterministic, versioned record assignments."""

    version: str
    seed: int
    n_folds: int
    holdout_fraction: float
    algorithm: str
    source_groups_sha256: str
    assignments: tuple[SplitAssignment, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SplitManifest:
        """Deserialize a manifest using an exact schema and stored SHA check."""

        if not isinstance(payload, Mapping):
            raise SplitValidationError("split manifest must be a JSON object")
        keys = set(payload)
        if keys != _SPLIT_MANIFEST_KEYS:
            missing = sorted(_SPLIT_MANIFEST_KEYS - keys)
            extra = sorted(keys - _SPLIT_MANIFEST_KEYS)
            raise SplitValidationError(
                f"split manifest schema mismatch; missing={missing!r}, extra={extra!r}"
            )

        version = _required_nonempty_string(payload["version"], "version")
        seed = _required_int(payload["seed"], "seed")
        n_folds = _required_int(payload["n_folds"], "n_folds")
        holdout_fraction = payload["holdout_fraction"]
        if isinstance(holdout_fraction, bool) or not isinstance(holdout_fraction, float):
            raise SplitValidationError("holdout_fraction must be a JSON number with decimals")
        algorithm = _required_nonempty_string(payload["algorithm"], "algorithm")
        source_groups_sha256 = _required_sha256(
            payload["source_groups_sha256"], "source_groups_sha256"
        )
        stored_sha256 = _required_sha256(payload["sha256"], "sha256")

        raw_assignments = payload["assignments"]
        if not isinstance(raw_assignments, list) or not raw_assignments:
            raise SplitValidationError("assignments must be a non-empty JSON array")
        assignments: list[SplitAssignment] = []
        for index, raw_assignment in enumerate(raw_assignments):
            if not isinstance(raw_assignment, Mapping):
                raise SplitValidationError(f"assignment {index} must be a JSON object")
            assignment_keys = set(raw_assignment)
            if assignment_keys != _ASSIGNMENT_KEYS:
                missing = sorted(_ASSIGNMENT_KEYS - assignment_keys)
                extra = sorted(assignment_keys - _ASSIGNMENT_KEYS)
                raise SplitValidationError(
                    f"assignment {index} schema mismatch; missing={missing!r}, extra={extra!r}"
                )
            record_id = _required_nonempty_string(
                raw_assignment["record_id"], f"assignment {index} record_id"
            )
            group_id = _required_nonempty_string(
                raw_assignment["group_id"], f"assignment {index} group_id"
            )
            raw_partition = raw_assignment["partition"]
            if not isinstance(raw_partition, str):
                raise SplitValidationError(f"assignment {index} partition must be a string")
            try:
                partition = SplitPartition(raw_partition)
            except ValueError as exc:
                raise SplitValidationError(
                    f"assignment {index} has unknown partition {raw_partition!r}"
                ) from exc
            raw_fold = raw_assignment["fold"]
            if raw_fold is not None and (
                isinstance(raw_fold, bool) or not isinstance(raw_fold, int)
            ):
                raise SplitValidationError(
                    f"assignment {index} fold must be an integer or null"
                )
            assignments.append(
                SplitAssignment(
                    record_id=record_id,
                    group_id=group_id,
                    partition=partition,
                    fold=raw_fold,
                )
            )

        manifest = cls(
            version=version,
            seed=seed,
            n_folds=n_folds,
            holdout_fraction=holdout_fraction,
            algorithm=algorithm,
            source_groups_sha256=source_groups_sha256,
            assignments=tuple(assignments),
        )
        manifest.validate()

        _validate_actual_counts(payload["actual_counts"], manifest.actual_counts())
        if stored_sha256 != manifest.sha256:
            raise SplitValidationError(
                f"split manifest sha256 mismatch: stored={stored_sha256}, "
                f"computed={manifest.sha256}"
            )
        return manifest

    @classmethod
    def from_json(cls, text: str) -> SplitManifest:
        """Deserialize one strict JSON manifest, rejecting duplicate keys."""

        if not isinstance(text, str):
            raise SplitValidationError("split manifest JSON must be text")
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise SplitValidationError(f"invalid split manifest JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        manifest_key: str | None = None,
    ) -> SplitManifest:
        """Load a UTF-8 JSON manifest, optionally from an artifact wrapper key."""

        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise SplitValidationError(f"cannot read split manifest {source}: {exc}") from exc
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise SplitValidationError(f"invalid split manifest JSON {source}: {exc}") from exc
        if manifest_key is not None:
            if not isinstance(manifest_key, str) or not manifest_key:
                raise SplitValidationError("manifest_key must be a non-empty string")
            if not isinstance(payload, Mapping) or manifest_key not in payload:
                raise SplitValidationError(
                    f"split manifest JSON does not contain key {manifest_key!r}"
                )
            payload = payload[manifest_key]
        return cls.from_dict(payload)

    @property
    def sha256(self) -> str:
        """Hash of identity fields, stable across added derived metadata.

        ``actual_counts`` is fully derivable from assignments and is excluded
        so adding it to serialized output does not invalidate older hashes.
        """

        identity = self.to_dict(include_sha256=False)
        identity.pop("actual_counts")
        payload = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        """Return a JSON-compatible representation with stable assignment order."""

        result: dict[str, object] = {
            "version": self.version,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "holdout_fraction": self.holdout_fraction,
            "algorithm": self.algorithm,
            "source_groups_sha256": self.source_groups_sha256,
            "actual_counts": self.actual_counts(),
            "assignments": [
                {
                    "record_id": assignment.record_id,
                    "group_id": assignment.group_id,
                    "partition": assignment.partition.value,
                    "fold": assignment.fold,
                }
                for assignment in self.assignments
            ],
        }
        if include_sha256:
            result["sha256"] = self.sha256
        return result

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the manifest without mutating or writing external state."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent) + "\n"

    def assignment_by_id(self) -> dict[str, SplitAssignment]:
        return {assignment.record_id: assignment for assignment in self.assignments}

    def final_holdout_ids(self) -> tuple[str, ...]:
        return tuple(
            assignment.record_id
            for assignment in self.assignments
            if assignment.partition is SplitPartition.FINAL_LOCKED_HOLDOUT
        )

    def fold_ids(self, fold: int) -> tuple[str, ...]:
        self._validate_fold(fold)
        return tuple(
            assignment.record_id
            for assignment in self.assignments
            if assignment.partition is SplitPartition.CROSS_VALIDATION
            and assignment.fold == fold
        )

    def training_ids(self, fold: int) -> tuple[str, ...]:
        """Return CV training IDs, excluding both ``fold`` and locked holdout.

        This is the safe API for fold training.  Constructing training IDs as
        merely "everything outside the validation fold" would silently include
        the final locked holdout and invalidate model selection.
        """

        self._validate_fold(fold)
        return tuple(
            assignment.record_id
            for assignment in self.assignments
            if assignment.partition is SplitPartition.CROSS_VALIDATION
            and assignment.fold != fold
        )

    def actual_counts(self) -> dict[str, dict[str, int]]:
        """Return realized record/group counts by partition and CV fold."""

        partition_records = {partition.value: 0 for partition in SplitPartition}
        partition_groups: dict[str, set[str]] = {
            partition.value: set() for partition in SplitPartition
        }
        fold_records = {str(fold): 0 for fold in range(self.n_folds)}
        fold_groups: dict[str, set[str]] = {
            str(fold): set() for fold in range(self.n_folds)
        }
        for assignment in self.assignments:
            partition = assignment.partition.value
            partition_records[partition] += 1
            partition_groups[partition].add(assignment.group_id)
            if assignment.partition is SplitPartition.CROSS_VALIDATION:
                if assignment.fold is None:
                    raise SplitValidationError(
                        f"CV assignment {assignment.record_id!r} is missing its fold"
                    )
                fold_key = str(assignment.fold)
                fold_records[fold_key] += 1
                fold_groups[fold_key].add(assignment.group_id)
        return {
            "partition_records": partition_records,
            "partition_groups": {
                partition: len(groups) for partition, groups in partition_groups.items()
            },
            "fold_records": fold_records,
            "fold_groups": {fold: len(groups) for fold, groups in fold_groups.items()},
        }

    def _validate_fold(self, fold: int) -> None:
        if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0 or fold >= self.n_folds:
            raise SplitValidationError(f"fold must be in [0, {self.n_folds}), got {fold}")

    def validate(self) -> None:
        """Re-check uniqueness, fold bounds, and whole-group containment."""

        if (
            not isinstance(self.version, str)
            or not self.version
            or self.version != self.version.strip()
        ):
            raise SplitValidationError("version must be a non-empty trimmed string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SplitValidationError("seed must be an integer")
        if (
            isinstance(self.n_folds, bool)
            or not isinstance(self.n_folds, int)
            or self.n_folds < 2
        ):
            raise SplitValidationError("n_folds must be an integer of at least 2")
        if (
            isinstance(self.holdout_fraction, bool)
            or not isinstance(self.holdout_fraction, float)
            or not 0.0 < self.holdout_fraction < 1.0
        ):
            raise SplitValidationError("holdout_fraction must be a float strictly between 0 and 1")
        if (
            not isinstance(self.algorithm, str)
            or not self.algorithm
            or self.algorithm != self.algorithm.strip()
        ):
            raise SplitValidationError("algorithm must be a non-empty trimmed string")
        _required_sha256(self.source_groups_sha256, "source_groups_sha256")
        if not self.assignments:
            raise SplitValidationError("assignments must not be empty")

        seen: set[str] = set()
        group_destinations: dict[str, tuple[SplitPartition, int | None]] = {}
        for assignment in self.assignments:
            if (
                not isinstance(assignment.record_id, str)
                or not assignment.record_id
                or assignment.record_id != assignment.record_id.strip()
            ):
                raise SplitValidationError("every record ID must be a non-empty trimmed string")
            if (
                not isinstance(assignment.group_id, str)
                or not assignment.group_id
                or assignment.group_id != assignment.group_id.strip()
            ):
                raise SplitValidationError("every group ID must be a non-empty trimmed string")
            if not isinstance(assignment.partition, SplitPartition):
                raise SplitValidationError(
                    f"invalid partition {assignment.partition!r} for {assignment.record_id!r}"
                )
            if assignment.record_id in seen:
                raise SplitValidationError(f"duplicate assignment for {assignment.record_id!r}")
            seen.add(assignment.record_id)
            if assignment.partition is SplitPartition.FINAL_LOCKED_HOLDOUT:
                if assignment.fold is not None:
                    raise SplitValidationError("final holdout assignment cannot have a CV fold")
            elif (
                assignment.fold is None
                or isinstance(assignment.fold, bool)
                or not isinstance(assignment.fold, int)
                or not 0 <= assignment.fold < self.n_folds
            ):
                raise SplitValidationError(
                    f"invalid CV fold {assignment.fold!r} for {assignment.record_id!r}"
                )
            destination = (assignment.partition, assignment.fold)
            previous = group_destinations.setdefault(assignment.group_id, destination)
            if previous != destination:
                raise SplitValidationError(
                    f"group {assignment.group_id!r} is split across "
                    f"{previous!r} and {destination!r}"
                )

        ordered_ids = tuple(assignment.record_id for assignment in self.assignments)
        if ordered_ids != tuple(sorted(ordered_ids)):
            raise SplitValidationError("assignments must be sorted by record_id")
        if not any(
            assignment.partition is SplitPartition.FINAL_LOCKED_HOLDOUT
            for assignment in self.assignments
        ):
            raise SplitValidationError("final locked holdout must not be empty")
        populated_folds = {
            assignment.fold
            for assignment in self.assignments
            if assignment.partition is SplitPartition.CROSS_VALIDATION
        }
        expected_folds = set(range(self.n_folds))
        if populated_folds != expected_folds:
            raise SplitValidationError(
                f"CV folds mismatch; expected={sorted(expected_folds)!r}, "
                f"actual={sorted(populated_folds)!r}"
            )
        group_by_id = {
            assignment.record_id: assignment.group_id for assignment in self.assignments
        }
        computed_source_hash = _source_groups_sha256(ordered_ids, group_by_id)
        if self.source_groups_sha256 != computed_source_hash:
            raise SplitValidationError(
                "source_groups_sha256 does not match assignment record/group mapping"
            )


def build_group_clusters(
    record_ids: Iterable[str],
    *,
    union_pairs: Iterable[tuple[str, str]] = (),
    group_labels: Mapping[str, str | Iterable[str]] | None = None,
) -> dict[str, str]:
    """Resolve pairwise unions and shared labels into deterministic clusters.

    ``group_labels`` may provide one label or multiple labels per record.  Any
    records sharing at least one label are transitively unioned.  Supplying a
    label therefore asserts verified hard equivalence; do **not** pass raw fuzzy
    similarity, nearest-neighbor, or number-masked-template matches here.  Keep
    those as soft review signals and pass only adjudicated unions.  The returned
    group ID is the lexicographically smallest member ID, making it stable
    across input ordering and Python processes.
    """

    ids = _validated_record_ids(record_ids)
    id_set = set(ids)
    parent = {record_id: record_id for record_id in ids}
    rank = dict.fromkeys(ids, 0)

    def find(record_id: str) -> str:
        root = record_id
        while parent[root] != root:
            root = parent[root]
        while parent[record_id] != record_id:
            next_id = parent[record_id]
            parent[record_id] = root
            record_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for pair in union_pairs:
        if len(pair) != 2:
            raise SplitValidationError(f"union pair must have two IDs, got {pair!r}")
        left, right = pair
        unknown = {left, right} - id_set
        if unknown:
            raise SplitValidationError(f"union pair references unknown IDs: {sorted(unknown)!r}")
        union(left, right)

    if group_labels is not None:
        unknown_label_ids = set(group_labels) - id_set
        if unknown_label_ids:
            raise SplitValidationError(
                f"group labels reference unknown IDs: {sorted(unknown_label_ids)!r}"
            )
        first_by_label: dict[str, str] = {}
        for record_id in sorted(group_labels):
            raw_labels = group_labels[record_id]
            labels = (raw_labels,) if isinstance(raw_labels, str) else tuple(raw_labels)
            for label in labels:
                if not isinstance(label, str) or not label:
                    raise SplitValidationError(
                        f"group label for {record_id!r} must be a non-empty string"
                    )
                first = first_by_label.setdefault(label, record_id)
                union(first, record_id)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for record_id in ids:
        members_by_root[find(record_id)].append(record_id)
    group_id_by_root = {
        root: min(members) for root, members in members_by_root.items()
    }
    return {
        record_id: group_id_by_root[find(record_id)]
        for record_id in sorted(ids)
    }


def make_grouped_split_manifest(
    record_ids: Iterable[str],
    group_by_id: Mapping[str, str],
    *,
    n_folds: int = 5,
    holdout_fraction: float = 0.1,
    seed: int = 20260731,
    version: str = "v1",
) -> SplitManifest:
    """Create a locked final holdout followed by balanced grouped K-folds."""

    ids = _validated_record_ids(record_ids)
    _validate_split_parameters(ids, group_by_id, n_folds, holdout_fraction, version)

    members_by_group: dict[str, list[str]] = defaultdict(list)
    for record_id in ids:
        members_by_group[group_by_id[record_id]].append(record_id)
    for members in members_by_group.values():
        members.sort()

    group_sizes = {group_id: len(members) for group_id, members in members_by_group.items()}
    holdout_groups = _choose_holdout_groups(
        group_sizes,
        total_records=len(ids),
        holdout_fraction=holdout_fraction,
        n_folds=n_folds,
        seed=seed,
        version=version,
    )
    cv_groups = set(members_by_group) - holdout_groups
    folds_by_group = _assign_cv_folds(
        cv_groups,
        group_sizes,
        n_folds=n_folds,
        seed=seed,
        version=version,
    )

    assignments: list[SplitAssignment] = []
    for record_id in sorted(ids):
        group_id = group_by_id[record_id]
        if group_id in holdout_groups:
            assignments.append(
                SplitAssignment(
                    record_id=record_id,
                    group_id=group_id,
                    partition=SplitPartition.FINAL_LOCKED_HOLDOUT,
                    fold=None,
                )
            )
        else:
            assignments.append(
                SplitAssignment(
                    record_id=record_id,
                    group_id=group_id,
                    partition=SplitPartition.CROSS_VALIDATION,
                    fold=folds_by_group[group_id],
                )
            )

    manifest = SplitManifest(
        version=version,
        seed=seed,
        n_folds=n_folds,
        holdout_fraction=holdout_fraction,
        algorithm="locked-holdout-balanced-group-kfold-v2",
        source_groups_sha256=_source_groups_sha256(ids, group_by_id),
        assignments=tuple(assignments),
    )
    manifest.validate()
    if any(not manifest.fold_ids(fold) for fold in range(n_folds)):
        raise SplitValidationError("each CV fold must contain at least one complete group")
    if not manifest.final_holdout_ids():
        raise SplitValidationError("final locked holdout must not be empty")
    return manifest


def deterministic_group_kfold(
    record_ids: Iterable[str],
    group_by_id: Mapping[str, str],
    **kwargs: int | float | str,
) -> SplitManifest:
    """Readable alias for :func:`make_grouped_split_manifest`."""

    return make_grouped_split_manifest(record_ids, group_by_id, **kwargs)


def expand_hard_group_exclusions(
    manifest: SplitManifest,
    excluded_ids: Iterable[str],
) -> tuple[str, ...]:
    """Expand direct exclusions to every member of their verified hard groups.

    Group membership comes only from the validated split manifest.  It must not
    be replaced by fuzzy or number-masked soft similarities.
    """

    excluded_groups = _excluded_group_ids(manifest, excluded_ids)
    return tuple(
        assignment.record_id
        for assignment in manifest.assignments
        if assignment.group_id in excluded_groups
    )


def eligible_training_ids(
    manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return group-safe eligible train IDs without ever adding holdout IDs."""

    candidate_ids = manifest.training_ids(fold)
    return _filter_excluded_groups(manifest, candidate_ids, excluded_ids)


def eligible_validation_ids(
    manifest: SplitManifest,
    fold: int,
    excluded_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return group-safe eligible validation IDs from exactly one CV fold."""

    candidate_ids = manifest.fold_ids(fold)
    return _filter_excluded_groups(manifest, candidate_ids, excluded_ids)


def load_split_manifest_json(
    path: str | Path,
    *,
    manifest_key: str | None = None,
) -> SplitManifest:
    """Load a validated split manifest from a direct or wrapped JSON artifact."""

    return SplitManifest.load_json(path, manifest_key=manifest_key)


def _filter_excluded_groups(
    manifest: SplitManifest,
    candidate_ids: Sequence[str],
    excluded_ids: Iterable[str],
) -> tuple[str, ...]:
    excluded_groups = _excluded_group_ids(manifest, excluded_ids)
    assignment_by_id = manifest.assignment_by_id()
    return tuple(
        record_id
        for record_id in candidate_ids
        if assignment_by_id[record_id].group_id not in excluded_groups
    )


def _excluded_group_ids(
    manifest: SplitManifest,
    excluded_ids: Iterable[str],
) -> frozenset[str]:
    manifest.validate()
    direct_ids = _validated_exclusion_ids(excluded_ids)
    assignment_by_id = manifest.assignment_by_id()
    unknown_ids = set(direct_ids) - set(assignment_by_id)
    if unknown_ids:
        raise SplitValidationError(
            f"exclusions reference IDs absent from split manifest: {sorted(unknown_ids)!r}"
        )
    return frozenset(assignment_by_id[record_id].group_id for record_id in direct_ids)


def _validated_exclusion_ids(excluded_ids: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(excluded_ids)
    for record_id in ids:
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id != record_id.strip()
            or "\x00" in record_id
        ):
            raise SplitValidationError(
                "every exclusion ID must be a non-empty trimmed string without NUL"
            )
    if len(set(ids)) != len(ids):
        raise SplitValidationError("exclusion IDs contain duplicates")
    return tuple(sorted(ids))


def _validated_record_ids(record_ids: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(record_ids)
    if not ids:
        raise SplitValidationError("record_ids must not be empty")
    if any(
        not isinstance(record_id, str)
        or not record_id
        or record_id != record_id.strip()
        or "\x00" in record_id
        for record_id in ids
    ):
        raise SplitValidationError(
            "every record ID must be a non-empty trimmed string without NUL"
        )
    if len(set(ids)) != len(ids):
        raise SplitValidationError("record_ids contain duplicates")
    return ids


def _required_nonempty_string(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise SplitValidationError(f"{name} must be a non-empty trimmed string without NUL")
    return value


def _required_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitValidationError(f"{name} must be an integer")
    return value


def _required_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SplitValidationError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _validate_actual_counts(
    value: object,
    expected: Mapping[str, Mapping[str, int]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise SplitValidationError("stored actual_counts schema does not match assignments")
    for section, expected_counts in expected.items():
        actual_section = value[section]
        if not isinstance(actual_section, Mapping) or set(actual_section) != set(
            expected_counts
        ):
            raise SplitValidationError(
                f"stored actual_counts section {section!r} has an invalid schema"
            )
        for key, expected_count in expected_counts.items():
            actual_count = actual_section[key]
            if (
                isinstance(actual_count, bool)
                or not isinstance(actual_count, int)
                or actual_count != expected_count
            ):
                raise SplitValidationError(
                    f"stored actual_counts value {section}.{key} does not match assignments"
                )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SplitValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SplitValidationError(f"non-standard JSON numeric constant {value!r}")


def _validate_split_parameters(
    ids: Sequence[str],
    group_by_id: Mapping[str, str],
    n_folds: int,
    holdout_fraction: float,
    version: str,
) -> None:
    if n_folds < 2:
        raise SplitValidationError("n_folds must be at least 2")
    if not 0.0 < holdout_fraction < 1.0:
        raise SplitValidationError("holdout_fraction must be strictly between 0 and 1")
    if not version:
        raise SplitValidationError("version must be non-empty")
    expected = set(ids)
    actual = set(group_by_id)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SplitValidationError(f"group mapping mismatch; missing={missing!r}, extra={extra!r}")
    invalid_group = any(
        not isinstance(group_by_id[record_id], str) or not group_by_id[record_id]
        for record_id in ids
    )
    if invalid_group:
        raise SplitValidationError("every record must have a non-empty string group ID")
    group_count = len({group_by_id[record_id] for record_id in ids})
    if group_count < n_folds + 1:
        raise SplitValidationError(
            f"need at least {n_folds + 1} groups for {n_folds} CV folds plus holdout; "
            f"got {group_count}"
        )


def _choose_holdout_groups(
    group_sizes: Mapping[str, int],
    *,
    total_records: int,
    holdout_fraction: float,
    n_folds: int,
    seed: int,
    version: str,
) -> set[str]:
    target = total_records * holdout_fraction
    ordered = sorted(
        group_sizes,
        key=lambda group_id: (
            _stable_rank("holdout", seed, version, group_id),
            group_id,
        ),
    )
    selected: set[str] = set()
    selected_size = 0
    for group_id in ordered:
        candidate_size = selected_size + group_sizes[group_id]
        if abs(candidate_size - target) < abs(selected_size - target):
            selected.add(group_id)
            selected_size = candidate_size

    if not selected:
        best = min(
            ordered,
            key=lambda group_id: (
                abs(group_sizes[group_id] - target),
                _stable_rank("holdout-fallback", seed, version, group_id),
            ),
        )
        selected.add(best)
        selected_size = group_sizes[best]

    # Preserve enough whole groups to populate every CV fold.  Removal is also
    # deterministic and chooses the smallest damage to the requested target.
    while len(group_sizes) - len(selected) < n_folds:
        removed = min(
            selected,
            key=lambda group_id: (
                abs((selected_size - group_sizes[group_id]) - target),
                _stable_rank("holdout-trim", seed, version, group_id),
            ),
        )
        selected.remove(removed)
        selected_size -= group_sizes[removed]

    if not selected:
        raise SplitValidationError("holdout request leaves no group available for holdout")
    return selected


def _assign_cv_folds(
    cv_groups: set[str],
    group_sizes: Mapping[str, int],
    *,
    n_folds: int,
    seed: int,
    version: str,
) -> dict[str, int]:
    fold_sizes = [0] * n_folds
    fold_group_counts = [0] * n_folds
    fold_tie_rank = [
        _stable_rank("fold", seed, version, str(fold)) for fold in range(n_folds)
    ]
    assignment: dict[str, int] = {}
    ordered = sorted(
        cv_groups,
        key=lambda group_id: (
            -group_sizes[group_id],
            _stable_rank("cv-group", seed, version, group_id),
            group_id,
        ),
    )
    for group_id in ordered:
        fold = min(
            range(n_folds),
            key=lambda candidate: (
                fold_sizes[candidate],
                fold_group_counts[candidate],
                fold_tie_rank[candidate],
            ),
        )
        assignment[group_id] = fold
        fold_sizes[fold] += group_sizes[group_id]
        fold_group_counts[fold] += 1
    return assignment


def _source_groups_sha256(ids: Sequence[str], group_by_id: Mapping[str, str]) -> str:
    payload = json.dumps(
        [[record_id, group_by_id[record_id]] for record_id in sorted(ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_rank(namespace: str, seed: int, version: str, value: str) -> str:
    material = f"{namespace}\0{seed}\0{version}\0{value}".encode()
    return hashlib.sha256(material).hexdigest()


__all__ = [
    "SplitAssignment",
    "SplitManifest",
    "SplitPartition",
    "SplitValidationError",
    "build_group_clusters",
    "deterministic_group_kfold",
    "eligible_training_ids",
    "eligible_validation_ids",
    "expand_hard_group_exclusions",
    "load_split_manifest_json",
    "make_grouped_split_manifest",
]
