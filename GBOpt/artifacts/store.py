# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Track runtime artifact references without performing destructive cleanup.

The store consumes validated :class:`RetentionCandidate` values and optional retention
policy decisions, and returns immutable :class:`ArtifactRecord` snapshots plus pruning
eligibility. It owns mutable operational pins and scientific retention reasons. Archive
materialization, provenance output, and filesystem deletion belong to later lifecycle
layers rather than this module.
"""

from __future__ import annotations

from pathlib import Path

from GBOpt.artifacts.policy import ArtifactPolicyError, ArtifactRetentionPolicy
from GBOpt.artifacts.types import (
    ArtifactError,
    ArtifactPin,
    ArtifactRecord,
    ArtifactValueError,
    RetentionCandidate,
)

_STORE_VERSION = 1
_RULE_REASON_PREFIX = "rule:"


class ArtifactStoreError(ArtifactError):
    """Raised when runtime artifact-store state or operations are invalid."""


def _normalize_reason(reason: object) -> str:
    """Validate one non-empty retention reason.

    :param reason: Candidate scientific retention reason.
    :return: Validated reason string.
    :raises ArtifactStoreError: If the reason is not a non-empty string.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ArtifactStoreError("retention reason must be a non-empty string")
    return reason


def _normalize_source_path(path: object) -> str | None:
    """Normalize one optional source artifact path without assigning it identity.

    :param path: Candidate evaluator artifact path or ``None``.
    :return: String path or ``None``.
    :raises ArtifactStoreError: If the path is not a non-empty string/path value.
    """
    if path is not None:
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ArtifactStoreError("source_path must be a non-empty path or None")
        return str(path)
    return path


def _record_from_state(raw_record: object) -> ArtifactRecord:
    """Restore one immutable artifact record from serialized state.

    :param raw_record: JSON-decoded record state.
    :return: Validated immutable artifact record.
    :raises ArtifactStoreError: If record or reference state is malformed.
    """
    if not isinstance(raw_record, dict):
        raise ArtifactStoreError("artifact store record state is invalid")
    try:
        candidate = RetentionCandidate.from_state(raw_record["candidate"])
        source_path = _normalize_source_path(raw_record.get("source_path"))
        archive_path = _normalize_source_path(raw_record.get("archive_path"))
        raw_pins = raw_record["pins"]
        raw_reasons = raw_record["retention_reasons"]
        if not isinstance(raw_pins, list) or not isinstance(raw_reasons, list):
            raise ArtifactStoreError("artifact store references state is invalid")
        pins = tuple(ArtifactPin(value) for value in raw_pins)
        reasons = tuple(_normalize_reason(reason) for reason in raw_reasons)
        return ArtifactRecord(
            candidate=candidate,
            source_path=source_path,
            archive_path=archive_path,
            pins=pins,
            retention_reasons=reasons,
        )
    except (KeyError, TypeError, ValueError, ArtifactValueError) as exc:
        raise ArtifactStoreError("artifact store state is malformed") from exc


class ArtifactStore:
    """Track candidate artifacts, operational pins, and scientific retention reasons.

    The store performs no filesystem deletion. ``is_prunable`` only reports lifecycle
    eligibility after all references have disappeared and pruning is enabled by policy.

    :param policy: Optional retention policy. ``None`` preserves keep-all behavior.
    :raises ArtifactStoreError: If policy has the wrong type.
    """

    def __init__(self, policy: ArtifactRetentionPolicy | None = None) -> None:
        """Initialize an empty runtime store.

        :param policy: Optional retention policy, defaults to ``None``; preserves
            keep-all behavior when omitted.
        :raises ArtifactStoreError: If policy has the wrong type.
        """
        if policy is not None and not isinstance(policy, ArtifactRetentionPolicy):
            raise ArtifactStoreError(
                "policy must be an ArtifactRetentionPolicy or None"
            )
        self.policy = policy
        self._candidates: dict[str, RetentionCandidate] = {}
        self._source_paths: dict[str, str | None] = {}
        self._archive_paths: dict[str, str | None] = {}
        self._pins: dict[str, set[ArtifactPin]] = {}
        self._reasons: dict[str, set[str]] = {}

    @property
    def pruning_enabled(self) -> bool:
        """Return whether unreferenced candidates may be reported as prunable."""
        return self.policy is not None and self.policy.prune

    def __len__(self) -> int:
        """Return the number of registered logical candidates."""
        return len(self._candidates)

    def __contains__(self, candidate_id: object) -> bool:
        """Return whether a logical candidate identity is registered.

        :param candidate_id: Candidate identity to test.
        :return: Whether the identity is a registered non-empty string.
        """
        return isinstance(candidate_id, str) and candidate_id in self._candidates

    def _require_candidate(self, candidate_id: object) -> str:
        """Validate and resolve one registered candidate identity.

        :param candidate_id: Candidate identity to resolve.
        :return: Validated registered identity.
        :raises ArtifactStoreError: If the identity is malformed or unknown.
        """
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ArtifactStoreError("candidate_id must be a non-empty string")
        if candidate_id not in self._candidates:
            raise ArtifactStoreError(f"unknown candidate_id {candidate_id!r}")
        return candidate_id

    def register_candidate(
        self,
        candidate: RetentionCandidate,
        *,
        source_path: str | Path | None = None,
    ) -> ArtifactRecord:
        """Register one evaluated candidate and refresh scientific membership.

        Registration is idempotent only when both logical candidate state and source
        path agree. A conflicting reuse of a stable candidate identity is rejected.

        :param candidate: Evaluated scientific candidate metadata.
        :param source_path: Keyword argument, optional, defaults to ``None``. Evaluator
            artifact path.
        :return: Current immutable record snapshot.
        :raises ArtifactStoreError: If candidate state conflicts with an existing
            identity or the configured policy cannot evaluate the prospective
            population.
        """
        if not isinstance(candidate, RetentionCandidate):
            raise ArtifactStoreError("candidate must be a RetentionCandidate")
        normalized_path = _normalize_source_path(source_path)
        candidate_id = candidate.candidate_id
        if candidate_id in self._candidates:
            if self._candidates[candidate_id] != candidate:
                raise ArtifactStoreError(
                    f"candidate identity {candidate_id!r} was reused for different "
                    "state"
                )
            if self._source_paths[candidate_id] != normalized_path:
                raise ArtifactStoreError(
                    f"candidate identity {candidate_id!r} was reused with a different "
                    "source path"
                )
            return self.record(candidate_id)

        memberships = None
        if self.policy is not None:
            try:
                memberships = self.policy.evaluate(
                    (*self._candidates.values(), candidate)
                )
            except ArtifactPolicyError as exc:
                raise ArtifactStoreError(
                    f"retention policy could not register candidate {candidate_id!r}"
                ) from exc
            self._validate_policy_memberships(
                memberships,
                candidate_ids=(*self._candidates, candidate_id),
            )

        self._candidates[candidate_id] = candidate
        self._source_paths[candidate_id] = normalized_path
        self._archive_paths[candidate_id] = None
        self._pins[candidate_id] = set()
        self._reasons[candidate_id] = set()
        if memberships is not None:
            self._apply_policy_memberships(memberships)
        return self.record(candidate_id)

    def pin(self, candidate_id: str, pin: ArtifactPin) -> None:
        """Add an operational pin idempotently.

        :param candidate_id: Registered logical candidate identity.
        :param pin: Operational restart dependency to add.
        :raises ArtifactStoreError: If candidate identity or pin is invalid.
        """
        candidate_id = self._require_candidate(candidate_id)
        if not isinstance(pin, ArtifactPin):
            raise ArtifactStoreError("pin must be an ArtifactPin")
        self._pins[candidate_id].add(pin)

    def release_pin(self, candidate_id: str, pin: ArtifactPin) -> None:
        """Release one operational pin idempotently.

        :param candidate_id: Registered logical candidate identity.
        :param pin: Operational restart dependency to release.
        :raises ArtifactStoreError: If candidate identity or pin is invalid.
        """
        candidate_id = self._require_candidate(candidate_id)
        if not isinstance(pin, ArtifactPin):
            raise ArtifactStoreError("pin must be an ArtifactPin")
        self._pins[candidate_id].discard(pin)

    def replace_pin(self, pin: ArtifactPin, candidate_id: str | None) -> None:
        """Move a singleton-style operational pin to a new candidate.

        This is intended for references such as ``BEST_RESULT``. Passing ``None``
        releases the pin from every candidate.

        :param pin: Singleton-style operational pin to move.
        :param candidate_id: New registered owner, or ``None`` to release globally.
        :raises ArtifactStoreError: If pin or candidate identity is invalid.
        """
        if not isinstance(pin, ArtifactPin):
            raise ArtifactStoreError("pin must be an ArtifactPin")
        if candidate_id is not None:
            candidate_id = self._require_candidate(candidate_id)
        for pins in self._pins.values():
            pins.discard(pin)
        if candidate_id is not None:
            self._pins[candidate_id].add(pin)

    def add_retention_reason(self, candidate_id: str, reason: str) -> None:
        """Add one scientific retention reason idempotently.

        :param candidate_id: Registered logical candidate identity.
        :param reason: Scientific retention reason.
        :raises ArtifactStoreError: If candidate identity or reason is invalid.
        """
        candidate_id = self._require_candidate(candidate_id)
        self._reasons[candidate_id].add(_normalize_reason(reason))

    def remove_retention_reason(self, candidate_id: str, reason: str) -> None:
        """Remove one scientific retention reason idempotently.

        :param candidate_id: Registered logical candidate identity.
        :param reason: Scientific retention reason.
        :raises ArtifactStoreError: If candidate identity or reason is invalid.
        """
        candidate_id = self._require_candidate(candidate_id)
        self._reasons[candidate_id].discard(_normalize_reason(reason))

    def refresh_retention(self) -> None:
        """Recompute all policy-owned rule reasons from registered candidates.

        Rule evaluation completes before any reason is mutated, so a failing property or
        callback cannot partially update archive membership.

        :raises ArtifactStoreError: If the policy cannot evaluate candidates or
            evaluated membership violates store invariants.
        """
        if self.policy is None:
            return
        try:
            memberships = self.policy.evaluate(tuple(self._candidates.values()))
        except ArtifactPolicyError as exc:
            raise ArtifactStoreError(
                "retention policy could not refresh registered candidates"
            ) from exc
        self._apply_policy_memberships(memberships)

    def _validate_policy_memberships(
        self,
        memberships: dict[str, tuple[str, ...]],
        *,
        candidate_ids: tuple[str, ...],
    ) -> None:
        """Validate policy membership before mutating runtime reference state.

        :param memberships: Rule-name to retained candidate identities.
        :param candidate_ids: Keyword argument, required. Identities valid for this
            evaluation boundary.
        :raises ArtifactStoreError: If rule names, membership shape, duplicates, or
            candidate identities are invalid.
        """
        if self.policy is None:
            raise ArtifactStoreError("policy membership requires a configured policy")
        expected_rules = set(self.policy.rule_names)
        if set(memberships) != expected_rules:
            raise ArtifactStoreError(
                "retention policy returned membership for unexpected rule names"
            )
        known = set(candidate_ids)
        for rule_name, members in memberships.items():
            if not isinstance(members, tuple) or any(
                not isinstance(candidate_id, str) for candidate_id in members
            ):
                raise ArtifactStoreError(
                    f"retention rule {rule_name!r} returned invalid membership state"
                )
            if len(set(members)) != len(members):
                raise ArtifactStoreError(
                    f"retention rule {rule_name!r} returned duplicate candidate "
                    "identities"
                )
            unknown = sorted(set(members).difference(known))
            if unknown:
                raise ArtifactStoreError(
                    "retention policy returned unknown candidate identities: "
                    + ", ".join(unknown)
                )

    def _apply_policy_memberships(
        self, memberships: dict[str, tuple[str, ...]]
    ) -> None:
        """Replace only policy-owned rule reasons with evaluated membership.

        :param memberships: Rule-name to retained candidate identities.
        :raises ArtifactStoreError: If a policy returns an unknown candidate identity.
        """
        if self.policy is None:
            raise ArtifactStoreError("policy membership requires a configured policy")
        self._validate_policy_memberships(
            memberships, candidate_ids=tuple(self._candidates)
        )
        policy_reasons = {
            f"{_RULE_REASON_PREFIX}{rule_name}" for rule_name in self.policy.rule_names
        }
        for reasons in self._reasons.values():
            reasons.difference_update(policy_reasons)
        for rule_name, candidate_ids in memberships.items():
            reason = f"{_RULE_REASON_PREFIX}{rule_name}"
            for candidate_id in candidate_ids:
                self._reasons[candidate_id].add(reason)

    def set_archive_path(
        self, candidate_id: str, archive_path: str | Path | None
    ) -> None:
        """Set or clear the canonical retained structure path.

        :param candidate_id: Registered logical candidate identity.
        :param archive_path: Canonical archive path, or ``None`` after eviction.
        :raises ArtifactStoreError: If candidate identity or path is invalid.
        """
        candidate_id = self._require_candidate(candidate_id)
        self._archive_paths[candidate_id] = _normalize_source_path(archive_path)

    def archive_path(self, candidate_id: str) -> str | None:
        """Return the canonical retained structure path for one candidate.

        :param candidate_id: Registered logical candidate identity.
        :return: Archive path or ``None`` when no canonical copy exists.
        :raises ArtifactStoreError: If candidate identity is malformed or unknown.
        """
        candidate_id = self._require_candidate(candidate_id)
        return self._archive_paths[candidate_id]

    def source_is_prunable(self, candidate_id: str) -> bool:
        """Return whether the evaluator source file may be removed after commit.

        Scientific reasons may be satisfied by a canonical archive copy. Restart pins
        other than ``BEST_RESULT`` block source pruning until they are released or
        rebased; ``BEST_RESULT`` may be satisfied by a validated canonical archive copy.

        :param candidate_id: Registered logical candidate identity.
        :return: Whether the source artifact is eligible for exact-file cleanup.
        :raises ArtifactStoreError: If candidate identity is malformed or unknown.
        """
        record = self.record(candidate_id)
        if not self.pruning_enabled or record.source_path is None:
            return False
        blocking_pins = set(record.pins).difference({ArtifactPin.BEST_RESULT})
        if blocking_pins:
            return False
        if (record.retention_reasons or ArtifactPin.BEST_RESULT in record.pins) and (
            record.archive_path is None
        ):
            return False
        return True

    def record(self, candidate_id: str) -> ArtifactRecord:
        """Return an immutable snapshot for one registered candidate.

        :param candidate_id: Registered logical candidate identity.
        :return: Immutable artifact record.
        :raises ArtifactStoreError: If candidate identity is malformed or unknown.
        """
        candidate_id = self._require_candidate(candidate_id)
        # pyrc: ignore=DOC115[ArtifactValueError] Candidate, path, pin, and reason state
        #   is normalized on every store mutation, so snapshot construction only
        #   revalidates already-enforced invariants.
        return ArtifactRecord(
            candidate=self._candidates[candidate_id],
            source_path=self._source_paths[candidate_id],
            archive_path=self._archive_paths[candidate_id],
            pins=tuple(self._pins[candidate_id]),
            retention_reasons=tuple(self._reasons[candidate_id]),
        )

    def records(self) -> tuple[ArtifactRecord, ...]:
        """Return all records in stable candidate-ID lexical order.

        :return: Immutable artifact records ordered by candidate identity.
        :raises ArtifactStoreError: If internal candidate reference state is
            inconsistent.
        """
        return tuple(
            self.record(candidate_id) for candidate_id in sorted(self._candidates)
        )

    def pins(self, candidate_id: str) -> tuple[ArtifactPin, ...]:
        """Return active operational pins in deterministic order.

        :param candidate_id: Registered logical candidate identity.
        :return: Active operational pins.
        :raises ArtifactStoreError: If candidate identity is malformed or unknown.
        """
        return self.record(candidate_id).pins

    def retention_reasons(self, candidate_id: str) -> tuple[str, ...]:
        """Return active scientific retention reasons in deterministic order.

        :param candidate_id: Registered logical candidate identity.
        :return: Active scientific retention reasons.
        :raises ArtifactStoreError: If candidate identity is malformed or unknown.
        """
        return self.record(candidate_id).retention_reasons

    def is_prunable(self, candidate_id: str) -> bool:
        """Return whether policy permits pruning and all references are absent.

        :param candidate_id: Registered logical candidate identity.
        :return: Whether the candidate is currently eligible for later cleanup.
        :raises ArtifactStoreError: If candidate identity is malformed or unknown.
        """
        record = self.record(candidate_id)
        return self.pruning_enabled and not record.pins and not record.retention_reasons

    def prunable_candidate_ids(self) -> tuple[str, ...]:
        """Return all currently prunable candidate IDs in lexical order.

        :return: Logical identities currently eligible for later cleanup.
        :raises ArtifactStoreError: If internal candidate reference state is
            inconsistent.
        """
        return tuple(
            candidate_id
            for candidate_id in sorted(self._candidates)
            if self.is_prunable(candidate_id)
        )

    def to_state(self) -> dict[str, object]:
        """Return deterministic JSON-safe runtime store state.

        :return: Callback-free store state for checkpoint integration.
        :raises ArtifactStoreError: If internal candidate reference state is
            inconsistent.
        """
        return {
            "version": _STORE_VERSION,
            "policy_signature": None if self.policy is None else self.policy.signature,
            "records": [
                {
                    "candidate": record.candidate.to_state(),
                    "source_path": record.source_path,
                    "archive_path": record.archive_path,
                    "pins": [pin.value for pin in record.pins],
                    "retention_reasons": list(record.retention_reasons),
                }
                for record in self.records()
            ],
        }

    def _restore_record(self, record: ArtifactRecord) -> None:
        """Install one validated record into restored runtime state.

        :param record: Validated immutable artifact record.
        :raises ArtifactStoreError: If the candidate identity is duplicated.
        """
        candidate_id = record.candidate_id
        if candidate_id in self._candidates:
            raise ArtifactStoreError(
                f"duplicate candidate identity {candidate_id!r} in store state"
            )
        self._candidates[candidate_id] = record.candidate
        self._source_paths[candidate_id] = record.source_path
        self._archive_paths[candidate_id] = record.archive_path
        self._pins[candidate_id] = set(record.pins)
        self._reasons[candidate_id] = set(record.retention_reasons)

    def _validate_restored_policy_membership(self) -> None:
        """Verify persisted rule reasons against the configured runtime policy.

        :raises ArtifactStoreError: If policy evaluation fails or persisted rule
            membership does not match the configured policy.
        """
        if self.policy is None:
            return
        try:
            expected_memberships = self.policy.evaluate(
                tuple(self._candidates.values())
            )
        except ArtifactPolicyError as exc:
            raise ArtifactStoreError(
                "artifact store state cannot be evaluated by the configured policy"
            ) from exc
        self._validate_policy_memberships(
            expected_memberships, candidate_ids=tuple(self._candidates)
        )

        expected_rule_reasons = {
            candidate_id: set() for candidate_id in self._candidates
        }
        for rule_name, candidate_ids in expected_memberships.items():
            reason = f"{_RULE_REASON_PREFIX}{rule_name}"
            for candidate_id in candidate_ids:
                expected_rule_reasons[candidate_id].add(reason)

        configured_reasons = {
            f"{_RULE_REASON_PREFIX}{name}" for name in self.policy.rule_names
        }
        for candidate_id, reasons in self._reasons.items():
            actual = reasons.intersection(configured_reasons)
            if actual != expected_rule_reasons[candidate_id]:
                raise ArtifactStoreError(
                    "artifact store rule membership does not match the configured "
                    "policy"
                )

    @classmethod
    def from_state(
        cls,
        state: object,
        *,
        policy: ArtifactRetentionPolicy | None = None,
    ) -> "ArtifactStore":
        """Restore deterministic runtime state and verify policy compatibility.

        :param state: JSON-decoded store state.
        :param policy: Keyword argument, optional, defaults to ``None``. Runtime
            policy/callback configuration.
        :return: Restored in-memory store.
        :raises ArtifactStoreError: If state is malformed, policy identity differs, or
            persisted rule membership is inconsistent.
        """
        if not isinstance(state, dict):
            raise ArtifactStoreError("artifact store state must be a dictionary")
        if state.get("version") != _STORE_VERSION:
            raise ArtifactStoreError("unsupported artifact store state version")
        expected_signature = None if policy is None else policy.signature
        if state.get("policy_signature") != expected_signature:
            raise ArtifactStoreError("artifact retention policy signature mismatch")
        raw_records = state.get("records")
        if not isinstance(raw_records, list):
            raise ArtifactStoreError("artifact store records state must be a list")

        store = cls(policy=policy)
        for raw_record in raw_records:
            store._restore_record(_record_from_state(raw_record))
        store._validate_restored_policy_membership()
        return store


__all__ = ["ArtifactStore", "ArtifactStoreError"]
