# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import pytest

from GBOpt.artifacts.cleanup import (
    ArtifactCleanupError,
    ArtifactCleanupRequest,
    _ArtifactCleaner,
    remove_managed_path,
)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"candidate_id": "", "source_path": "candidate.data"},
            "candidate_id",
            id="empty-candidate-id",
        ),
        pytest.param(
            {"candidate_id": "GA_1", "source_path": ""},
            "source_path",
            id="empty-source-path",
        ),
        pytest.param(
            {"candidate_id": "GA_1", "source_path": b"candidate.data"},
            "source_path",
            id="bytes-source-path",
        ),
    ],
)
def test_cleanup_request_rejects_malformed_fields(kwargs, match):
    with pytest.raises(ArtifactCleanupError, match=match):
        ArtifactCleanupRequest(**kwargs)


def test_cleanup_request_normalizes_paths_without_requiring_existing_files(tmp_path):
    request = ArtifactCleanupRequest(
        candidate_id="GA_1_g0_c0",
        source_path=tmp_path / "missing" / "candidate.data",
        archive_path=tmp_path / "archive" / "candidate.data",
    )

    assert request.source_path.is_absolute()
    assert request.archive_path is not None
    assert request.archive_path.is_absolute()


def test_remove_managed_file_is_idempotent(tmp_path):
    managed_root = tmp_path / "evaluations"
    managed_root.mkdir()
    artifact = managed_root / "candidate.data"
    artifact.write_text("candidate", encoding="utf-8")

    remove_managed_path(artifact, managed_root=managed_root)
    remove_managed_path(artifact, managed_root=managed_root)

    assert not artifact.exists()


def test_remove_managed_directory_is_recursive(tmp_path):
    managed_root = tmp_path / "evaluations"
    work_dir = managed_root / "workdir.17" / "gen_4"
    work_dir.mkdir(parents=True)
    (work_dir / "candidate.data").write_text("candidate", encoding="utf-8")
    (work_dir / "log.lammps").write_text("log", encoding="utf-8")

    remove_managed_path(work_dir.parent, managed_root=managed_root)

    assert not work_dir.parent.exists()
    assert managed_root.is_dir()


@pytest.mark.parametrize("relative", ["../outside.data", "../../outside.data"])
def test_remove_managed_path_rejects_lexical_escape(tmp_path, relative):
    managed_root = tmp_path / "evaluations"
    managed_root.mkdir()
    outside = tmp_path / "outside.data"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(ArtifactCleanupError, match="outside managed artifact root"):
        remove_managed_path(managed_root / relative, managed_root=managed_root)

    assert outside.is_file()


def test_remove_managed_path_rejects_managed_root_itself(tmp_path):
    managed_root = tmp_path / "evaluations"
    managed_root.mkdir()

    with pytest.raises(ArtifactCleanupError, match="managed artifact root itself"):
        remove_managed_path(managed_root, managed_root=managed_root)

    assert managed_root.is_dir()


def test_remove_managed_path_rejects_symlink_escape(tmp_path):
    managed_root = tmp_path / "evaluations"
    managed_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "candidate.data"
    outside.write_text("outside", encoding="utf-8")
    escape = managed_root / "escape"
    escape.symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ArtifactCleanupError, match="outside managed artifact root"):
        remove_managed_path(escape / "candidate.data", managed_root=managed_root)

    assert outside.is_file()


def test_remove_managed_path_unlinks_in_root_symlink_without_deleting_target(tmp_path):
    managed_root = tmp_path / "evaluations"
    managed_root.mkdir()
    target = managed_root / "target.data"
    target.write_text("target", encoding="utf-8")
    link = managed_root / "candidate.data"
    link.symlink_to(target)

    remove_managed_path(link, managed_root=managed_root)

    assert not link.exists()
    assert target.is_file()


def test_artifact_cleaner_managed_root_removes_only_source_path(tmp_path):
    managed_root = tmp_path / "evaluations"
    managed_root.mkdir()
    source = managed_root / "candidate.data"
    source.write_text("candidate", encoding="utf-8")
    cleaner = _ArtifactCleaner(managed_artifact_root=managed_root)

    cleaner.cleanup_source(
        ArtifactCleanupRequest(candidate_id="GA_1_g0_c0", source_path=source)
    )

    assert not source.exists()


def test_artifact_cleaner_callback_receives_commit_safe_request(tmp_path):
    source = tmp_path / "workdir.1" / "gen_0" / "candidate.dump"
    source.parent.mkdir(parents=True)
    source.write_text("candidate", encoding="utf-8")
    archive = tmp_path / "run.artifacts" / "structures" / "GA_1_g0_c0.data"
    observed = []

    def cleanup_candidate(request):
        observed.append(request)

    cleaner = _ArtifactCleaner(cleanup_candidate=cleanup_candidate)
    request = ArtifactCleanupRequest(
        candidate_id="GA_1_g0_c0",
        source_path=source,
        archive_path=archive,
    )

    cleaner.cleanup_source(request)

    assert observed == [request]
    assert source.is_file()


def test_artifact_cleaner_wraps_callback_failure(tmp_path):
    def cleanup_candidate(_request):
        raise OSError("backend cleanup failed")

    cleaner = _ArtifactCleaner(cleanup_candidate=cleanup_candidate)
    request = ArtifactCleanupRequest(
        candidate_id="GA_1_g0_c0",
        source_path=tmp_path / "candidate.data",
    )

    with pytest.raises(
        ArtifactCleanupError, match="cleanup callback failed"
    ) as exc_info:
        cleaner.cleanup_source(request)

    assert isinstance(exc_info.value.__cause__, OSError)


def test_artifact_cleaner_rejects_ambiguous_cleanup_ownership(tmp_path):
    with pytest.raises(ArtifactCleanupError, match="either managed_artifact_root"):
        _ArtifactCleaner(
            managed_artifact_root=tmp_path,
            cleanup_candidate=lambda _request: None,
        )


def test_artifact_cleaner_rejects_non_callable_callback():
    with pytest.raises(ArtifactCleanupError, match="must be callable"):
        _ArtifactCleaner(
            cleanup_candidate=object()  # ty: ignore[invalid-argument-type]
        )
