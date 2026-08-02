from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from dataset_contracts.progress_annotations import RefinementBoundaryAnnotationV1
from dataset_contracts.progress_profile import (
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_refinement import (
    RefinementBoundaryDefinitionV1,
    TaskRefinementProfileV1,
)
from dataset_interface.progress_coverage import (
    ProgressCoverageResolutionError,
    derive_refinement_annotation_coverage,
)
from dataset_interface.progress_coverage_catalog import (
    ProgressCoverageCatalogError,
    RefinementCoverageCatalog,
)
from dataset_interface.progress_refinement import (
    SelectedRefinementAnnotationSet,
)
from dataset_interface.progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _progress_profile() -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="coverage-progress",
        profile_version="1",
        task_id="coverage",
        steps=(
            ProgressStepDefinitionV1("a", "MANIPULATE", "A", 1),
            ProgressStepDefinitionV1("c", "INSPECT", "C", 2),
        ),
        initial_step_id="a",
        terminal_step_ids=("c",),
    )


def _refinement_profile() -> TaskRefinementProfileV1:
    return TaskRefinementProfileV1(
        profile_id="coverage-refinement",
        profile_version="1",
        task_progress_profile_hash=(
            _progress_profile().semantic_content_hash
        ),
        boundaries=(
            RefinementBoundaryDefinitionV1(
                progress_step_id="a",
                refinement_boundary_id="a1",
                ordinal=1,
                semantics="ENTER_STAGE",
            ),
            RefinementBoundaryDefinitionV1(
                progress_step_id="a",
                refinement_boundary_id="a2",
                ordinal=2,
                semantics="ENTER_STAGE",
            ),
        ),
    )


def _attempt(
    attempt_id: str,
    progress_step_id: str,
    ordinal: int,
    start_at: float,
    end_at: float,
) -> ResolvedProgressAttempt:
    return ResolvedProgressAttempt(
        attempt_id=attempt_id,
        progress_step_id=progress_step_id,
        attempt_ordinal_in_capture=ordinal,
        start_fact_id=f"start:{attempt_id}",
        start_at=start_at,
        clock_domain_id="episode-clock",
        end_fact_id=f"end:{attempt_id}",
        end_at=end_at,
        outcome="SUCCESS",
    )


def _timeline() -> ResolvedProgressTimeline:
    attempts = (
        _attempt("attempt:complete", "a", 1, 0.0, 10.0),
        _attempt("attempt:partial", "a", 2, 10.0, 20.0),
        _attempt("attempt:unannotated", "a", 3, 20.0, 30.0),
        _attempt("attempt:conflicted", "a", 4, 30.0, 40.0),
        _attempt("attempt:invalid", "a", 5, 40.0, 50.0),
        _attempt("attempt:not-applicable", "c", 6, 50.0, 60.0),
    )
    return ResolvedProgressTimeline(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=_progress_profile().semantic_content_hash,
        progress_episode_binding_hash="2" * 64,
        clock_domain_id="episode-clock",
        fact_bundle_hash="3" * 64,
        fact_count=12,
        sealed=True,
        phase="TERMINATED",
        attempts=attempts,
        transitions=(),
        capture_pause_intervals=(),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id="c",
        active_attempt_id=None,
        terminal_fact_id="termination",
        termination_cause="FULL_TASK_COMPLETED",
        full_task_outcome="SUCCESS",
        source_progress_manifest_hash="4" * 64,
    )


def _annotation(
    annotation_id: str,
    attempt_id: str,
    boundary_id: str,
    at: float,
    *,
    clock_domain_id: str = "episode-clock",
    confidence: float = 1.0,
) -> RefinementBoundaryAnnotationV1:
    return RefinementBoundaryAnnotationV1(
        annotation_id=annotation_id,
        coarse_attempt_id=attempt_id,
        refinement_boundary_id=boundary_id,
        effective_at=at,
        clock_domain_id=clock_domain_id,
        confidence=confidence,
    )


def _annotation_set(
    *annotations: RefinementBoundaryAnnotationV1,
    sealed: bool = True,
    source_hash: str = "4" * 64,
) -> SelectedRefinementAnnotationSet:
    return SelectedRefinementAnnotationSet(
        annotation_set_id="annotation-set:coverage",
        source_annotation_manifest_hash="5" * 64,
        source_progress_manifest_hash=source_hash,
        task_refinement_profile_hash=(
            _refinement_profile().semantic_content_hash
        ),
        sealed=sealed,
        observations=tuple(annotations),
    )


def _mixed_view():
    return derive_refinement_annotation_coverage(
        timeline=_timeline(),
        progress_profile=_progress_profile(),
        refinement_profile=_refinement_profile(),
        annotation_set=_annotation_set(
            _annotation(
                "complete:a1",
                "attempt:complete",
                "a1",
                1.0,
                confidence=0.01,
            ),
            _annotation(
                "complete:a2",
                "attempt:complete",
                "a2",
                2.0,
                confidence=0.01,
            ),
            _annotation("partial:a1", "attempt:partial", "a1", 11.0),
            _annotation(
                "conflict:a1:first",
                "attempt:conflicted",
                "a1",
                31.0,
            ),
            _annotation(
                "conflict:a1:second",
                "attempt:conflicted",
                "a1",
                32.0,
            ),
            _annotation("conflict:a2", "attempt:conflicted", "a2", 33.0),
            _annotation("invalid:a1", "attempt:invalid", "a1", 45.0),
            _annotation("invalid:a2", "attempt:invalid", "a2", 44.0),
            _annotation("orphan", "attempt:missing", "a1", 1.0),
        ),
    )


def test_coverage_classifies_mixed_attempts_without_using_confidence() -> None:
    view = _mixed_view()
    by_id = {item.attempt_id: item for item in view.attempts}

    assert by_id["attempt:complete"].status == "COMPLETE"
    assert by_id["attempt:partial"].status == "PARTIAL"
    assert by_id["attempt:partial"].missing_boundary_ids == ("a2",)
    assert by_id["attempt:unannotated"].status == "UNANNOTATED"
    assert by_id["attempt:conflicted"].status == "CONFLICTED"
    assert by_id["attempt:conflicted"].conflicting_boundary_ids == ("a1",)
    assert by_id["attempt:invalid"].status == "INVALID"
    assert by_id["attempt:invalid"].invalid_annotation_ids == (
        "invalid:a1",
        "invalid:a2",
    )
    assert by_id["attempt:not-applicable"].status == "NOT_APPLICABLE"
    assert view.unassigned_annotation_ids == ("orphan",)
    assert view.source_annotation_manifest_hash == "5" * 64
    assert view.has_invalid_annotations
    assert view.status_counts == {
        "COMPLETE": 1,
        "CONFLICTED": 1,
        "INVALID": 1,
        "NOT_APPLICABLE": 1,
        "PARTIAL": 1,
        "UNANNOTATED": 1,
    }


def test_invalid_annotation_facts_do_not_satisfy_coverage() -> None:
    view = derive_refinement_annotation_coverage(
        timeline=_timeline(),
        progress_profile=_progress_profile(),
        refinement_profile=_refinement_profile(),
        annotation_set=_annotation_set(
            _annotation(
                "foreign",
                "attempt:partial",
                "a1",
                11.0,
                clock_domain_id="foreign-clock",
            ),
            _annotation(
                "wrong-step",
                "attempt:not-applicable",
                "a1",
                51.0,
            ),
            _annotation(
                "outside",
                "attempt:complete",
                "a1",
                12.0,
            ),
        ),
    )
    by_id = {item.attempt_id: item for item in view.attempts}

    assert by_id["attempt:partial"].status == "INVALID"
    assert by_id["attempt:partial"].observed_boundary_ids == ()
    assert by_id["attempt:not-applicable"].status == "INVALID"
    assert by_id["attempt:complete"].status == "INVALID"


def test_coverage_requires_exact_sealed_capture_binding() -> None:
    kwargs = {
        "timeline": _timeline(),
        "progress_profile": _progress_profile(),
        "refinement_profile": _refinement_profile(),
    }
    with pytest.raises(ProgressCoverageResolutionError, match="sealed"):
        derive_refinement_annotation_coverage(
            **kwargs,
            annotation_set=_annotation_set(sealed=False),
        )
    with pytest.raises(ProgressCoverageResolutionError, match="not bound"):
        derive_refinement_annotation_coverage(
            **kwargs,
            annotation_set=_annotation_set(source_hash="9" * 64),
        )


def test_catalog_is_wal_queryable_and_rebuildable_after_deletion(
    tmp_path: Path,
) -> None:
    view = _mixed_view()
    database_path = tmp_path / "semantic_catalog.sqlite"

    with RefinementCoverageCatalog(database_path) as catalog:
        catalog.rebuild((view,))
        first_attempts = catalog.list_attempts()
        partial_statuses = tuple(
            item.status for item in catalog.list_attempts(status="partial")
        )
        assert partial_statuses == ("PARTIAL",)
        assert tuple(
            item.attempt_id
            for item in catalog.list_attempts(
                task_refinement_profile_hash=(
                    view.task_refinement_profile_hash
                ),
                attempt_id="attempt:complete",
            )
        ) == ("attempt:complete",)
        issue_sets = catalog.list_annotation_sets(with_issues_only=True)
        assert len(issue_sets) == 1
        assert issue_sets[0].source_annotation_manifest_hash == "5" * 64
        assert issue_sets[0].unassigned_annotation_ids == ("orphan",)
        assert catalog.journal_mode == "wal"

    database_path.unlink()
    with RefinementCoverageCatalog(database_path) as rebuilt:
        rebuilt.rebuild((view,))
        assert rebuilt.list_attempts() == first_attempts


def test_catalog_replace_is_atomic_and_rejects_duplicate_rebuild_keys(
    tmp_path: Path,
) -> None:
    view = _mixed_view()
    with RefinementCoverageCatalog(tmp_path / "coverage.sqlite") as catalog:
        catalog.replace_view(view)
        before = catalog.list_attempts()
        with pytest.raises(
            ProgressCoverageCatalogError,
            match="duplicate annotation-set keys",
        ):
            catalog.rebuild((view, view))
        assert catalog.list_attempts() == before


def test_catalog_keys_same_annotation_set_id_by_exact_manifest(
    tmp_path: Path,
) -> None:
    first = _mixed_view()
    second = replace(
        first,
        source_annotation_manifest_hash="6" * 64,
        attempts=first.attempts[:1],
    )
    with RefinementCoverageCatalog(tmp_path / "coverage.sqlite") as catalog:
        catalog.rebuild((first, second))

        assert len(catalog.list_annotation_sets()) == 2
        assert len(
            catalog.list_attempts(
                source_annotation_manifest_hash="5" * 64,
            )
        ) == len(first.attempts)
        assert len(
            catalog.list_attempts(
                source_annotation_manifest_hash="6" * 64,
            )
        ) == 1


def test_catalog_can_be_recreated_from_views_without_source_mutation(
    tmp_path: Path,
) -> None:
    view = _mixed_view()
    path = tmp_path / "coverage.sqlite"
    with RefinementCoverageCatalog(path) as catalog:
        catalog.rebuild((view,))
        expected = catalog.list_annotation_sets()

    with sqlite3.connect(path) as raw:
        raw.execute("DELETE FROM attempt_coverage")
        raw.commit()

    with RefinementCoverageCatalog(path) as catalog:
        assert catalog.list_attempts() == ()
        catalog.rebuild((view,))
        assert catalog.list_annotation_sets() == expected
        assert len(catalog.list_attempts()) == len(view.attempts)


def test_catalog_wal_replacement_preserves_an_active_reader_snapshot(
    tmp_path: Path,
) -> None:
    view = _mixed_view()
    reduced_view = replace(view, attempts=view.attempts[:1])
    path = tmp_path / "coverage.sqlite"

    with RefinementCoverageCatalog(path) as catalog:
        catalog.rebuild((view,))
        with sqlite3.connect(path, isolation_level=None) as reader:
            reader.execute("BEGIN")
            assert reader.execute(
                "SELECT COUNT(*) FROM attempt_coverage"
            ).fetchone() == (len(view.attempts),)

            catalog.replace_view(reduced_view)
            assert reader.execute(
                "SELECT COUNT(*) FROM attempt_coverage"
            ).fetchone() == (len(view.attempts),)

            reader.execute("COMMIT")
            assert reader.execute(
                "SELECT COUNT(*) FROM attempt_coverage"
            ).fetchone() == (1,)
