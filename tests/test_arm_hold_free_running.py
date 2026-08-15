from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dataset_interface.arm_hold_windows import (
    ArmHoldHistoryContext,
    ArmHoldWindowLocation,
    ArmHoldWindowSample,
)
from policy_learning.arm_hold import (
    arm_hold_free_running_evaluation,
    arm_hold_offset_zero_evaluation,
)


def _location(
    start: int,
    *,
    continuity: int = 0,
    episode: str = "episode-a",
) -> ArmHoldWindowLocation:
    return ArmHoldWindowLocation(
        source_dataset_id="source",
        source_episode_index=0,
        episode_capture_identity_hash=episode,
        start_index=start,
        front_video_frame_index=start,
        continuity_segment_index=continuity,
    )


def _sample(
    location: ArmHoldWindowLocation,
    *,
    prediction_from_state: tuple[int, int],
    target: tuple[int, int],
    transition: tuple[bool, bool],
    teacher_history: tuple[int, int] | None = None,
) -> ArmHoldWindowSample:
    return ArmHoldWindowSample(
        front_rgb=None,
        observation_state=np.asarray(prediction_from_state, dtype=np.float32),
        arm_hold_target=np.repeat(
            np.asarray(target, dtype=np.float32)[None], 2, axis=0
        ),
        arm_hold_valid_mask=np.ones((2, 2), dtype=bool),
        arm_hold_loss_mask=np.ones((2, 2), dtype=bool),
        arm_hold_transition_mask=np.repeat(
            np.asarray(transition, dtype=bool)[None], 2, axis=0
        ),
        arm_hold_history_context=(
            None
            if teacher_history is None
            else ArmHoldHistoryContext(
                previous_mode=np.asarray(teacher_history, dtype=np.float32),
                available_mask=np.asarray([True, True]),
                dwell_steps=np.asarray([9, 9], dtype=np.int64),
            )
        ),
        metadata={
            "source_dataset_id": location.source_dataset_id,
            "source_episode_index": location.source_episode_index,
            "episode_capture_identity_hash": (
                location.episode_capture_identity_hash
            ),
            "start_index": location.start_index,
            "continuity_segment_index": location.continuity_segment_index,
        },
    )


class _StatePredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            input_mode="state_only", label_context_mode="none"
        )

    def forward_batch(self, batch):  # type: ignore[no-untyped-def]
        values = batch.observation_state
        assert values is not None
        logits = torch.where(values > 0.5, 8.0, -8.0)
        return logits[:, None].expand(-1, 2, -1)


class _PreviousModeEchoModel(nn.Module):
    def __init__(self, context_mode: str = "previous_mode") -> None:
        super().__init__()
        self.config = SimpleNamespace(
            input_mode="state_only", label_context_mode=context_mode
        )
        self.received: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        *,
        front_rgb,
        observation_state,
        history_previous_mode,
        history_available_mask,
        history_dwell_steps,
    ):  # type: ignore[no-untyped-def]
        del front_rgb, observation_state
        self.received.append(
            (
                history_previous_mode.detach().cpu().clone(),
                history_available_mask.detach().cpu().clone(),
                history_dwell_steps.detach().cpu().clone(),
            )
        )
        bootstrap = torch.tensor(
            [1.0, 0.0], device=history_previous_mode.device
        ).expand_as(history_previous_mode)
        prediction = torch.where(
            history_available_mask, history_previous_mode, bootstrap
        )
        logits = torch.where(prediction > 0.5, 8.0, -8.0)
        return logits[:, None].expand(-1, 2, -1)


class _ScriptedDwellModel(nn.Module):
    def __init__(self, context_mode: str = "previous_mode_dwell") -> None:
        super().__init__()
        self.config = SimpleNamespace(
            input_mode="state_only", label_context_mode=context_mode
        )
        self.received: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._predictions = (
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
        )

    def forward(
        self,
        *,
        front_rgb,
        observation_state,
        history_previous_mode,
        history_available_mask,
        history_dwell_steps,
    ):  # type: ignore[no-untyped-def]
        del front_rgb, observation_state
        self.received.append(
            (
                history_previous_mode.detach().cpu().clone(),
                history_available_mask.detach().cpu().clone(),
                history_dwell_steps.detach().cpu().clone(),
            )
        )
        prediction = torch.tensor(
            self._predictions[len(self.received) - 1],
            dtype=torch.float32,
            device=history_previous_mode.device,
        ).expand_as(history_previous_mode)
        logits = torch.where(prediction > 0.5, 8.0, -8.0)
        return logits[:, None].expand(-1, 2, -1)


class _ScriptedHistorySequenceModel(nn.Module):
    def __init__(self, context_mode: str) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            input_mode="state_only", label_context_mode=context_mode
        )
        self.received: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._predictions = (
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
        )

    def forward(
        self,
        *,
        front_rgb,
        observation_state,
        history_sequence_previous_mode,
        history_sequence_available_mask,
        history_sequence_dwell_steps,
    ):  # type: ignore[no-untyped-def]
        del front_rgb, observation_state
        self.received.append(
            (
                history_sequence_previous_mode.detach().cpu().clone(),
                history_sequence_available_mask.detach().cpu().clone(),
                history_sequence_dwell_steps.detach().cpu().clone(),
            )
        )
        prediction = torch.tensor(
            self._predictions[len(self.received) - 1],
            dtype=torch.float32,
            device=history_sequence_previous_mode.device,
        ).expand(history_sequence_previous_mode.shape[0], -1)
        logits = torch.where(prediction > 0.5, 8.0, -8.0)
        return logits[:, None].expand(-1, 2, -1)


class _FrontStateDwellModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            input_mode="front_state",
            label_context_mode="previous_dwell_bucket",
        )
        self.received: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        *,
        front_rgb,
        observation_state,
        history_previous_mode,
        history_available_mask,
        history_dwell_steps,
    ):  # type: ignore[no-untyped-def]
        assert front_rgb is not None
        assert observation_state is not None
        self.received.append(
            (
                int(front_rgb.shape[0]),
                history_available_mask.detach().cpu().clone(),
                history_dwell_steps.detach().cpu().clone(),
            )
        )
        prediction = observation_state > 0.5
        logits = torch.where(prediction, 8.0, -8.0)
        return logits[:, None].expand(-1, 2, -1)


@pytest.mark.parametrize(
    "context_mode", ["previous_mode", "previous_mode_dwell"]
)
def test_history_rollout_uses_prediction_not_teacher_label_and_resets(
    context_mode: str,
) -> None:
    locations = (
        _location(0),
        _location(1),
        _location(2),
        _location(6, continuity=1),
    )
    samples = tuple(
        _sample(
            location,
            prediction_from_state=(0, 0),
            target=target,
            transition=transition,
            teacher_history=(0, 1),
        )
        for location, target, transition in zip(
            locations,
            ((0, 0), (1, 0), (1, 0), (0, 1)),
            ((False, False), (True, False), (False, False), (False, False)),
            strict=True,
        )
    )
    model = _PreviousModeEchoModel(context_mode)

    read_order: list[int] = []

    def sample_at(index: int) -> ArmHoldWindowSample:
        read_order.append(index)
        return samples[index]

    report = arm_hold_free_running_evaluation(
        model=model,  # type: ignore[arg-type]
        locations=locations,
        sample_at=sample_at,
        device="cpu",
    )

    assert read_order == [0, 1, 2, 3]
    assert report["rollout_segment_count"] == 2
    assert report["bootstrap_context_window_count"] == 2
    assert report["feedback_context_window_count"] == 2
    first_previous, first_available, first_dwell = model.received[0]
    assert first_previous.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert not first_available.any()
    assert first_dwell.tolist() == [[0, 0], [0, 0]]
    second_previous, second_available, second_dwell = model.received[1]
    assert second_previous.tolist() == [[1.0, 0.0]]
    assert second_available.tolist() == [[True, True]]
    assert second_dwell.tolist() == [[1, 1]]
    third_previous, _, third_dwell = model.received[2]
    assert third_previous.tolist() == [[1.0, 0.0]]
    assert third_dwell.tolist() == [[2, 2]]
    if context_mode == "previous_mode_dwell":
        assert report["dwell_dynamics"]["predicted_feedback"]["per_side"][
            "left"
        ]["support"] == 2
    else:
        assert "dwell_dynamics" not in report


@pytest.mark.parametrize(
    ("context_mode", "feedback_semantics"),
    (
        (
            "previous_mode_dwell",
            "previous_predicted_offset_zero_mode_and_consecutive_dwell_v1",
        ),
        (
            "previous_dwell_bucket",
            "previous_predicted_offset_zero_consecutive_dwell_bucket_v1",
        ),
    ),
)
def test_dwell_rollout_increments_and_resets_from_predictions(
    context_mode: str,
    feedback_semantics: str,
) -> None:
    locations = tuple(_location(index) for index in range(4))
    samples = tuple(
        _sample(
            location,
            prediction_from_state=(0, 0),
            target=(0, 0),
            transition=(False, False),
            teacher_history=(1, 1),
        )
        for location in locations
    )
    model = _ScriptedDwellModel(context_mode)

    report = arm_hold_free_running_evaluation(
        model=model,  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
    )

    assert report["context_mode"] == context_mode
    assert report["feedback_semantics"] == feedback_semantics
    assert report["bootstrap_context_window_count"] == 1
    assert report["feedback_context_window_count"] == 3
    dynamics = report["dwell_dynamics"]
    assert dynamics["canonical_reference"][
        "evaluation_only_not_model_input"
    ] is True
    predicted_left = dynamics["predicted_feedback"]["per_side"]["left"]
    canonical_left = dynamics["canonical_reference"]["per_side"]["left"]
    assert predicted_left["support"] == 3
    assert canonical_left["support"] == 3
    assert predicted_left["by_previous_mode_and_dwell_bucket"][
        "fixed_hold:1"
    ] == {
        "support": 1,
        "predicted_switch_count": 0,
        "predicted_switch_rate": 0.0,
        "target_disagreement_count": 1,
        "target_disagreement_rate": 1.0,
    }
    assert predicted_left["by_previous_mode_and_dwell_bucket"][
        "fixed_hold:2"
    ]["predicted_switch_count"] == 1
    assert canonical_left["by_previous_mode_and_dwell_bucket"][
        "teleop:1"
    ]["target_switch_count"] == 0
    assert dynamics["joint_mode_dwell_distribution_total_variation"][
        "left"
    ] == pytest.approx(2.0 / 3.0)
    assert dynamics["joint_mode_dwell_distribution_total_variation"][
        "right"
    ] == pytest.approx(0.0)
    assert len(model.received) == 4
    first_previous, first_available, first_dwell = model.received[0]
    assert first_previous.tolist() == [[0.0, 0.0]]
    assert not first_available.any()
    assert first_dwell.tolist() == [[0, 0]]
    assert model.received[1][0].tolist() == [[1.0, 0.0]]
    assert model.received[1][2].tolist() == [[1, 1]]
    assert model.received[2][0].tolist() == [[1.0, 0.0]]
    assert model.received[2][2].tolist() == [[2, 2]]
    assert model.received[3][0].tolist() == [[0.0, 0.0]]
    assert model.received[3][2].tolist() == [[1, 3]]


def test_dwell_dynamics_reports_saturated_bucket_without_false_switch() -> None:
    locations = tuple(_location(index) for index in range(18))
    samples = tuple(
        _sample(
            location,
            prediction_from_state=(0, 0),
            target=(0, 0),
            transition=(False, False),
        )
        for location in locations
    )

    report = arm_hold_free_running_evaluation(
        model=_PreviousModeEchoModel("previous_mode_dwell"),  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
    )

    dynamics = report["dwell_dynamics"]
    predicted = dynamics["predicted_feedback"]["per_side"]
    canonical = dynamics["canonical_reference"]["per_side"]
    assert predicted["left"]["support"] == 17
    assert predicted["left"]["saturated_16_plus_support"] == 2
    assert predicted["left"]["saturated_16_plus_rate"] == pytest.approx(
        2.0 / 17.0
    )
    assert predicted["left"]["by_previous_mode_and_dwell_bucket"][
        "fixed_hold:16+"
    ]["predicted_switch_count"] == 0
    assert canonical["left"]["saturated_16_plus_support"] == 2
    assert canonical["left"]["by_previous_mode_and_dwell_bucket"][
        "teleop:16+"
    ]["target_switch_count"] == 0


@pytest.mark.parametrize(
    ("context_mode", "feedback_semantics"),
    (
        (
            "previous_two_dwell_bucket",
            "previous_two_predicted_offset_zero_consecutive_dwell_buckets_v1",
        ),
        (
            "previous_two_mode_dwell",
            "previous_two_predicted_offset_zero_modes_and_consecutive_dwell_v1",
        ),
    ),
)
def test_two_step_rollout_shifts_predicted_label_dwell_history(
    context_mode: str,
    feedback_semantics: str,
) -> None:
    locations = tuple(_location(index) for index in range(4))
    samples = tuple(
        _sample(
            location,
            prediction_from_state=(0, 0),
            target=(0, 0),
            transition=(False, False),
            teacher_history=(1, 1),
        )
        for location in locations
    )
    model = _ScriptedHistorySequenceModel(context_mode)

    report = arm_hold_free_running_evaluation(
        model=model,  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
    )

    assert report["context_mode"] == context_mode
    assert report["feedback_semantics"] == feedback_semantics
    assert report["dwell_dynamics"]["predicted_feedback"]["per_side"][
        "left"
    ]["support"] == 3
    assert len(model.received) == 4
    first_previous, first_available, first_dwell = model.received[0]
    assert first_previous.tolist() == [[[0.0, 0.0], [0.0, 0.0]]]
    assert not first_available.any()
    assert not first_dwell.any()
    assert model.received[1][0].tolist() == [
        [[0.0, 0.0], [1.0, 0.0]]
    ]
    assert model.received[1][1].tolist() == [
        [[False, False], [True, True]]
    ]
    assert model.received[1][2].tolist() == [[[0, 0], [1, 1]]]
    assert model.received[2][0].tolist() == [
        [[1.0, 0.0], [1.0, 0.0]]
    ]
    assert model.received[2][2].tolist() == [[[1, 1], [2, 2]]]
    assert model.received[3][0].tolist() == [
        [[1.0, 0.0], [0.0, 0.0]]
    ]
    assert model.received[3][2].tolist() == [[[2, 2], [1, 3]]]


def test_front_state_rollout_batches_independent_segments_without_semantic_drift(
) -> None:
    starts = ((0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16))
    locations = tuple(
        _location(start, continuity=segment_index)
        for segment_index, segment_starts in enumerate(starts)
        for start in segment_starts
    )
    samples = tuple(
        replace(
            _sample(
                location,
                prediction_from_state=(1, location.continuity_segment_index % 2),
                target=(1, location.continuity_segment_index % 2),
                transition=(False, False),
                teacher_history=(0, 0),
            ),
            front_rgb=np.full(
                (4, 4, 3),
                location.start_index,
                dtype=np.uint8,
            ),
        )
        for location in locations
    )

    unbounded = arm_hold_free_running_evaluation(
        model=_FrontStateDwellModel(),  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
    )
    bounded_model = _FrontStateDwellModel()
    bounded = arm_hold_free_running_evaluation(
        model=bounded_model,  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
        inference_batch_size=2,
    )

    assert bounded["inference_batch_size"] == 2
    assert bounded["inference_segment_cohort_count"] == 3
    comparable = dict(bounded)
    comparable.pop("inference_batch_size")
    comparable.pop("inference_segment_cohort_count")
    assert comparable == unbounded
    assert [batch_size for batch_size, _, _ in bounded_model.received] == [
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    for call_index, (_, available, dwell) in enumerate(bounded_model.received):
        if call_index % 2 == 0:
            assert not available.any()
            assert not dwell.any()
        else:
            assert available.all()
            assert dwell.tolist() == [[1, 1], [1, 1]]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_free_running_rejects_nonpositive_inference_batch_size(
    batch_size: int,
) -> None:
    location = _location(0)
    sample = _sample(
        location,
        prediction_from_state=(0, 0),
        target=(0, 0),
        transition=(False, False),
    )

    with pytest.raises(ValueError, match="inference_batch_size must be positive"):
        arm_hold_free_running_evaluation(
            model=_StatePredictionModel(),  # type: ignore[arg-type]
            locations=(location,),
            sample_at=(sample,).__getitem__,
            device="cpu",
            inference_batch_size=batch_size,
        )


def test_context_free_rollout_reports_error_recovery_and_constant_output() -> None:
    locations = tuple(_location(index) for index in range(5))
    targets = ((0, 0), (0, 0), (1, 0), (1, 0), (0, 0))
    predictions = ((0, 0), (1, 0), (1, 0), (0, 0), (0, 0))
    transitions = (
        (False, False),
        (False, False),
        (True, False),
        (False, False),
        (True, False),
    )
    samples = tuple(
        _sample(
            location,
            prediction_from_state=prediction,
            target=target,
            transition=transition,
        )
        for location, prediction, target, transition in zip(
            locations, predictions, targets, transitions, strict=True
        )
    )

    report = arm_hold_free_running_evaluation(
        model=_StatePredictionModel(),  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
    )
    logits = np.asarray(predictions, dtype=np.float32)
    logits = np.where(logits > 0.5, 8.0, -8.0)
    logits = np.repeat(logits[:, None], 2, axis=1)
    derived_report = arm_hold_offset_zero_evaluation(
        logits=logits[::-1].copy(),
        target=np.stack([sample.arm_hold_target for sample in samples])[::-1],
        valid_mask=np.stack([sample.arm_hold_valid_mask for sample in samples])[
            ::-1
        ],
        loss_mask=np.stack([sample.arm_hold_loss_mask for sample in samples])[
            ::-1
        ],
        transition_mask=np.stack(
            [sample.arm_hold_transition_mask for sample in samples]
        )[::-1],
        metadata=tuple(sample.metadata for sample in reversed(samples)),
        locations=locations,
    )

    assert derived_report == report
    left = report["error_dynamics"]["left"]
    assert left["first_error_offset_steps"]["mean"] == 1.0
    assert left["error_run_steps"]["count"] == 2
    assert left["error_run_steps"]["maximum"] == 1
    assert left["recovered_error_run_count"] == 2
    assert left["unrecovered_error_run_count"] == 0
    assert left["error_run_recovery_rate"] == 1.0
    assert left["predicted_positive_rate"] == pytest.approx(0.4)
    assert left["nontrivial_constant_prediction_segment_count"] == 0
    assert report["window_evaluation"]["pair"]["exact_match_accuracy"] == 0.6
    assert report["sequence_evaluation"]["per_side"]["left"][
        "stable_false_switch_count"
    ] == 2


def test_missing_admitted_window_breaks_feedback_even_with_same_continuity() -> None:
    locations = (_location(0), _location(2))
    samples = tuple(
        _sample(
            location,
            prediction_from_state=(0, 0),
            target=(1, 0),
            transition=(False, False),
        )
        for location in locations
    )
    model = _PreviousModeEchoModel()

    report = arm_hold_free_running_evaluation(
        model=model,  # type: ignore[arg-type]
        locations=locations,
        sample_at=samples.__getitem__,
        device="cpu",
    )

    assert report["rollout_segment_count"] == 2
    assert len(model.received) == 1
    assert not model.received[0][1].any()


def test_free_running_rejects_noncanonical_order_and_sample_mismatch() -> None:
    first = _location(0)
    second = _location(1)
    samples = (
        _sample(
            first,
            prediction_from_state=(0, 0),
            target=(0, 0),
            transition=(False, False),
        ),
        _sample(
            second,
            prediction_from_state=(0, 0),
            target=(0, 0),
            transition=(False, False),
        ),
    )
    model = _StatePredictionModel()

    with pytest.raises(ValueError, match="canonical inventory order"):
        arm_hold_free_running_evaluation(
            model=model,  # type: ignore[arg-type]
            locations=(second, first),
            sample_at=samples.__getitem__,
            device="cpu",
        )
    with pytest.raises(ValueError, match="does not match"):
        arm_hold_free_running_evaluation(
            model=model,  # type: ignore[arg-type]
            locations=(first, second),
            sample_at=lambda index: samples[1 - index],
            device="cpu",
        )


def test_offset_zero_evaluation_rejects_excluded_row_and_metadata_mismatch() -> None:
    location = _location(0)
    sample = _sample(
        location,
        prediction_from_state=(0, 0),
        target=(0, 0),
        transition=(False, False),
    )
    logits = np.zeros((1, 2, 2), dtype=np.float32)
    target = sample.arm_hold_target[None]
    valid = sample.arm_hold_valid_mask[None]
    loss = sample.arm_hold_loss_mask[None]
    transition = sample.arm_hold_transition_mask[None]

    excluded = valid.copy()
    excluded[0, 0, 0] = False
    with pytest.raises(ValueError, match="Every admitted offset-zero row"):
        arm_hold_offset_zero_evaluation(
            logits=logits,
            target=target,
            valid_mask=excluded,
            loss_mask=loss,
            transition_mask=transition,
            metadata=(sample.metadata,),
            locations=(location,),
        )

    mismatched = dict(sample.metadata)
    mismatched["start_index"] = 9
    with pytest.raises(ValueError, match="do not match"):
        arm_hold_offset_zero_evaluation(
            logits=logits,
            target=target,
            valid_mask=valid,
            loss_mask=loss,
            transition_mask=transition,
            metadata=(mismatched,),
            locations=(location,),
        )
