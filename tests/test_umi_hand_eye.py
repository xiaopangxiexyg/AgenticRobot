from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataset_contracts.pose_frames import PoseFrameContract
from utils.hand_eye.camera import TargetConfig, TargetDetector
from utils.hand_eye.transforms import invert_transform, make_transform, transform_distance
from utils.umi_hand_eye.auto_record import AutoRecordConfig, AutoRecordState
from utils.umi_hand_eye.calibration import (
    UmiCalibrationResult,
    UmiCalibrationSample,
    load_result,
    load_samples,
    save_result,
    save_samples,
    solve_umi_hand_eye,
)
from utils.umi_hand_eye.config import load_config
from utils.umi_hand_eye.pose_coverage import PoseCoverageConfig, evaluate_pose_coverage
from utils.umi_hand_eye.run_calibrate import (
    _UmiTargetDetector,
    _apply_umi_sample_guidance,
    _finish_record_session,
    _output_paths,
    _require_result_binding,
    _start_record_session,
    build_parser,
)
from utils.umi_hand_eye.tcp_extrinsic import derive_umi_tcp_extrinsic
from utils.umi_hand_eye.ui import (
    compose_dashboard,
    compose_previews,
    current_window_render_size,
    open_resizable_window,
)
from utils.umi_hand_eye.visualization import compute_live_check_pose, draw_live_check_overlay


def _transform(rpy_deg, translation):
    return make_transform(Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix(), translation)


def _synthetic_samples(count: int = 30):
    rng = np.random.default_rng(42)
    T_camera_to_tracker = _transform([13.0, -21.0, 8.0], [0.035, -0.018, 0.062])
    T_board_to_vive = _transform([-7.0, 16.0, 31.0], [0.45, 0.12, 1.08])
    samples = []
    for index in range(count):
        T_tracker_to_vive = make_transform(
            Rotation.from_rotvec(rng.uniform(-1.1, 1.1, size=3)).as_matrix(),
            rng.uniform([-0.35, -0.25, 0.55], [0.35, 0.25, 1.25]),
        )
        T_board_to_camera = (
            invert_transform(T_camera_to_tracker)
            @ invert_transform(T_tracker_to_vive)
            @ T_board_to_vive
        )
        samples.append(
            UmiCalibrationSample(
                T_tracker_to_vive_world=T_tracker_to_vive,
                T_board_to_camera=T_board_to_camera,
                metadata={"sample_index": index},
            )
        )
    return samples, T_camera_to_tracker, T_board_to_vive


def test_solve_recovers_explicit_umi_transforms():
    samples, expected_camera_to_tracker, expected_board_to_vive = _synthetic_samples()
    created = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc)
    result = solve_umi_hand_eye(
        samples,
        side="left",
        camera_serial="camera-left",
        tracker_name="tracker_1",
        tracker_role="left_wrist",
        validity_seconds=3600.0,
        created_at=created,
        refine=True,
    )
    translation, rotation = transform_distance(
        result.T_camera_to_tracker, expected_camera_to_tracker
    )
    assert translation < 1e-6
    assert rotation < 1e-4
    translation, rotation = transform_distance(
        result.T_board_to_vive_world, expected_board_to_vive
    )
    assert translation < 1e-6
    assert rotation < 1e-4
    assert np.allclose(result.T_vive_world_to_desk, invert_transform(expected_board_to_vive))
    assert result.expires_at_utc == "2026-07-17T02:02:03Z"
    assert result.tcp_extrinsic is not None
    assert result.tcp_extrinsic.tcp_profile == "dmgripper"
    assert result.tcp_extrinsic.resolved_tool_profile == "DMTAC_M"
    assert np.allclose(
        result.T_tcp_to_tracker,
        result.T_camera_to_tracker @ result.tcp_extrinsic.T_tcp_to_camera,
    )
    json.dumps(result.to_dict())


def test_dmgripper_tcp_geometry_uses_dual_urdf_and_side_tool_profile():
    camera_to_tracker = _transform([3.0, -5.0, 7.0], [0.04, -0.02, 0.08])
    left = derive_umi_tcp_extrinsic(
        side="left",
        T_camera_to_tracker=camera_to_tracker,
    )
    right = derive_umi_tcp_extrinsic(
        side="right",
        T_camera_to_tracker=camera_to_tracker,
    )
    assert left.endpoint_frame == "left_arm_link6"
    assert left.camera_frame == "left_camera_rgb"
    assert right.endpoint_frame == "right_arm_link6"
    assert right.camera_frame == "right_camera_rgb"
    assert left.tool_profile_path.endswith("rm65_left_DMTAC_M.json")
    assert right.tool_profile_path.endswith("rm65_right_DMTAC_M.json")
    assert left.T_tcp_to_endpoint[:3, 3].tolist() == pytest.approx([0.0, 0.0, 0.14])
    assert right.T_tcp_to_endpoint[:3, 3].tolist() == pytest.approx([0.0, 0.0, 0.14])
    np.testing.assert_allclose(left.T_tcp_to_camera, right.T_tcp_to_camera, atol=1e-12)
    np.testing.assert_allclose(
        left.T_tcp_to_tracker,
        camera_to_tracker @ left.T_tcp_to_camera,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        left.T_tracker_to_tcp @ left.T_tcp_to_tracker,
        np.eye(4),
        atol=1e-12,
    )
    # Golden values guard the URDF direction and prevent an accidental inverse.
    np.testing.assert_allclose(
        left.T_tcp_to_camera[:3, 3],
        [0.00900002, 0.0246394849, 0.1111588463],
        atol=1e-9,
    )


def test_result_expiry_binding_and_pose_contract():
    created = datetime(2026, 7, 17, tzinfo=timezone.utc)
    result = UmiCalibrationResult(
        T_camera_to_tracker=np.eye(4),
        T_board_to_vive_world=_transform([0, 0, 90], [1, 2, 3]),
        created_at_utc="2026-07-17T00:00:00Z",
        expires_at_utc="2026-07-17T01:00:00Z",
        side="left",
        camera_serial="cam",
        tracker_name="tracker_1",
        tracker_role="left_wrist",
        method="test",
        metrics={},
        sample_count=25,
    )
    result.require_valid(
        when=created + timedelta(minutes=30),
        camera_serial="cam",
        tracker_name="tracker_1",
    )
    with pytest.raises(RuntimeError, match="expired"):
        result.require_valid(when=created + timedelta(hours=2))
    with pytest.raises(RuntimeError, match="camera serial"):
        result.require_valid(when=created, camera_serial="other")

    contract = PoseFrameContract.from_dict(result.pose_frame_contract())
    assert contract.pose_storage_frame == "desk_absolute"
    assert (
        contract.source.metadata["canonical_desk_urdf_frame"]
        == "workspace_near_boundary_frame"
    )
    assert np.allclose(
        contract.source.effective_raw_to_desk_transform("right"),
        result.T_vive_world_to_desk,
    )


def test_sample_result_and_default_config_round_trip(tmp_path):
    samples, _, _ = _synthetic_samples(3)
    samples_path = tmp_path / "samples.json"
    save_samples(samples_path, samples, metadata={"side": "left"})
    restored, metadata = load_samples(samples_path)
    assert len(restored) == 3
    assert metadata == {"side": "left"}
    assert np.allclose(
        restored[0].T_tracker_to_vive_world, samples[0].T_tracker_to_vive_world
    )

    result = UmiCalibrationResult(
        T_camera_to_tracker=np.eye(4),
        T_board_to_vive_world=np.eye(4),
        created_at_utc="2026-07-17T00:00:00Z",
        expires_at_utc="2026-07-17T01:00:00Z",
        side="right",
        camera_serial="cam",
        tracker_name="tracker_2",
        tracker_role="right_wrist",
        method="test",
        metrics={"value": 1.0},
        sample_count=25,
    )
    result_path = tmp_path / "result.json"
    save_result(result_path, result)
    assert load_result(result_path).tracker_name == "tracker_2"
    json.loads(result_path.read_text())

    config_path = Path(__file__).parents[1] / "utils/umi_hand_eye/configs/rm65_dual_wrist_umi_v1.json"
    config = load_config(config_path)
    assert config.validity_seconds == 10800.0
    assert config.camera_resolution == (1280, 720)
    assert config.camera_fps == 15
    assert config.target["allow_aruco_board_fallback"] is True
    assert config.side("left").camera_serial == "261122273924"
    assert config.side("right").camera_serial == "260522278055"
    assert config.side("left").tcp_profile == "dmgripper"
    assert config.urdf_path.name == "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"


def test_live_camera_startup_recovery_defaults_on_and_can_be_disabled():
    defaults = build_parser().parse_args(["--side", "left"])
    disabled = build_parser().parse_args(
        ["--side", "left", "--no-camera-reset-on-failure", "--no-auto-record"]
    )

    assert defaults.camera_startup_warmup_frames == 5
    assert defaults.camera_reset_on_failure is True
    assert defaults.auto_record is True
    assert defaults.min_samples == 10
    assert defaults.auto_stable_sec == pytest.approx(0.5)
    assert defaults.auto_target_still_translation_m == pytest.approx(0.003)
    assert defaults.auto_target_still_rotation_deg == pytest.approx(0.8)
    assert defaults.auto_new_pose_translation_m == pytest.approx(0.04)
    assert defaults.auto_new_pose_rotation_deg == pytest.approx(10.0)
    assert defaults.pose_change_require_both is True
    assert defaults.coverage_translation_m == pytest.approx(0.10)
    assert defaults.coverage_rotation_deg == pytest.approx(30.0)
    assert defaults.coverage_axis_separation_deg == pytest.approx(25.0)
    assert defaults.coverage_axis_observation_rotation_deg == pytest.approx(8.0)
    assert defaults.window_width == 1600
    assert defaults.window_height == 900
    assert defaults.visualize_after_solve is True
    assert defaults.view_result is False
    assert disabled.camera_reset_on_failure is False
    assert disabled.auto_record is False


def test_saved_result_view_parser_supports_single_and_dual_side_overrides():
    single = build_parser().parse_args(
        ["--view-result", "--side", "left", "--result", "left.json"]
    )
    dual = build_parser().parse_args(
        [
            "--view-result",
            "--side",
            "both",
            "--left-result",
            "left.json",
            "--right-result",
            "right.json",
        ]
    )

    assert single.view_result is True
    assert single.result == "left.json"
    assert dual.left_result == "left.json"
    assert dual.right_result == "right.json"


def test_saved_result_view_rejects_a_wrong_side_device_binding():
    result = SimpleNamespace(side="left", camera_serial="cam-left", tracker_name="tracker_1")
    side_config = SimpleNamespace(camera_serial="cam-left", tracker_name="tracker_1")
    _require_result_binding(result, side="left", side_config=side_config)

    with pytest.raises(RuntimeError, match="camera serial"):
        _require_result_binding(
            result,
            side="left",
            side_config=SimpleNamespace(camera_serial="other", tracker_name="tracker_1"),
        )


def test_umi_target_overrides_are_exposed_by_the_live_parser():
    args = build_parser().parse_args(
        [
            "--side",
            "left",
            "--target-type",
            "charuco",
            "--dictionary",
            "DICT_6X6_100",
            "--charuco-squares-x",
            "5",
            "--charuco-squares-y",
            "7",
            "--charuco-square-length-m",
            "0.035",
            "--charuco-marker-length-m",
            "0.026",
            "--min-charuco-corners",
            "12",
        ]
    )
    assert args.target_type == "charuco"
    assert args.dictionary == "DICT_6X6_100"
    assert args.charuco_squares_x == 5
    assert args.charuco_squares_y == 7
    assert args.charuco_square_length_m == pytest.approx(0.035)
    assert args.charuco_marker_length_m == pytest.approx(0.026)
    assert args.min_charuco_corners == 12


def test_auto_record_state_requires_still_visible_new_pose():
    pose = np.eye(4)
    state = AutoRecordState(
        AutoRecordConfig(
            stable_sec=3.0,
            still_translation_m=0.0015,
            still_rotation_deg=0.25,
            new_pose_translation_m=0.03,
            new_pose_rotation_deg=8.0,
            cooldown_sec=1.0,
        )
    )
    should_record, status = state.update(pose, target_visible=True, now=0.0)
    assert not should_record
    assert "waiting" in status
    should_record, status = state.update(pose, target_visible=False, now=2.0)
    assert not should_record
    assert "not visible" in status
    should_record, status = state.update(pose, target_visible=True, now=3.1)
    assert not should_record
    assert "still" in status
    should_record, status = state.update(pose, target_visible=True, now=5.1)
    assert should_record
    assert "recording" in status

    state.mark_recorded(pose, now=5.1)
    should_record, status = state.update(pose, target_visible=True, now=6.2)
    assert not should_record
    assert "pose too similar" in status

    moved_pose = pose.copy()
    moved_pose[0, 3] = 0.04
    should_record, status = state.update(moved_pose, target_visible=True, now=7.0)
    assert not should_record
    assert "moving" in status


def test_auto_record_dwell_resets_when_target_moves_or_disappears():
    state = AutoRecordState(
        AutoRecordConfig(
            stable_sec=0.5,
            still_translation_m=0.0015,
            still_rotation_deg=0.25,
            target_still_translation_m=0.003,
            target_still_rotation_deg=0.8,
        )
    )
    tracker_pose = np.eye(4)
    target_pose = np.eye(4)

    should_record, _ = state.update(
        tracker_pose,
        target_visible=True,
        target_pose=target_pose,
        now=0.0,
    )
    assert not should_record

    moved_target = target_pose.copy()
    moved_target[0, 3] = 0.01
    should_record, status = state.update(
        tracker_pose,
        target_visible=True,
        target_pose=moved_target,
        now=0.6,
    )
    assert not should_record
    assert "target moving" in status

    should_record, _ = state.update(
        tracker_pose,
        target_visible=True,
        target_pose=moved_target,
        now=1.2,
    )
    assert should_record

    should_record, status = state.update(
        tracker_pose,
        target_visible=False,
        now=1.3,
    )
    assert not should_record
    assert "timer reset" in status
    should_record, status = state.update(
        tracker_pose,
        target_visible=True,
        target_pose=moved_target,
        now=1.7,
    )
    assert not should_record
    assert "waiting" in status


def test_auto_record_requires_change_from_every_recorded_pose_in_both_components():
    state = AutoRecordState(
        AutoRecordConfig(
            stable_sec=0.5,
            new_pose_translation_m=0.04,
            new_pose_rotation_deg=10.0,
            require_both_pose_changes=True,
        )
    )
    origin = np.eye(4)
    state.mark_recorded(origin, now=0.0)

    translation_only = _transform([0.0, 0.0, 0.0], [0.06, 0.0, 0.0])
    state.update(translation_only, target_visible=True, now=1.0)
    should_record, status = state.update(
        translation_only, target_visible=True, now=1.6
    )
    assert not should_record
    assert "pose too similar" in status

    diverse = _transform([15.0, 0.0, 0.0], [0.06, 0.0, 0.0])
    state.update(diverse, target_visible=True, now=2.0)
    should_record, _ = state.update(diverse, target_visible=True, now=2.6)
    assert should_record
    state.mark_recorded(diverse, now=2.6)

    near_origin = _transform([12.0, 0.0, 0.0], [0.045, 0.0, 0.0])
    state.update(near_origin, target_visible=True, now=3.0)
    should_record, _ = state.update(near_origin, target_visible=True, now=3.6)
    assert not should_record


def test_pose_coverage_requires_count_span_and_nonparallel_rotation_axes():
    config = PoseCoverageConfig(
        min_samples=10,
        min_translation_span_m=0.10,
        min_rotation_span_deg=30.0,
        min_rotation_axis_separation_deg=25.0,
    )
    weak = [
        _transform([0.0, 0.0, 0.0], [0.012 * index, 0.0, 0.0])
        for index in range(10)
    ]
    weak_coverage = evaluate_pose_coverage(weak, config)
    assert not weak_coverage.ready
    assert weak_coverage.translation_span_m >= 0.10
    assert "rotation" in weak_coverage.hint

    diverse = [
        _transform(
            [4.0 * index, 12.0 * (index % 3), 3.0 * (index % 2)],
            [0.012 * index, 0.004 * (index % 3), 0.0],
        )
        for index in range(10)
    ]
    coverage = evaluate_pose_coverage(diverse, config)
    assert coverage.ready
    assert coverage.rotation_span_deg >= 30.0
    assert coverage.has_rotation_axis_diversity
    assert "READY" in coverage.overlay_lines(config)[-1]


def test_dual_side_parser_and_adaptive_preview_layout():
    cv2 = pytest.importorskip("cv2")
    args = build_parser().parse_args(["--side", "both"])
    assert args.side == "both"
    left = np.zeros((720, 1280, 3), dtype=np.uint8)
    right = np.ones((720, 1280, 3), dtype=np.uint8)
    combined = compose_previews(
        cv2,
        {"left": left, "right": right},
        max_width=1600,
        max_height=900,
    )
    assert combined.shape == (900, 1600, 3)
    assert np.count_nonzero(combined[:200]) == 0

    enlarged = compose_previews(
        cv2,
        {"left": left},
        max_width=1600,
        max_height=900,
    )
    assert enlarged.shape == (900, 1600, 3)

    dashboard = compose_dashboard(
        cv2,
        {"left": left, "right": right},
        {
            "left": ["target: ok", "samples 4/10"],
            "right": ["target: not detected", "NEXT: show the board"],
        },
        {"left": False, "right": False},
        width=1600,
        height=900,
    )
    assert dashboard.shape == (900, 1600, 3)
    assert np.count_nonzero(dashboard[700:]) > 0

    class FakeCv2:
        @staticmethod
        def getWindowImageRect(name):
            assert name == "preview"
            return 20, 30, 1234, 678

    assert current_window_render_size(
        FakeCv2,
        "preview",
        fallback_width=1600,
        fallback_height=900,
    ) == (1234, 678)

    class FakeWindowCv2:
        WINDOW_NORMAL = 0
        WINDOW_FREERATIO = 256
        WND_PROP_ASPECT_RATIO = 2
        calls = []

        @classmethod
        def namedWindow(cls, name, flags):
            cls.calls.append(("namedWindow", name, flags))

        @classmethod
        def setWindowProperty(cls, name, prop, value):
            cls.calls.append(("setWindowProperty", name, prop, value))

        @classmethod
        def imshow(cls, name, image):
            cls.calls.append(("imshow", name, image.shape))

        @classmethod
        def resizeWindow(cls, name, width, height):
            cls.calls.append(("resizeWindow", name, width, height))

        @classmethod
        def waitKey(cls, delay):
            cls.calls.append(("waitKey", delay))
            return -1

    open_resizable_window(FakeWindowCv2, "preview", width=1600, height=900)
    assert [call[0] for call in FakeWindowCv2.calls] == [
        "namedWindow",
        "setWindowProperty",
        "imshow",
        "resizeWindow",
        "waitKey",
    ]
    assert FakeWindowCv2.calls[0][2] == FakeWindowCv2.WINDOW_FREERATIO
    assert FakeWindowCv2.calls[1] == (
        "setWindowProperty",
        "preview",
        FakeWindowCv2.WND_PROP_ASPECT_RATIO,
        FakeWindowCv2.WINDOW_FREERATIO,
    )
    assert FakeWindowCv2.calls[2][2] == (900, 1600, 3)

    paths = _output_paths(Path("records"), "session", "left", dual=True)
    assert paths.samples == Path("records/session_left_samples.json")
    assert paths.result == Path("records/session_left.json")
    assert paths.latest == Path("records/latest_left.json")


def test_umi_ten_sample_guidance_does_not_keep_shared_25_sample_warning():
    metrics = {
        "quality_warnings": [
            "sample_count below 25; collect more diverse poses.",
            "robot rotation span is small; add wrist rotations around multiple axes.",
            "collect at least 25 samples for high-precision calibration; 40 is better.",
        ],
        "sample_plan": {
            "recommended_sample_count": "25-40",
            "warnings": [
                "collect at least 25 samples for high-precision calibration; 40 is better."
            ],
        },
    }
    _apply_umi_sample_guidance(metrics, min_samples=10)
    assert metrics["sample_plan"]["recommended_sample_count"].startswith(">=10")
    assert metrics["sample_plan"]["warnings"] == []
    assert metrics["quality_warnings"] == [
        "robot rotation span is small; add wrist rotations around multiple axes."
    ]


def test_dual_bursts_are_scheduled_independently_without_blocking_main_loop():
    class PendingFuture:
        def done(self):
            return False

    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, **kwargs):
            future = PendingFuture()
            self.calls.append((function, kwargs, future))
            return future

    executor = FakeExecutor()
    burst_config = object()
    left = SimpleNamespace(
        side="left",
        capture_future=None,
        capture_automatic=False,
        camera="left-camera",
        detector="left-detector",
        tracker="left-tracker",
        samples=[],
    )
    right = SimpleNamespace(
        side="right",
        capture_future=None,
        capture_automatic=False,
        camera="right-camera",
        detector="right-detector",
        tracker="right-tracker",
        samples=[],
    )

    assert _start_record_session(
        left,
        executor=executor,
        burst_config=burst_config,
        automatic=True,
    )
    assert _start_record_session(
        right,
        executor=executor,
        burst_config=burst_config,
        automatic=True,
    )
    assert len(executor.calls) == 2
    assert left.capture_future is executor.calls[0][2]
    assert right.capture_future is executor.calls[1][2]
    assert executor.calls[0][1]["camera"] == "left-camera"
    assert executor.calls[1][1]["camera"] == "right-camera"
    assert not _finish_record_session(left)
    assert not _finish_record_session(right)


def test_umi_rejects_single_marker_charuco_board_fallback(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    detector = _UmiTargetDetector(
        TargetConfig(
            target_type="charuco",
            dictionary="DICT_6X6_100",
            charuco_squares_x=5,
            charuco_squares_y=7,
            charuco_square_length_m=0.035,
            charuco_marker_length_m=0.026,
            min_charuco_corners=12,
            allow_aruco_board_fallback=True,
        ),
        min_aruco_board_markers=4,
    )

    def fake_detect(self, *args, draw=False, **kwargs):
        self._last_status = "target: ok aruco-board markers=1 charuco=0/12"
        return SimpleNamespace(marker_ids=[7]), np.zeros((120, 160, 3), dtype=np.uint8)

    monkeypatch.setattr(TargetDetector, "detect", fake_detect)
    detection, preview = detector.detect(
        np.zeros((120, 160, 3), dtype=np.uint8),
        np.eye(3),
        np.zeros(5),
        draw=True,
    )
    assert detection is None
    assert "markers=1/4" in detector.last_status
    assert np.count_nonzero(preview) == 0
    assert detector._cv2 is cv2


def test_charuco_detector_reports_actual_corner_count():
    cv2 = pytest.importorskip("cv2")
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
    board = aruco.CharucoBoard((5, 7), 0.035, 0.026, dictionary)
    gray = board.generateImage((500, 700), marginSize=20)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    intrinsics = np.array(
        [[700.0, 0.0, 250.0], [0.0, 700.0, 350.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)

    detector = TargetDetector(
        TargetConfig(
            target_type="charuco",
            dictionary="DICT_6X6_100",
            charuco_squares_x=5,
            charuco_squares_y=7,
            charuco_square_length_m=0.035,
            charuco_marker_length_m=0.026,
            min_charuco_corners=12,
        )
    )
    detection, _ = detector.detect(frame, intrinsics, distortion, draw=False)
    assert detection is not None
    assert detection.corner_count == 24
    assert "markers=17" in detector.last_status
    assert "charuco=24/24" in detector.last_status

    strict_detector = TargetDetector(
        TargetConfig(
            target_type="charuco",
            dictionary="DICT_6X6_100",
            charuco_squares_x=5,
            charuco_squares_y=7,
            charuco_square_length_m=0.035,
            charuco_marker_length_m=0.026,
            min_charuco_corners=25,
            allow_aruco_board_fallback=False,
        )
    )
    strict_detection, _ = strict_detector.detect(frame, intrinsics, distortion, draw=False)
    assert strict_detection is None
    assert "charuco corners=24/25" in strict_detector.last_status

    fallback_detector = TargetDetector(
        TargetConfig(
            target_type="charuco",
            dictionary="DICT_6X6_100",
            charuco_squares_x=5,
            charuco_squares_y=7,
            charuco_square_length_m=0.035,
            charuco_marker_length_m=0.026,
            min_charuco_corners=25,
            allow_aruco_board_fallback=True,
        )
    )
    fallback_detection, _ = fallback_detector.detect(
        frame, intrinsics, distortion, draw=False
    )
    assert fallback_detection is not None
    assert sorted(fallback_detection.marker_ids) == list(range(17))
    assert fallback_detection.corner_count == 68
    assert "ok aruco-board markers=17 charuco=24/25 (board_max=24)" in (
        fallback_detector.last_status
    )


def test_live_check_reports_tracker_pose_in_calibrated_desk_frame():
    T_camera_to_tracker = _transform([5.0, -8.0, 12.0], [0.04, -0.02, 0.06])
    T_desk_to_vive = _transform([0.0, 0.0, 30.0], [0.4, 0.1, 0.8])
    T_tracker_to_vive = _transform([-10.0, 15.0, 40.0], [0.6, -0.1, 1.0])
    result = UmiCalibrationResult(
        T_camera_to_tracker=T_camera_to_tracker,
        T_board_to_vive_world=T_desk_to_vive,
        created_at_utc="2026-07-17T00:00:00Z",
        expires_at_utc="2026-07-17T01:00:00Z",
        side="left",
        camera_serial="cam",
        tracker_name="tracker_1",
        tracker_role="left_wrist",
        method="test",
        metrics={},
        sample_count=25,
    )

    pose = compute_live_check_pose(result, T_tracker_to_vive)
    expected_tracker_to_desk = invert_transform(T_desk_to_vive) @ T_tracker_to_vive
    expected_camera_to_desk = expected_tracker_to_desk @ T_camera_to_tracker

    np.testing.assert_allclose(pose.T_tracker_to_desk, expected_tracker_to_desk, atol=1e-12)
    np.testing.assert_allclose(pose.T_camera_to_desk, expected_camera_to_desk, atol=1e-12)
    np.testing.assert_allclose(
        pose.T_desk_to_camera @ pose.T_camera_to_desk,
        np.eye(4),
        atol=1e-12,
    )
    np.testing.assert_allclose(pose.tracker_xyz_m, expected_tracker_to_desk[:3, 3])
    np.testing.assert_allclose(pose.camera_xyz_m, expected_camera_to_desk[:3, 3])
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", pose.camera_rpy_deg, degrees=True).as_matrix(),
        expected_camera_to_desk[:3, :3],
        atol=1e-12,
    )


def test_live_check_overlay_renders_camera_and_tracker_desk_panel():
    cv2 = pytest.importorskip("cv2")
    result = UmiCalibrationResult(
        T_camera_to_tracker=np.eye(4),
        T_board_to_vive_world=np.eye(4),
        created_at_utc="2026-07-17T00:00:00Z",
        expires_at_utc="2026-07-17T01:00:00Z",
        side="left",
        camera_serial="cam",
        tracker_name="tracker_1",
        tracker_role="left_wrist",
        method="test",
        metrics={},
        sample_count=25,
    )
    pose = compute_live_check_pose(
        result,
        _transform([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    )

    rendered = draw_live_check_overlay(
        cv2,
        np.zeros((480, 640, 3), dtype=np.uint8),
        intrinsics_matrix=np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coeffs=np.zeros(5),
        pose=pose,
        tracker_name="tracker_1",
        board_residual=(0.002, 0.3),
    )

    assert rendered.shape == (480, 1000, 3)
    assert np.count_nonzero(rendered) > 0
