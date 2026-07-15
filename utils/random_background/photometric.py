from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


IlluminationKind = Literal["none", "vignette", "soft_shadow", "low_frequency"]
QualityKind = Literal[
    "none",
    "gaussian_noise",
    "poisson_noise",
    "jpeg",
    "defocus_blur",
    "motion_blur",
    "downsample",
    "combo",
]

_ILLUMINATION_KINDS = ("vignette", "soft_shadow", "low_frequency")
_QUALITY_KINDS = (
    "gaussian_noise",
    "poisson_noise",
    "jpeg",
    "defocus_blur",
    "motion_blur",
    "downsample",
    "combo",
)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("photometric image augmentation requires opencv-python.") from exc
    return cv2


def _camera_hash(camera_key: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(str(camera_key)))


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


def _check_range(name: str, value: tuple[float, float], *, lower_bound: float = 0.0) -> None:
    low, high = (float(item) for item in value)
    if low < lower_bound or high < low:
        raise ValueError(f"{name} must satisfy {lower_bound} <= low <= high")


@dataclass(frozen=True)
class PhotometricAugmentConfig:
    """Non-geometric full-image augmentation for training-time camera robustness.

    Geometric movement and occlusion/dropout are deliberately absent.  The
    strongest values are sampled only intermittently, and the quality branch
    chooses at most one degradation profile for each episode/camera/epoch.
    """

    enabled: bool = True
    # ``None`` inherits RandomBackgroundAugmentConfig.enable_epoch_variation.
    enable_epoch_variation: bool | None = None

    exposure_probability: float = 0.55
    exposure_gain_range: tuple[float, float] = (0.55, 1.55)
    contrast_probability: float = 0.45
    contrast_gain_range: tuple[float, float] = (0.65, 1.40)
    gamma_probability: float = 0.30
    gamma_range: tuple[float, float] = (0.65, 1.55)
    white_balance_probability: float = 0.35
    warm_bgr_gains: tuple[float, float, float] = (0.85, 1.00, 1.15)
    cool_bgr_gains: tuple[float, float, float] = (1.15, 1.00, 0.50)
    saturation_probability: float = 0.25
    saturation_gain_range: tuple[float, float] = (0.70, 1.00)
    hue_probability: float = 0.10
    hue_delta_deg: float = 7.0

    illumination_probability: float = 0.35
    illumination_kinds: tuple[IlluminationKind, ...] = _ILLUMINATION_KINDS
    vignette_max_strength: float = 0.38
    soft_shadow_max_strength: float = 0.46
    low_frequency_light_max_amplitude: float = 0.24

    quality_probability: float = 0.45
    quality_kinds: tuple[QualityKind, ...] = _QUALITY_KINDS
    gaussian_noise_sigma_max: float = 8.0
    poisson_peak_range: tuple[float, float] = (45.0, 160.0)
    jpeg_quality_range: tuple[int, int] = (25, 90)
    defocus_blur_sigma_max: float = 1.8
    motion_blur_length_range: tuple[int, int] = (3, 11)
    downsample_scale_range: tuple[float, float] = (0.45, 0.80)
    combo_noise_sigma: float = 4.0
    combo_blur_sigma: float = 0.70
    combo_jpeg_quality: int = 35

    def __post_init__(self) -> None:
        for name in (
            "exposure_probability",
            "contrast_probability",
            "gamma_probability",
            "white_balance_probability",
            "saturation_probability",
            "hue_probability",
            "illumination_probability",
            "quality_probability",
        ):
            _check_probability(f"PhotometricAugmentConfig.{name}", float(getattr(self, name)))
        for name in (
            "exposure_gain_range",
            "contrast_gain_range",
            "gamma_range",
            "saturation_gain_range",
            "poisson_peak_range",
            "downsample_scale_range",
        ):
            _check_range(f"PhotometricAugmentConfig.{name}", getattr(self, name))
        if int(self.jpeg_quality_range[0]) < 1 or int(self.jpeg_quality_range[1]) > 100:
            raise ValueError("PhotometricAugmentConfig.jpeg_quality_range must be within [1, 100]")
        if int(self.jpeg_quality_range[0]) > int(self.jpeg_quality_range[1]):
            raise ValueError("PhotometricAugmentConfig.jpeg_quality_range must satisfy low <= high")
        if int(self.motion_blur_length_range[0]) < 1 or int(self.motion_blur_length_range[0]) > int(
            self.motion_blur_length_range[1]
        ):
            raise ValueError("PhotometricAugmentConfig.motion_blur_length_range must be positive and ordered")
        for name in (
            "hue_delta_deg",
            "vignette_max_strength",
            "soft_shadow_max_strength",
            "low_frequency_light_max_amplitude",
            "gaussian_noise_sigma_max",
            "defocus_blur_sigma_max",
            "combo_noise_sigma",
            "combo_blur_sigma",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"PhotometricAugmentConfig.{name} must be non-negative")
        if not 1 <= int(self.combo_jpeg_quality) <= 100:
            raise ValueError("PhotometricAugmentConfig.combo_jpeg_quality must be within [1, 100]")
        for name in ("warm_bgr_gains", "cool_bgr_gains"):
            gains = tuple(float(item) for item in getattr(self, name))
            if len(gains) != 3 or any(item <= 0.0 for item in gains):
                raise ValueError(f"PhotometricAugmentConfig.{name} must contain three positive BGR gains")
        if not self.illumination_kinds or any(item not in _ILLUMINATION_KINDS for item in self.illumination_kinds):
            raise ValueError(f"illumination_kinds must be a non-empty subset of {_ILLUMINATION_KINDS!r}")
        if not self.quality_kinds or any(item not in _QUALITY_KINDS for item in self.quality_kinds):
            raise ValueError(f"quality_kinds must be a non-empty subset of {_QUALITY_KINDS!r}")


@dataclass(frozen=True)
class _PhotometricPlan:
    exposure_gain: float
    contrast_gain: float
    gamma: float
    white_balance_bgr: tuple[float, float, float]
    saturation_gain: float
    hue_delta_deg: float
    illumination_kind: IlluminationKind
    illumination_strength: float
    illumination_seed: int
    quality_kind: QualityKind
    gaussian_noise_sigma: float
    poisson_peak: float
    jpeg_quality: int
    defocus_blur_sigma: float
    motion_blur_length: int
    motion_blur_angle_deg: float
    downsample_scale: float
    noise_seed: int


@dataclass(frozen=True)
class _PreparedPhotometricPlan:
    plan: _PhotometricPlan
    illumination_gain: np.ndarray | None
    motion_kernel: np.ndarray | None


class PhotometricImageAugmenter:
    """Caches deterministic non-geometric image plans inside one DataLoader worker."""

    def __init__(self, config: PhotometricAugmentConfig | None = None, *, cache_size: int = 128):
        self.config = config if config is not None else PhotometricAugmentConfig()
        self.cache_size = max(1, int(cache_size))
        self._plans: OrderedDict[tuple[int, int, int, str, int], _PhotometricPlan] = OrderedDict()
        self._prepared: OrderedDict[tuple[int, int, int, str, int], _PreparedPhotometricPlan] = OrderedDict()

    @staticmethod
    def _key(*, width: int, height: int, episode_index: int, camera_key: str, epoch: int) -> tuple[int, int, int, str, int]:
        return (int(width), int(height), int(episode_index), str(camera_key), int(epoch))

    @staticmethod
    def _seed(*, global_seed: int, episode_index: int, camera_key: str, epoch: int, suffix: str) -> int:
        return (
            int(global_seed)
            + int(episode_index) * 100003
            + _camera_hash(f"{camera_key}:{suffix}") * 1009
            + int(epoch) * 10000019
        )

    @staticmethod
    def _draw_gain(random: np.random.Generator, probability: float, value_range: tuple[float, float]) -> float:
        if float(random.random()) >= float(probability):
            return 1.0
        return float(random.uniform(float(value_range[0]), float(value_range[1])))

    def _build_plan(self, *, episode_index: int, camera_key: str, epoch: int, global_seed: int) -> _PhotometricPlan:
        random = np.random.default_rng(
            self._seed(
                global_seed=int(global_seed),
                episode_index=int(episode_index),
                camera_key=str(camera_key),
                epoch=int(epoch),
                suffix="photometric_plan",
            )
        )
        config = self.config
        exposure_gain = self._draw_gain(random, config.exposure_probability, config.exposure_gain_range)
        contrast_gain = self._draw_gain(random, config.contrast_probability, config.contrast_gain_range)
        gamma = self._draw_gain(random, config.gamma_probability, config.gamma_range)
        if float(random.random()) < float(config.white_balance_probability):
            white_balance_bgr = (
                config.warm_bgr_gains if float(random.random()) < 0.5 else config.cool_bgr_gains
            )
        else:
            white_balance_bgr = (1.0, 1.0, 1.0)
        saturation_gain = self._draw_gain(random, config.saturation_probability, config.saturation_gain_range)
        hue_delta_deg = (
            float(random.uniform(-config.hue_delta_deg, config.hue_delta_deg))
            if float(random.random()) < float(config.hue_probability)
            else 0.0
        )

        illumination_kind: IlluminationKind = "none"
        illumination_strength = 0.0
        illumination_seed = int(random.integers(0, 2**63 - 1))
        if float(random.random()) < float(config.illumination_probability):
            illumination_kind = str(random.choice(config.illumination_kinds))  # type: ignore[assignment]
            maximum = {
                "vignette": config.vignette_max_strength,
                "soft_shadow": config.soft_shadow_max_strength,
                "low_frequency": config.low_frequency_light_max_amplitude,
            }[illumination_kind]
            illumination_strength = float(random.uniform(0.05 * maximum, maximum)) if maximum > 0.0 else 0.0

        quality_kind: QualityKind = "none"
        gaussian_noise_sigma = 0.0
        poisson_peak = 0.0
        jpeg_quality = 100
        defocus_blur_sigma = 0.0
        motion_blur_length = 0
        motion_blur_angle_deg = 0.0
        downsample_scale = 1.0
        noise_seed = int(random.integers(0, 2**63 - 1))
        if float(random.random()) < float(config.quality_probability):
            quality_kind = str(random.choice(config.quality_kinds))  # type: ignore[assignment]
            if quality_kind == "gaussian_noise":
                gaussian_noise_sigma = float(random.uniform(0.20 * config.gaussian_noise_sigma_max, config.gaussian_noise_sigma_max))
            elif quality_kind == "poisson_noise":
                poisson_peak = float(random.uniform(*config.poisson_peak_range))
            elif quality_kind == "jpeg":
                jpeg_quality = int(random.integers(int(config.jpeg_quality_range[0]), int(config.jpeg_quality_range[1]) + 1))
            elif quality_kind == "defocus_blur":
                defocus_blur_sigma = float(random.uniform(0.20 * config.defocus_blur_sigma_max, config.defocus_blur_sigma_max))
            elif quality_kind == "motion_blur":
                motion_blur_length = int(
                    random.integers(int(config.motion_blur_length_range[0]), int(config.motion_blur_length_range[1]) + 1)
                )
                motion_blur_angle_deg = float(random.uniform(0.0, 180.0))
            elif quality_kind == "downsample":
                downsample_scale = float(random.uniform(*config.downsample_scale_range))
            elif quality_kind == "combo":
                gaussian_noise_sigma = float(config.combo_noise_sigma)
                defocus_blur_sigma = float(config.combo_blur_sigma)
                jpeg_quality = int(config.combo_jpeg_quality)

        return _PhotometricPlan(
            exposure_gain=exposure_gain,
            contrast_gain=contrast_gain,
            gamma=gamma,
            white_balance_bgr=tuple(float(item) for item in white_balance_bgr),
            saturation_gain=saturation_gain,
            hue_delta_deg=hue_delta_deg,
            illumination_kind=illumination_kind,
            illumination_strength=illumination_strength,
            illumination_seed=illumination_seed,
            quality_kind=quality_kind,
            gaussian_noise_sigma=gaussian_noise_sigma,
            poisson_peak=poisson_peak,
            jpeg_quality=jpeg_quality,
            defocus_blur_sigma=defocus_blur_sigma,
            motion_blur_length=motion_blur_length,
            motion_blur_angle_deg=motion_blur_angle_deg,
            downsample_scale=downsample_scale,
            noise_seed=noise_seed,
        )

    def _plan(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        global_seed: int,
    ) -> tuple[tuple[int, int, int, str, int], _PhotometricPlan]:
        key = self._key(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = self._build_plan(
                episode_index=int(episode_index),
                camera_key=str(camera_key),
                epoch=int(epoch),
                global_seed=int(global_seed),
            )
            self._plans[key] = plan
            if len(self._plans) > self.cache_size:
                stale_key, _ = self._plans.popitem(last=False)
                self._prepared.pop(stale_key, None)
        else:
            self._plans.move_to_end(key)
        return key, plan

    @staticmethod
    def _motion_kernel(length: int, angle_deg: float) -> np.ndarray | None:
        if int(length) <= 1:
            return None
        cv2 = _require_cv2()
        size = max(3, int(length) | 1)
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        radius = center
        radians = np.deg2rad(float(angle_deg))
        start = (int(round(center - radius * np.cos(radians))), int(round(center - radius * np.sin(radians))))
        end = (int(round(center + radius * np.cos(radians))), int(round(center + radius * np.sin(radians))))
        cv2.line(kernel, start, end, 1.0, 1, cv2.LINE_AA)
        kernel /= max(float(kernel.sum()), 1e-6)
        return kernel

    @staticmethod
    def _illumination_gain(plan: _PhotometricPlan, *, width: int, height: int) -> np.ndarray | None:
        if plan.illumination_kind == "none" or plan.illumination_strength <= 0.0:
            return None
        if plan.illumination_kind == "vignette":
            y, x = np.ogrid[-1.0:1.0:complex(int(height)), -1.0:1.0:complex(int(width))]
            radius2 = np.minimum(1.0, x * x + y * y)
            return np.clip(1.0 - float(plan.illumination_strength) * radius2, 0.05, 1.0).astype(np.float32)
        cv2 = _require_cv2()
        random = np.random.default_rng(int(plan.illumination_seed))
        if plan.illumination_kind == "soft_shadow":
            x = np.linspace(-1.0, 1.0, int(width), dtype=np.float32)[None, :]
            y = np.linspace(-1.0, 1.0, int(height), dtype=np.float32)[:, None]
            center_x = float(random.uniform(-0.55, 0.55))
            center_y = float(random.uniform(-0.55, 0.55))
            spread_x = float(random.uniform(0.35, 0.95))
            spread_y = float(random.uniform(0.50, 1.30))
            field = np.exp(-((x - center_x) ** 2 / spread_x + (y - center_y) ** 2 / spread_y))
            return np.clip(1.0 - float(plan.illumination_strength) * field, 0.05, 1.0).astype(np.float32)
        coarse = random.normal(0.0, 1.0, size=(8, 8)).astype(np.float32)
        field = cv2.resize(coarse, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(int(width), int(height)) / 12.0)
        field -= float(field.mean())
        field /= max(float(np.max(np.abs(field))), 1e-6)
        return np.clip(1.0 + float(plan.illumination_strength) * field, 0.05, 2.0).astype(np.float32)

    def _prepared_plan(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        global_seed: int,
    ) -> _PreparedPhotometricPlan:
        key, plan = self._plan(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
            global_seed=int(global_seed),
        )
        prepared = self._prepared.get(key)
        if prepared is None:
            prepared = _PreparedPhotometricPlan(
                plan=plan,
                illumination_gain=self._illumination_gain(plan, width=int(width), height=int(height)),
                motion_kernel=self._motion_kernel(plan.motion_blur_length, plan.motion_blur_angle_deg),
            )
            self._prepared[key] = prepared
            if len(self._prepared) > self.cache_size:
                self._prepared.popitem(last=False)
        else:
            self._prepared.move_to_end(key)
        return prepared

    @staticmethod
    def _clip_u8(value: np.ndarray) -> np.ndarray:
        return np.clip(value, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _frame_seed(plan: _PhotometricPlan, frame_index: int) -> int:
        return int(plan.noise_seed) + int(frame_index) * 9176

    def _apply_global(self, image: np.ndarray, prepared: _PreparedPhotometricPlan) -> np.ndarray:
        plan = prepared.plan
        source = np.asarray(image, dtype=np.uint8)
        exposure_active = abs(float(plan.exposure_gain) - 1.0) > 1e-6
        contrast_active = abs(float(plan.contrast_gain) - 1.0) > 1e-6
        gamma_active = abs(float(plan.gamma) - 1.0) > 1e-6
        white_balance = np.asarray(plan.white_balance_bgr, dtype=np.float32).reshape(1, 1, 3)
        white_balance_active = bool(np.any(np.abs(white_balance - 1.0) > 1e-6))
        if exposure_active or contrast_active or gamma_active or white_balance_active or prepared.illumination_gain is not None:
            out = source.astype(np.float32) / 255.0
            if exposure_active:
                out *= float(plan.exposure_gain)
            if contrast_active:
                out = (out - 0.5) * float(plan.contrast_gain) + 0.5
            if gamma_active:
                out = np.power(np.clip(out, 0.0, 1.0), float(plan.gamma))
            if white_balance_active:
                out *= white_balance
            if prepared.illumination_gain is not None:
                out *= prepared.illumination_gain[..., None]
            result = self._clip_u8(out * 255.0)
        else:
            # Most per-episode plans leave several groups disabled. Avoid a
            # float round-trip when only HSV or camera-quality work remains.
            result = source.copy()
        if abs(float(plan.saturation_gain) - 1.0) <= 1e-6 and abs(float(plan.hue_delta_deg)) <= 1e-6:
            return result
        cv2 = _require_cv2()
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        if abs(float(plan.saturation_gain) - 1.0) > 1e-6:
            hsv[..., 1] = np.clip(
                hsv[..., 1].astype(np.float32) * float(plan.saturation_gain), 0.0, 255.0
            ).astype(np.uint8)
        if abs(float(plan.hue_delta_deg)) > 1e-6:
            delta = int(round(float(plan.hue_delta_deg) / 2.0))
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + delta) % 180).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_quality(
        self,
        image: np.ndarray,
        prepared: _PreparedPhotometricPlan,
        *,
        frame_index: int,
    ) -> np.ndarray:
        plan = prepared.plan
        kind = plan.quality_kind
        if kind == "none":
            return image
        cv2 = _require_cv2()
        if kind in ("gaussian_noise", "combo"):
            random = np.random.default_rng(self._frame_seed(plan, frame_index))
            image = self._clip_u8(
                image.astype(np.float32)
                + random.normal(0.0, float(plan.gaussian_noise_sigma), size=image.shape).astype(np.float32)
            )
        if kind == "poisson_noise":
            random = np.random.default_rng(self._frame_seed(plan, frame_index))
            normalized = image.astype(np.float32) / 255.0
            image = self._clip_u8(
                random.poisson(normalized * float(plan.poisson_peak)).astype(np.float32)
                / float(plan.poisson_peak)
                * 255.0
            )
        if kind in ("defocus_blur", "combo") and float(plan.defocus_blur_sigma) > 0.0:
            image = cv2.GaussianBlur(
                image,
                (0, 0),
                sigmaX=float(plan.defocus_blur_sigma),
                sigmaY=float(plan.defocus_blur_sigma),
                borderType=cv2.BORDER_REFLECT,
            )
        if kind == "motion_blur" and prepared.motion_kernel is not None:
            image = cv2.filter2D(image, -1, prepared.motion_kernel, borderType=cv2.BORDER_REFLECT)
        if kind == "downsample":
            height, width = image.shape[:2]
            small_width = max(1, int(round(width * float(plan.downsample_scale))))
            small_height = max(1, int(round(height * float(plan.downsample_scale))))
            image = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        if kind in ("jpeg", "combo"):
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(plan.jpeg_quality)])
            if not ok:
                raise RuntimeError("JPEG image augmentation encode failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("JPEG image augmentation decode failed")
            image = decoded
        return np.asarray(image, dtype=np.uint8)

    def _apply_prepared(self, image: np.ndarray, prepared: _PreparedPhotometricPlan, *, frame_index: int) -> np.ndarray:
        return self._apply_quality(self._apply_global(image, prepared), prepared, frame_index=int(frame_index))

    def apply(
        self,
        image: np.ndarray,
        *,
        episode_index: int,
        camera_key: str,
        frame_index: int,
        epoch: int,
        global_seed: int,
    ) -> np.ndarray:
        source = np.asarray(image, dtype=np.uint8)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"image must be HxWx3, got shape={source.shape!r}")
        if not self.config.enabled:
            return source.copy()
        prepared = self._prepared_plan(
            width=int(source.shape[1]),
            height=int(source.shape[0]),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
            global_seed=int(global_seed),
        )
        return self._apply_prepared(source, prepared, frame_index=int(frame_index))

    def apply_batch(
        self,
        images: np.ndarray,
        *,
        episode_index: int,
        camera_key: str,
        frame_indices: list[int] | np.ndarray,
        epoch: int,
        global_seed: int,
    ) -> np.ndarray:
        source = np.asarray(images, dtype=np.uint8)
        if source.ndim != 4 or source.shape[-1] != 3:
            raise ValueError(f"images must be NxHxWx3, got shape={source.shape!r}")
        indices = [int(item) for item in frame_indices]
        if len(indices) != len(source):
            raise ValueError("frame_indices length must match images")
        if not self.config.enabled:
            return source.copy()
        prepared = self._prepared_plan(
            width=int(source.shape[2]),
            height=int(source.shape[1]),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
            global_seed=int(global_seed),
        )
        out = np.empty_like(source, dtype=np.uint8)
        for index, frame_index in enumerate(indices):
            out[index] = self._apply_prepared(source[index], prepared, frame_index=frame_index)
        return out

    def metadata(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        global_seed: int,
        static_scope: str,
    ) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        plan = self._prepared_plan(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
            global_seed=int(global_seed),
        ).plan
        return {
            "type": "full_image_photometric_v1",
            "static_scope": str(static_scope),
            "effective_epoch": int(epoch),
            "exposure_gain": float(plan.exposure_gain),
            "contrast_gain": float(plan.contrast_gain),
            "gamma": float(plan.gamma),
            "white_balance_bgr": [float(item) for item in plan.white_balance_bgr],
            "saturation_gain": float(plan.saturation_gain),
            "hue_delta_deg": float(plan.hue_delta_deg),
            "illumination": {
                "kind": plan.illumination_kind,
                "strength": float(plan.illumination_strength),
            },
            "quality": {
                "kind": plan.quality_kind,
                "gaussian_noise_sigma": float(plan.gaussian_noise_sigma),
                "poisson_peak": float(plan.poisson_peak),
                "jpeg_quality": int(plan.jpeg_quality),
                "defocus_blur_sigma": float(plan.defocus_blur_sigma),
                "motion_blur_length": int(plan.motion_blur_length),
                "motion_blur_angle_deg": float(plan.motion_blur_angle_deg),
                "downsample_scale": float(plan.downsample_scale),
            },
        }
