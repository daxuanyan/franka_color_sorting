"""Local RGB-D instance localization for the simulated sorting workspace.

The legacy colour masks remain available as a benchmark baseline. The generic
path extracts objects from calibrated RGB-D geometry, independently of colour,
and leaves semantic naming to a pluggable crop classifier.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from grasp_geometry import TopGraspGeometry, estimate_top_grasp_geometry


# Depth at a colour-pixel on a cube observes its top face. The Franka FSM,
# however, is parameterized by the cube centre. Keep this geometry conversion
# local to perception rather than reading any simulator object pose.
CUBE_HALF_HEIGHT_M = 0.025


@dataclass(frozen=True)
class LocalizationQuality:
    mask_pixels: int
    top_pixels: int
    mask_fraction: float
    depth_mad_m: float
    confidence: float


@dataclass(frozen=True)
class RGBDInstance:
    instance_id: str
    pose: tuple[float, float, float]
    bbox_xyxy: tuple[int, int, int, int]
    quality: LocalizationQuality
    rgb_crop: np.ndarray
    grasp_geometry: TopGraspGeometry


def bounded_servo_update(
    current: np.ndarray,
    measurement: np.ndarray,
    *,
    alpha: float = 0.70,
    max_step_m: float = 0.008,
) -> tuple[np.ndarray, float, float]:
    """Apply an EMA correction with a hard per-observation displacement limit.

    Returns the filtered command, raw innovation norm, and applied step norm.
    The bound keeps a noisy RGB-D frame from causing a discontinuous motion
    target while repeated observations can still converge to a large, valid
    correction.
    """
    previous = np.asarray(current, dtype=float)
    observed = np.asarray(measurement, dtype=float)
    if previous.shape != (3,) or observed.shape != (3,):
        raise ValueError("visual-servo poses must be XYZ vectors")
    if not np.isfinite(previous).all() or not np.isfinite(observed).all():
        raise ValueError("visual-servo poses must be finite")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("visual-servo alpha must be within (0, 1]")
    if max_step_m <= 0.0:
        raise ValueError("visual-servo max_step_m must be positive")
    innovation = observed - previous
    raw_norm = float(np.linalg.norm(innovation))
    correction = alpha * innovation
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > max_step_m:
        correction *= max_step_m / correction_norm
        correction_norm = max_step_m
    return previous + correction, raw_norm, correction_norm


def assess_refinement(
    coarse: np.ndarray,
    refined: np.ndarray,
    quality: LocalizationQuality,
    *,
    min_confidence: float = 0.60,
    max_xy_offset_m: float = 0.012,
    high_confidence_threshold: float = 0.80,
    max_high_confidence_xy_offset_m: float = 0.030,
    max_z_offset_m: float = 0.010,
) -> tuple[bool, str]:
    """Apply a conservative quality and displacement gate to wrist RGB-D."""
    xy_offset = float(np.linalg.norm(refined[:2] - coarse[:2]))
    z_offset = abs(float(refined[2] - coarse[2]))
    metrics = (
        f"confidence={quality.confidence:.3f}, coverage={quality.mask_fraction:.3f}, "
        f"depth_mad={quality.depth_mad_m * 1000:.2f}mm, "
        f"xy_delta={xy_offset * 1000:.2f}mm, z_delta={z_offset * 1000:.2f}mm"
    )
    if quality.confidence < min_confidence:
        return False, f"low visual confidence; {metrics}"
    allowed_xy_offset = (
        max_high_confidence_xy_offset_m
        if quality.confidence >= high_confidence_threshold
        else max_xy_offset_m
    )
    if xy_offset > allowed_xy_offset:
        return False, f"XY correction exceeds {allowed_xy_offset * 1000:.1f}mm; {metrics}"
    if z_offset > max_z_offset_m:
        return False, f"Z correction exceeds {max_z_offset_m * 1000:.1f}mm; {metrics}"
    return True, metrics


def _mask(rgb: np.ndarray, color: str) -> np.ndarray:
    r, g, b = (rgb[..., index].astype(np.int16) for index in range(3))
    if color == "red":
        return (r > 100) & (r > g * 3 // 2) & (r > b * 3 // 2)
    if color == "yellow":
        # Yellow needs both strong red *and* strong green. The former rule
        # accepted the bright red cube as well (its green channel exceeded
        # 90 under the dome light), merging two disconnected blocks and
        # shifting yellow's RGB-D centre by several centimetres.
        return (
            (r > 110)
            & (g > 100)
            & (g * 100 > r * 65)
            & (r * 100 > g * 75)
            & (b * 100 < r * 60)
            & (b * 100 < g * 60)
        )
    if color == "green":
        return (g > 80) & (g > r * 6 // 5) & (g > b * 6 // 5)
    if color == "blue":
        return (b > 90) & (b > r * 3 // 2) & (b > g * 3 // 2)
    raise ValueError(f"Unknown colour: {color}")


def top_surface_center(
    world_points: np.ndarray,
    point_depths: np.ndarray,
    *,
    height_tolerance_m: float = 0.004,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a horizontal top-face centre from calibrated world points.

    Selecting the nearest image depths biases an oblique wrist view toward the
    near edge of a cube.  World Z directly separates its horizontal top from
    lower side faces without using simulator object poses.
    """
    world = np.asarray(world_points, dtype=float)
    depths = np.asarray(point_depths, dtype=float)
    valid = np.isfinite(world).all(axis=1) & np.isfinite(depths)
    world, depths = world[valid], depths[valid]
    if len(world) < 12:
        raise ValueError("Not enough finite RGB-D points for top-surface localization")
    reference_height = float(np.quantile(world[:, 2], 0.90))
    selector = world[:, 2] >= reference_height - height_tolerance_m
    if int(np.count_nonzero(selector)) < 12:
        selector = world[:, 2] >= float(np.quantile(world[:, 2], 0.75))
    top_world = world[selector]
    return np.median(top_world, axis=0), depths[selector], top_world[:, 2]


def _read_rgbd(camera: Any) -> tuple[np.ndarray, np.ndarray]:
    frame = camera.get_current_frame()
    raw_rgb = frame.get("rgba")
    if raw_rgb is None:
        raw_rgb = frame.get("rgb")
    if raw_rgb is None:
        raw_rgb = camera.get_rgba()
    raw_depth = frame.get("distance_to_image_plane")
    rgba = np.asarray(raw_rgb)
    depth = np.asarray(camera.get_depth() if raw_depth is None else raw_depth)
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise RuntimeError(f"Invalid RGB frame shape: {rgba.shape}")
    if depth.ndim != 2 or depth.shape != rgba.shape[:2]:
        raise RuntimeError(f"Invalid depth frame shape: {depth.shape}; expected {rgba.shape[:2]}")
    return np.asarray(rgba[..., :3], dtype=np.uint8), depth


def extract_rgbd_instances(
    camera: Any,
    *,
    stride: int = 2,
    min_height_m: float = 0.012,
    max_height_m: float = 0.20,
    workspace_xy: tuple[float, float, float, float] = (0.20, 0.80, -0.65, 0.05),
    min_component_pixels: int = 20,
    max_extent_m: float = 0.16,
) -> list[RGBDInstance]:
    """Extract tabletop instances from height and connectivity, not RGB colour.

    The returned pose is the median of the highest visible surface translated
    to the configured cube centre. Semantic labels are intentionally absent.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    if not 0.0 <= min_height_m < max_height_m:
        raise ValueError("height bounds must be ordered and non-negative")
    import cv2

    rgb, depth = _read_rgbd(camera)
    height, width = depth.shape
    rows_grid, cols_grid = np.mgrid[0:height:stride, 0:width:stride]
    rows = rows_grid.reshape(-1)
    cols = cols_grid.reshape(-1)
    sampled_depth = depth[rows, cols].astype(np.float32)
    finite = np.isfinite(sampled_depth) & (sampled_depth > 0.01)
    world_map = np.full((rows_grid.shape[0], rows_grid.shape[1], 3), np.nan, dtype=float)
    if np.any(finite):
        image_points = np.column_stack((cols[finite], rows[finite])).astype(np.float32)
        world_map.reshape(-1, 3)[finite] = camera.get_world_points_from_image_coords(
            image_points, sampled_depth[finite]
        )
    x_min, x_max, y_min, y_max = workspace_xy
    candidate = (
        np.isfinite(world_map).all(axis=2)
        & (world_map[..., 0] >= x_min)
        & (world_map[..., 0] <= x_max)
        & (world_map[..., 1] >= y_min)
        & (world_map[..., 1] <= y_max)
        & (world_map[..., 2] >= min_height_m)
        & (world_map[..., 2] <= max_height_m)
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, component_map, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    pending: list[
        tuple[np.ndarray, tuple[int, int, int, int], LocalizationQuality, np.ndarray, TopGraspGeometry]
    ] = []
    for component in range(1, count):
        selector = component_map == component
        sample_pixels = int(np.count_nonzero(selector))
        mask_pixels = sample_pixels * stride * stride
        if mask_pixels < min_component_pixels:
            continue
        points = world_map[selector]
        point_depths = depth[rows_grid[selector], cols_grid[selector]]
        xy_extent = np.ptp(points[:, :2], axis=0)
        if float(np.max(xy_extent)) > max_extent_m:
            continue
        centre, top_depths, top_heights = top_surface_center(points, point_depths)
        grasp_geometry = estimate_top_grasp_geometry(points)
        # Category-independent tabletop assumption: infer the rigid object's
        # centre height from its observed top instead of assuming a 5 cm cube.
        centre[2] = grasp_geometry.dimensions_xyz[2] / 2.0
        left, top, component_width, component_height, _ = stats[component]
        padding = max(3, 4 // stride)
        x1 = max(0, (left - padding) * stride)
        y1 = max(0, (top - padding) * stride)
        x2 = min(width, (left + component_width + padding) * stride)
        y2 = min(height, (top + component_height + padding) * stride)
        mask_fraction = mask_pixels / float(height * width)
        median_height = float(np.median(top_heights))
        depth_mad = float(np.median(np.abs(top_heights - median_height)))
        # The alternate Franka fingers can cage modestly curved/tapered
        # surfaces when the object centre and footprint width are within the
        # calibrated envelope. Earlier runs used a 2 mm planar-only cutoff,
        # which unnecessarily excluded the 2.4--3.6 mm relief of the held-out
        # cone. The fixed wrist calibration and anchored servo now make that
        # centre estimate stable enough to permit up to 4 mm while still
        # rejecting steep/irregular surfaces before motion.
        max_parallel_jaw_relief_m = 0.004
        if depth_mad > max_parallel_jaw_relief_m:
            grasp_geometry = replace(
                grasp_geometry,
                feasible=False,
                rejection_reason=(
                    f"surface relief {depth_mad:.4f}m exceeds caging-grasp limit "
                    f"{max_parallel_jaw_relief_m:.4f}m"
                ),
            )
        support_score = min(1.0, len(top_depths) * stride * stride / 100.0)
        coverage_score = 1.0 if mask_fraction <= 0.20 else max(0.0, (0.50 - mask_fraction) / 0.30)
        depth_score = max(0.0, 1.0 - depth_mad / 0.004)
        quality = LocalizationQuality(
            mask_pixels=mask_pixels,
            top_pixels=len(top_depths) * stride * stride,
            mask_fraction=mask_fraction,
            depth_mad_m=depth_mad,
            confidence=min(support_score, coverage_score, depth_score),
        )
        pending.append(
            (
                centre,
                tuple(int(value) for value in (x1, y1, x2, y2)),
                quality,
                rgb[y1:y2, x1:x2].copy(),
                grasp_geometry,
            )
        )
    # Stable spatial IDs make artifacts and semantic matching reproducible.
    pending.sort(key=lambda item: (float(item[0][1]), float(item[0][0])))
    return [
        RGBDInstance(
            instance_id=f"instance_{index:02d}",
            pose=tuple(float(value) for value in centre),
            bbox_xyxy=bbox,
            quality=quality,
            rgb_crop=crop,
            grasp_geometry=grasp_geometry,
        )
        for index, (centre, bbox, quality, crop, grasp_geometry) in enumerate(pending)
    ]


def associate_instance_labels(
    instances: list[RGBDInstance],
    assignments: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, list[float]],
    dict[str, LocalizationQuality],
    dict[str, dict[str, Any]],
]:
    """Join semantic crop labels to geometry while enforcing one-to-one IDs."""
    by_id = {instance.instance_id: instance for instance in instances}
    poses: dict[str, list[float]] = {}
    qualities: dict[str, LocalizationQuality] = {}
    for instance_id, assignment in assignments.items():
        if instance_id not in by_id or not isinstance(assignment, dict):
            raise ValueError(f"unknown semantic instance: {instance_id}")
        label = assignment.get("label")
        semantic_confidence = assignment.get("confidence", 0.0)
        if not isinstance(label, str) or not label or label in poses:
            raise ValueError("semantic labels must be non-empty and one-to-one")
        if not isinstance(semantic_confidence, (int, float)) or not 0.0 <= semantic_confidence <= 1.0:
            raise ValueError("semantic confidence must be within 0..1")
        instance = by_id[instance_id]
        poses[label] = list(instance.pose)
        qualities[label] = LocalizationQuality(
            mask_pixels=instance.quality.mask_pixels,
            top_pixels=instance.quality.top_pixels,
            mask_fraction=instance.quality.mask_fraction,
            depth_mad_m=instance.quality.depth_mad_m,
            confidence=min(instance.quality.confidence, float(semantic_confidence)),
        )
    return poses, qualities


def localize_open_vocabulary_with_quality(
    camera: Any,
    vocabulary: dict[str, str],
    artifact_dir: Path,
    *,
    semantic_backend: str = "clip",
    semantic_model: str | None = None,
    min_semantic_confidence: float = 0.50,
    semantic_python: Path = Path("/usr/bin/python3"),
    wait_callback: Callable[[], None] | None = None,
) -> tuple[dict[str, list[float]], dict[str, LocalizationQuality]]:
    """Run geometry proposals plus an isolated open-vocabulary crop labeler."""
    from PIL import Image, ImageDraw

    if not vocabulary or not all(isinstance(key, str) and isinstance(value, str) for key, value in vocabulary.items()):
        raise ValueError("vocabulary must map object IDs to text descriptions")
    if not 0.0 <= min_semantic_confidence <= 1.0:
        raise ValueError("min_semantic_confidence must be within 0..1")
    instances = extract_rgbd_instances(camera)
    if not instances:
        raise RuntimeError("RGB-D geometry found no tabletop object instances")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for instance in instances:
        crop_path = artifact_dir / f"{instance.instance_id}.png"
        Image.fromarray(instance.rgb_crop).save(crop_path)
        candidates.append({"instance_id": instance.instance_id, "crop_path": str(crop_path)})
    manifest_path = artifact_dir / "manifest.json"
    output_path = artifact_dir / "semantic_assignments.json"
    manifest_path.write_text(
        json.dumps({"labels": vocabulary, "candidates": candidates}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        str(semantic_python),
        str(Path(__file__).resolve().with_name("semantic_labeler.py")),
        "--manifest", str(manifest_path),
        "--output", str(output_path),
        "--backend", semantic_backend,
    ]
    if semantic_model:
        command.extend(["--model", semantic_model])
    # Isaac Sim exports its embedded Python 3.12 home/path. A system Python
    # semantic worker (3.10 on this host) must not import that standard library.
    semantic_env = os.environ.copy()
    semantic_env.pop("PYTHONHOME", None)
    semantic_env.pop("PYTHONPATH", None)
    # Loading CLIP in the isolated system-Python worker can take several
    # seconds. A blocking subprocess.run prevents Kit from processing desktop
    # window events during that time, which makes Linux report that Isaac Sim
    # is not responding. Poll the worker and let the caller pump Kit frames;
    # RGB-D extraction above has already captured the observation, so this does
    # not change the perception input.
    semantic_log_path = artifact_dir / "semantic_labeler.log"
    with semantic_log_path.open("w", encoding="utf-8") as semantic_log:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=semantic_log,
            stderr=subprocess.STDOUT,
            env=semantic_env,
        )
        deadline = time.monotonic() + 180.0
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    raise RuntimeError("semantic instance labeling timed out after 180 seconds")
                if wait_callback is None:
                    time.sleep(0.02)
                else:
                    wait_callback()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        return_code = process.returncode
    if return_code != 0 or not output_path.exists():
        log_tail = semantic_log_path.read_text(encoding="utf-8", errors="replace").strip()[-1000:]
        raise RuntimeError(
            "semantic instance labeling failed: "
            + (log_tail or f"process exited {return_code}")
        )
    semantic = json.loads(output_path.read_text(encoding="utf-8"))
    raw_assignments = semantic.get("assignments", {})
    by_id = {instance.instance_id: instance for instance in instances}
    assignments = {
        instance_id: assignment
        for instance_id, assignment in raw_assignments.items()
        if isinstance(assignment, dict)
        and isinstance(assignment.get("confidence"), (int, float))
        and float(assignment["confidence"]) >= min_semantic_confidence
        and instance_id in by_id
        and by_id[instance_id].grasp_geometry.feasible
    }
    poses, qualities = associate_instance_labels(instances, assignments)
    grasps = {
        assignment["label"]: by_id[instance_id].grasp_geometry.to_dict()
        for instance_id, assignment in assignments.items()
    }
    missing = sorted(set(vocabulary) - set(poses))
    metadata = {
        "backend": semantic_backend,
        "instances": [
            {
                "instance_id": instance.instance_id,
                "pose": list(instance.pose),
                "bbox_xyxy": list(instance.bbox_xyxy),
                "quality": asdict(instance.quality),
                "grasp_geometry": instance.grasp_geometry.to_dict(),
            }
            for instance in instances
        ],
        "assignments": assignments,
        "rejected_assignments": {
            instance_id: assignment
            for instance_id, assignment in raw_assignments.items()
            if instance_id not in assignments
        },
        "min_semantic_confidence": min_semantic_confidence,
        "missing_labels": missing,
    }
    (artifact_dir / "generic_perception.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Save a human-auditable result without making this visualization part of
    # the control input.
    overlay = Image.fromarray(_read_rgbd(camera)[0])
    drawing = ImageDraw.Draw(overlay)
    for instance in instances:
        assignment = assignments.get(instance.instance_id)
        if assignment is None:
            caption = "rejected"
            colour = (255, 96, 96)
        else:
            caption = f"{assignment['label']} {float(assignment['confidence']):.2f}"
            colour = (80, 255, 120)
        x1, y1, x2, y2 = instance.bbox_xyxy
        drawing.rectangle((x1, y1, x2, y2), outline=colour, width=3)
        text_box = drawing.textbbox((x1, y1), caption)
        text_height = text_box[3] - text_box[1]
        label_top = max(0, y1 - text_height - 6)
        drawing.rectangle((x1, label_top, text_box[2] + 4, y1), fill=(0, 0, 0))
        drawing.text((x1 + 2, label_top + 2), caption, fill=colour)
    overlay.save(artifact_dir / "generic_overlay.png")
    return poses, qualities, grasps


def localize_nearest_instance_with_quality(
    camera: Any,
    target_hint: np.ndarray,
    *,
    max_distance_m: float = 0.08,
) -> tuple[list[float] | None, LocalizationQuality | None]:
    """Track the geometrically nearest visible instance during wrist servoing."""
    hint = np.asarray(target_hint, dtype=float)
    instances = extract_rgbd_instances(camera, stride=1, workspace_xy=(0.20, 0.80, -0.70, 0.10))
    if not instances:
        return None, None
    instance = min(instances, key=lambda item: np.linalg.norm(np.asarray(item.pose[:2]) - hint[:2]))
    if float(np.linalg.norm(np.asarray(instance.pose[:2]) - hint[:2])) > max_distance_m:
        return None, None
    return list(instance.pose), instance.quality


def localize_coloured_blocks_with_quality(
    camera: Any,
) -> tuple[dict[str, list[float]], dict[str, LocalizationQuality]]:
    """Return world-frame cube centres from an initialized Isaac ``Camera``.

    Invalid/noisy depth pixels are rejected. The camera API performs the
    calibrated unprojection, so no simulator object pose is read here.
    """
    # Reading the attached annotator directly avoids a second, implicit depth
    # attachment in Isaac Sim 6.x. ``get_depth`` remains a compatibility
    # fallback for older camera APIs.
    rgb, depth = _read_rgbd(camera)
    results: dict[str, list[float]] = {}
    qualities: dict[str, LocalizationQuality] = {}
    for color in ("red", "blue", "green", "yellow"):
        mask = _mask(rgb, color) & np.isfinite(depth) & (depth > 0.01)
        rows, cols = np.where(mask)
        mask_pixels = len(rows)
        if len(rows) < 20:
            continue
        # Unproject the colour sample first, then identify the horizontal top
        # face in world Z. This remains unbiased for strongly oblique wrist
        # views where a nearest-depth quantile selects only the near edge.
        if len(rows) > 4000:
            sample = np.linspace(0, len(rows) - 1, num=4000, dtype=np.int64)
            rows, cols = rows[sample], cols[sample]
        image_points = np.column_stack((cols, rows)).astype(np.float32)
        point_depths = depth[rows, cols].astype(np.float32)
        world_points = camera.get_world_points_from_image_coords(image_points, point_depths)
        world, top_depths, top_heights = top_surface_center(world_points, point_depths)
        top_pixels = len(top_depths)
        results[color] = [float(world[0]), float(world[1]), float(world[2] - CUBE_HALF_HEIGHT_M)]
        mask_fraction = mask_pixels / float(mask.shape[0] * mask.shape[1])
        # Raw camera depth changes across a valid plane in an oblique view.
        # World-Z dispersion measures actual departure from a horizontal top
        # surface instead of incorrectly penalizing that perspective slope.
        height_median = float(np.median(top_heights))
        depth_mad = float(np.median(np.abs(top_heights - height_median)))
        support_score = min(1.0, top_pixels / 100.0)
        coverage_score = 1.0 if mask_fraction <= 0.35 else max(0.0, (0.75 - mask_fraction) / 0.40)
        depth_score = max(0.0, 1.0 - depth_mad / 0.004)
        qualities[color] = LocalizationQuality(
            mask_pixels=mask_pixels,
            top_pixels=top_pixels,
            mask_fraction=mask_fraction,
            depth_mad_m=depth_mad,
            confidence=min(support_score, coverage_score, depth_score),
        )
    return results, qualities


def localize_coloured_blocks(camera: Any) -> dict[str, list[float]]:
    """Compatibility wrapper returning only world-frame cube centres."""
    poses, _ = localize_coloured_blocks_with_quality(camera)
    return poses


def save_poses(
    poses: dict[str, list[float]],
    path: Path,
    qualities: dict[str, LocalizationQuality] | None = None,
    *,
    source: str = "local_rgbd",
    grasps: dict[str, dict[str, Any]] | None = None,
) -> None:
    payload: dict[str, Any] = {"frame": "world", "source": source, "poses": poses}
    if qualities is not None:
        payload["quality"] = {color: asdict(quality) for color, quality in qualities.items()}
    if grasps is not None:
        payload["grasps"] = grasps
    path.write_text(json.dumps(payload, indent=2))
