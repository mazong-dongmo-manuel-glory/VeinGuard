from __future__ import annotations

import hashlib
import json
from math import log10
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config


def _log_hu(hu_moments: np.ndarray) -> list[float]:
    values = []
    for value in hu_moments.flatten():
        if value == 0:
            values.append(0.0)
        else:
            values.append(float(-np.sign(value) * log10(abs(value))))
    return values


def _central_roi(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1 = int(width * 0.18)
    y1 = int(height * 0.10)
    x2 = int(width * 0.82)
    y2 = int(height * 0.94)
    return frame_bgr[y1:y2, x1:x2].copy()


def _create_clahe():
    grid = max(2, int(config.NOIR_CLAHE_GRID_SIZE))
    return cv2.createCLAHE(clipLimit=float(config.NOIR_CLAHE_CLIP_LIMIT), tileGridSize=(grid, grid))


def _elliptic_kernel(size: int) -> np.ndarray:
    odd_size = max(3, int(size))
    if odd_size % 2 == 0:
        odd_size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_size, odd_size))


def _preprocess_noir_gray(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return _create_clahe().apply(gray)


def _enhance_veins(gray_roi: np.ndarray, hand_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blackhat_small = cv2.morphologyEx(
        gray_roi,
        cv2.MORPH_BLACKHAT,
        _elliptic_kernel(config.NOIR_BLACKHAT_SMALL),
    )
    blackhat_large = cv2.morphologyEx(
        gray_roi,
        cv2.MORPH_BLACKHAT,
        _elliptic_kernel(config.NOIR_BLACKHAT_LARGE),
    )
    enhanced = cv2.addWeighted(blackhat_small, 0.55, blackhat_large, 0.45, 0.0)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = _create_clahe().apply(enhanced.astype(np.uint8))

    block_size = max(3, int(config.NOIR_ADAPTIVE_BLOCK_SIZE))
    if block_size % 2 == 0:
        block_size += 1
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(config.NOIR_ADAPTIVE_C),
    )
    adaptive = cv2.bitwise_and(adaptive, hand_mask)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, _elliptic_kernel(3), iterations=1)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, _elliptic_kernel(5), iterations=1)
    return enhanced, adaptive


def segment_hand(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    roi = _central_roi(frame_bgr)
    gray = _preprocess_noir_gray(roi)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = _elliptic_kernel(5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi, mask, None, gray

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < config.MIN_HAND_AREA:
        return roi, mask, None, gray
    return roi, mask, contour, gray


def _border_touch_count(x: int, y: int, w: int, h: int, shape: tuple[int, int], margin: int = 8) -> int:
    height, width = shape[:2]
    return sum(
        (
            x <= margin,
            y <= margin,
            x + w >= width - margin,
            y + h >= height - margin,
        )
    )


def _extract_geometry(contour: np.ndarray, frame_shape: tuple[int, int]) -> dict[str, Any]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area else 0.0
    aspect_ratio = (w / h) if h else 0.0
    extent = area / float(max(w * h, 1))
    roi_area = float(max(frame_shape[0] * frame_shape[1], 1))
    area_ratio = area / roi_area

    moments = cv2.moments(contour)
    hu_log = _log_hu(cv2.HuMoments(moments))
    center_x = (moments["m10"] / moments["m00"]) if moments["m00"] else x + (w / 2.0)
    center_y = (moments["m01"] / moments["m00"]) if moments["m00"] else y + (h / 2.0)
    roi_center_x = frame_shape[1] / 2.0
    roi_center_y = frame_shape[0] / 2.0
    max_center_distance = max(float(np.hypot(roi_center_x, roi_center_y)), 1.0)
    center_distance_ratio = float(
        np.hypot(center_x - roi_center_x, center_y - roi_center_y) / max_center_distance
    )

    defects_count = 0
    finger_peaks = 0
    if len(contour) >= 4:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) >= 4:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                finger_peaks = defects.shape[0]
                for defect in defects[:, 0]:
                    _, _, _, depth = defect
                    if depth > 2500:
                        defects_count += 1

    return {
        "area": round(area, 4),
        "perimeter": round(perimeter, 4),
        "aspect_ratio": round(aspect_ratio, 6),
        "extent": round(extent, 6),
        "area_ratio": round(area_ratio, 6),
        "hull_area": round(hull_area, 4),
        "solidity": round(solidity, 6),
        "center_distance_ratio": round(center_distance_ratio, 6),
        "border_touch_count": int(_border_touch_count(x, y, w, h, frame_shape)),
        "convexity_defects": int(defects_count),
        "finger_peaks": int(finger_peaks),
        "hu": [round(value, 6) for value in hu_log],
    }


def _extract_histogram(image: np.ndarray, mask: np.ndarray | None) -> list[float]:
    histogram = cv2.calcHist([image], [0], mask, [32], [0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return [round(float(value), 6) for value in histogram.tolist()]


def _extract_orb_signature(image: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=config.ORB_FEATURE_COUNT)
    keypoints, descriptors = orb.detectAndCompute(image, mask)

    if descriptors is None or len(keypoints) == 0:
        return {
            "vector": [0.0] * 32,
            "descriptor_rows": [],
            "keypoints": 0,
        }

    ranked = sorted(zip(keypoints, descriptors), key=lambda item: item[0].response, reverse=True)
    limited_descriptors = np.array(
        [descriptor for _, descriptor in ranked[: config.ORB_DESCRIPTOR_LIMIT]],
        dtype=np.uint8,
    )
    vector = limited_descriptors.mean(axis=0).astype(np.float32)
    return {
        "vector": [round(float(value), 4) for value in vector.tolist()],
        "descriptor_rows": limited_descriptors.astype(int).tolist(),
        "keypoints": len(keypoints),
    }


def _quality_score(hand_area: float, keypoints: int, vein_density: float, sharpness: float) -> float:
    area_component = min(hand_area / 25000.0, 1.0)
    kp_component = min(keypoints / 90.0, 1.0)
    density_component = min(vein_density / 0.22, 1.0)
    sharpness_component = min(sharpness / 250.0, 1.0)
    return round(
        float(0.28 * area_component + 0.27 * kp_component + 0.20 * density_component + 0.25 * sharpness_component),
        4,
    )


def _validation_profile(mode: str) -> dict[str, float | int]:
    if mode == "enrollment":
        return {
            "min_mask_fill_ratio": config.ENROLLMENT_MIN_MASK_FILL_RATIO,
            "max_mask_fill_ratio": config.MAX_MASK_FILL_RATIO,
            "max_hand_area_ratio": config.MAX_HAND_AREA_RATIO,
            "min_hand_extent": config.MIN_HAND_EXTENT,
            "min_hand_solidity": config.MIN_HAND_SOLIDITY,
            "max_hand_solidity": config.MAX_HAND_SOLIDITY,
            "min_hand_aspect_ratio": config.MIN_HAND_ASPECT_RATIO,
            "max_hand_aspect_ratio": config.MAX_HAND_ASPECT_RATIO,
            "max_hand_center_distance": config.ENROLLMENT_MAX_HAND_CENTER_DISTANCE,
            "max_border_touches": config.ENROLLMENT_MAX_BORDER_TOUCHES,
            "min_orb_keypoints": config.ENROLLMENT_MIN_ORB_KEYPOINTS,
            "min_sharpness": config.ENROLLMENT_MIN_SHARPNESS,
            "min_capture_quality": config.ENROLLMENT_MIN_CAPTURE_QUALITY,
        }
    return {
        "min_mask_fill_ratio": config.MIN_MASK_FILL_RATIO,
        "max_mask_fill_ratio": config.MAX_MASK_FILL_RATIO,
        "max_hand_area_ratio": config.MAX_HAND_AREA_RATIO,
        "min_hand_extent": config.MIN_HAND_EXTENT,
        "min_hand_solidity": config.MIN_HAND_SOLIDITY,
        "max_hand_solidity": config.MAX_HAND_SOLIDITY,
        "min_hand_aspect_ratio": config.MIN_HAND_ASPECT_RATIO,
        "max_hand_aspect_ratio": config.MAX_HAND_ASPECT_RATIO,
        "max_hand_center_distance": config.MAX_HAND_CENTER_DISTANCE,
        "max_border_touches": config.MAX_BORDER_TOUCHES,
        "min_orb_keypoints": config.MIN_ORB_KEYPOINTS,
        "min_sharpness": config.MIN_SHARPNESS,
        "min_capture_quality": config.MIN_CAPTURE_QUALITY,
    }


def _capture_validation(geometry: dict[str, Any], quality: dict[str, Any], mode: str = "scan") -> dict[str, Any]:
    rules = _validation_profile(mode)
    checks = {
        "hand_area_ratio": geometry["area_ratio"] <= float(rules["max_hand_area_ratio"]),
        "mask_fill_ratio": float(rules["min_mask_fill_ratio"]) <= quality["mask_fill_ratio"] <= float(rules["max_mask_fill_ratio"]),
        "extent": geometry["extent"] >= float(rules["min_hand_extent"]),
        "solidity": float(rules["min_hand_solidity"]) <= geometry["solidity"] <= float(rules["max_hand_solidity"]),
        "aspect_ratio": float(rules["min_hand_aspect_ratio"]) <= geometry["aspect_ratio"] <= float(rules["max_hand_aspect_ratio"]),
        "center_distance_ratio": geometry["center_distance_ratio"] <= float(rules["max_hand_center_distance"]),
        "border_touch_count": geometry["border_touch_count"] <= int(rules["max_border_touches"]),
        "keypoints": quality["keypoints"] >= int(rules["min_orb_keypoints"]),
        "sharpness": quality["sharpness"] >= float(rules["min_sharpness"]),
        "quality_score": quality["score"] >= float(rules["min_capture_quality"]),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    reason_map = {
        "hand_area_ratio": "main trop large ou decor dominant dans le cadre",
        "mask_fill_ratio": "main absente ou hors cadrage",
        "extent": "forme detectee trop diffuse pour etre une main",
        "solidity": "silhouette de main incoherente",
        "aspect_ratio": "orientation de la main non exploitable",
        "center_distance_ratio": "main trop decalee du centre",
        "border_touch_count": "main trop coupee par le bord",
        "keypoints": "texture palmaire insuffisante",
        "sharpness": "image trop floue",
        "quality_score": "qualite biométrique insuffisante",
    }
    reason = reason_map.get(failed_checks[0], "capture biométrique invalide") if failed_checks else ""
    return {
        "valid": not failed_checks,
        "failed_checks": failed_checks,
        "reason": reason,
        "checks": checks,
        "mode": mode,
    }


def build_multimodal_profile(frame_bgr: np.ndarray, mode: str = "scan") -> dict[str, Any]:
    roi, hand_mask, contour, gray_roi = segment_hand(frame_bgr)
    if contour is None:
        raise ValueError("Aucune paume exploitable detectee dans la ROI.")

    geometry = _extract_geometry(contour, gray_roi.shape)

    contour_mask = np.zeros(gray_roi.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    vein_enhanced, vein_mask = _enhance_veins(gray_roi, contour_mask)
    orb = _extract_orb_signature(vein_enhanced, contour_mask)

    hand_pixels = max(int(np.count_nonzero(contour_mask)), 1)
    vein_pixels = int(np.count_nonzero(vein_mask))
    vein_density = vein_pixels / float(hand_pixels)
    sharpness = float(cv2.Laplacian(vein_enhanced, cv2.CV_32F).var())

    quality = {
        "hand_area": geometry["area"],
        "keypoints": orb["keypoints"],
        "mask_fill_ratio": round(hand_pixels / float(contour_mask.size), 4),
        "vein_density": round(vein_density, 4),
        "sharpness": round(sharpness, 4),
        "score": _quality_score(geometry["area"], orb["keypoints"], vein_density, sharpness),
    }
    validation = _capture_validation(geometry, quality, mode=mode)
    quality["validation"] = validation
    if not validation["valid"]:
        raise ValueError(f"Capture biométrique invalide: {validation['reason']}")

    profile = {
        "schema_version": "3.0",
        "sensor": {
            "camera": "raspberry-pi-noir-v2",
            "preprocessing": [
                "grayscale",
                "clahe",
                "blackhat",
                "adaptive_threshold",
                "orb",
            ],
        },
        "modalities": ["palm_texture", "vein_pattern", "finger_geometry"],
        "palmprint": {
            "geometry": geometry,
            "intensity_histogram": _extract_histogram(vein_enhanced, contour_mask),
            "orb_signature": orb["vector"],
            "descriptor_rows": orb["descriptor_rows"],
            "quality": quality,
        },
        "vein_pattern": {
            "density": round(vein_density, 6),
            "binary_fill_ratio": round(vein_pixels / float(vein_mask.size), 6),
        },
        "finger_geometry": {
            "estimated_finger_gaps": geometry["convexity_defects"],
            "estimated_finger_peaks": geometry["finger_peaks"],
        },
    }
    profile["biometric_key"] = generate_biometric_key(profile)
    return profile


def generate_biometric_key(profile: dict[str, Any]) -> str:
    geometry = profile["palmprint"]["geometry"]
    payload = {
        "area": round(float(geometry["area"]), 2),
        "perimeter": round(float(geometry["perimeter"]), 2),
        "aspect_ratio": round(float(geometry["aspect_ratio"]), 4),
        "solidity": round(float(geometry["solidity"]), 4),
        "convexity_defects": int(geometry["convexity_defects"]),
        "finger_peaks": int(geometry["finger_peaks"]),
        "hu": [round(float(value), 4) for value in geometry["hu"]],
        "histogram": [round(float(value), 4) for value in profile["palmprint"]["intensity_histogram"]],
        "orb_signature": [round(float(value), 2) for value in profile["palmprint"]["orb_signature"]],
        "vein_density": round(float(profile["vein_pattern"]["density"]), 5),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mean_numeric(values: list[float | int]) -> float:
    return float(np.mean(np.array(values, dtype=np.float32)))


def _merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    base = samples[0]
    geometry = {
        "area": round(_mean_numeric([sample["palmprint"]["geometry"]["area"] for sample in samples]), 4),
        "perimeter": round(_mean_numeric([sample["palmprint"]["geometry"]["perimeter"] for sample in samples]), 4),
        "aspect_ratio": round(_mean_numeric([sample["palmprint"]["geometry"]["aspect_ratio"] for sample in samples]), 6),
        "hull_area": round(_mean_numeric([sample["palmprint"]["geometry"]["hull_area"] for sample in samples]), 4),
        "solidity": round(_mean_numeric([sample["palmprint"]["geometry"]["solidity"] for sample in samples]), 6),
        "convexity_defects": int(round(_mean_numeric([sample["palmprint"]["geometry"]["convexity_defects"] for sample in samples]))),
        "finger_peaks": int(round(_mean_numeric([sample["palmprint"]["geometry"]["finger_peaks"] for sample in samples]))),
        "hu": [
            round(
                _mean_numeric([sample["palmprint"]["geometry"]["hu"][index] for sample in samples]),
                6,
            )
            for index in range(len(base["palmprint"]["geometry"]["hu"]))
        ],
    }
    histogram_length = len(base["palmprint"]["intensity_histogram"])
    orb_length = len(base["palmprint"]["orb_signature"])
    best_sample = max(samples, key=lambda sample: sample["palmprint"]["quality"]["score"])

    fused = {
        "schema_version": "3.0",
        "sensor": base["sensor"],
        "modalities": base["modalities"],
        "palmprint": {
            "geometry": geometry,
            "intensity_histogram": [
                round(_mean_numeric([sample["palmprint"]["intensity_histogram"][index] for sample in samples]), 6)
                for index in range(histogram_length)
            ],
            "orb_signature": [
                round(_mean_numeric([sample["palmprint"]["orb_signature"][index] for sample in samples]), 4)
                for index in range(orb_length)
            ],
            "descriptor_rows": best_sample["palmprint"]["descriptor_rows"],
            "quality": {
                "hand_area": round(_mean_numeric([sample["palmprint"]["quality"]["hand_area"] for sample in samples]), 4),
                "keypoints": int(round(_mean_numeric([sample["palmprint"]["quality"]["keypoints"] for sample in samples]))),
                "mask_fill_ratio": round(_mean_numeric([sample["palmprint"]["quality"]["mask_fill_ratio"] for sample in samples]), 4),
                "vein_density": round(_mean_numeric([sample["palmprint"]["quality"]["vein_density"] for sample in samples]), 4),
                "sharpness": round(_mean_numeric([sample["palmprint"]["quality"]["sharpness"] for sample in samples]), 4),
                "score": round(_mean_numeric([sample["palmprint"]["quality"]["score"] for sample in samples]), 4),
            },
        },
        "vein_pattern": {
            "density": round(_mean_numeric([sample["vein_pattern"]["density"] for sample in samples]), 6),
            "binary_fill_ratio": round(_mean_numeric([sample["vein_pattern"]["binary_fill_ratio"] for sample in samples]), 6),
        },
        "finger_geometry": {
            "estimated_finger_gaps": geometry["convexity_defects"],
            "estimated_finger_peaks": geometry["finger_peaks"],
        },
        "sample_quality_ranking": [
            {
                "sample_index": index,
                "score": sample["palmprint"]["quality"]["score"],
            }
            for index, sample in sorted(
                enumerate(samples),
                key=lambda item: item[1]["palmprint"]["quality"]["score"],
                reverse=True,
            )
        ],
    }
    fused["biometric_key"] = generate_biometric_key(fused)
    return fused


def build_enrollment_profile(frames_bgr: list[np.ndarray]) -> dict[str, Any]:
    samples = []
    rejected_samples = []
    for index, frame in enumerate(frames_bgr, start=1):
        try:
            samples.append(build_multimodal_profile(frame, mode="enrollment"))
        except Exception as exc:
            rejected_samples.append({"sample_index": index, "reason": str(exc)})
    if not samples:
        raise ValueError("Aucun echantillon biométrique capturé.")

    fused = _merge_samples(samples)
    fused["samples"] = samples
    fused["sample_count"] = len(samples)
    fused["captured_frame_count"] = len(frames_bgr)
    fused["rejected_samples"] = rejected_samples
    fused["sample_keys"] = [sample["biometric_key"] for sample in samples]
    fused["fusion_mode"] = "multisample_average_best_descriptor"
    fused["biometric_key"] = hashlib.sha256("|".join(fused["sample_keys"]).encode("utf-8")).hexdigest()
    return fused


def build_identification_profile(frames_bgr: list[np.ndarray]) -> dict[str, Any]:
    samples = []
    rejected_samples = []
    for index, frame in enumerate(frames_bgr, start=1):
        try:
            samples.append(build_multimodal_profile(frame, mode="scan"))
        except Exception as exc:
            rejected_samples.append({"sample_index": index, "reason": str(exc)})

    if not samples:
        raise ValueError("Aucune capture exploitable pour l'identification.")

    if len(samples) == 1:
        fused = dict(samples[0])
    else:
        fused = _merge_samples(samples)

    fused["samples"] = samples
    fused["sample_count"] = len(samples)
    fused["captured_frame_count"] = len(frames_bgr)
    fused["rejected_samples"] = rejected_samples
    fused["fusion_mode"] = "multiframe_weighted_scan"
    fused["sample_keys"] = [sample["biometric_key"] for sample in samples]
    fused["biometric_key"] = hashlib.sha256("|".join(fused["sample_keys"]).encode("utf-8")).hexdigest()
    return fused


def _relative_score(value_a: float, value_b: float) -> float:
    denominator = max(abs(value_b), 1e-6)
    return abs(value_a - value_b) / denominator


def _descriptor_match_score(live_rows: list[list[int]], ref_rows: list[list[int]]) -> float:
    if not live_rows or not ref_rows:
        return 1.0

    live = np.array(live_rows, dtype=np.uint8)
    ref = np.array(ref_rows, dtype=np.uint8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(live, ref)
    if not matches:
        return 1.0

    good_matches = [match for match in matches if match.distance <= 48]
    denominator = max(min(len(live_rows), len(ref_rows)), 1)
    return 1.0 - min(len(good_matches) / float(denominator), 1.0)


def _compare_profiles(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> dict[str, Any]:
    live_geometry = live_profile["palmprint"]["geometry"]
    ref_geometry = stored_profile["palmprint"]["geometry"]

    numeric_fields = [
        "area",
        "perimeter",
        "aspect_ratio",
        "hull_area",
        "solidity",
        "convexity_defects",
        "finger_peaks",
    ]
    geometry_score = float(np.mean([_relative_score(live_geometry[field], ref_geometry[field]) for field in numeric_fields]))

    live_hu = np.array(live_geometry["hu"], dtype=np.float32)
    ref_hu = np.array(ref_geometry["hu"], dtype=np.float32)
    hu_score = float(np.mean(np.abs(live_hu - ref_hu) / np.maximum(np.abs(ref_hu), 1e-6)))

    live_orb = np.array(live_profile["palmprint"]["orb_signature"], dtype=np.float32)
    ref_orb = np.array(stored_profile["palmprint"]["orb_signature"], dtype=np.float32)
    orb_score = float(np.mean(np.abs(live_orb - ref_orb) / 255.0))
    component_weights = {
        "geometry": 0.28,
        "hu": 0.18,
        "orb": 0.18,
        "histogram": 0.16,
        "descriptor": 0.14,
        "vein_density": 0.06,
    }
    available_components = {
        "geometry": geometry_score,
        "hu": hu_score,
        "orb": orb_score,
    }

    live_hist = live_profile["palmprint"].get("intensity_histogram")
    ref_hist = stored_profile["palmprint"].get("intensity_histogram")
    histogram_score = None
    if live_hist and ref_hist:
        live_hist_arr = np.array(live_hist, dtype=np.float32)
        ref_hist_arr = np.array(ref_hist, dtype=np.float32)
        histogram_score = float(np.mean(np.abs(live_hist_arr - ref_hist_arr)))
        available_components["histogram"] = histogram_score

    live_rows = live_profile["palmprint"].get("descriptor_rows", [])
    ref_rows = stored_profile["palmprint"].get("descriptor_rows", [])
    descriptor_score = None
    if live_rows and ref_rows:
        descriptor_score = _descriptor_match_score(live_rows, ref_rows)
        available_components["descriptor"] = descriptor_score

    live_density = live_profile.get("vein_pattern", {}).get("density")
    ref_density = stored_profile.get("vein_pattern", {}).get("density")
    density_score = None
    if live_density is not None and ref_density is not None:
        density_score = _relative_score(live_density, ref_density)
        available_components["vein_density"] = density_score

    active_weight = sum(component_weights[name] for name in available_components)
    palm_score = sum(
        component_weights[name] * score
        for name, score in available_components.items()
    ) / max(active_weight, 1e-6)
    live_quality = live_profile["palmprint"].get("quality", {})
    live_validation = live_quality.get("validation", {})
    quality_gate_passed = live_validation.get("valid", True) and live_quality.get("score", 0.0) >= config.MIN_CAPTURE_QUALITY
    score_gate_passed = palm_score <= config.MATCH_THRESHOLD

    return {
        "match": bool(quality_gate_passed and score_gate_passed),
        "score": round(float(palm_score), 4),
        "threshold": config.MATCH_THRESHOLD,
        "quality_gate_passed": quality_gate_passed,
        "quality_reason": live_validation.get("reason"),
        "components": {
            "geometry": round(float(geometry_score), 4),
            "hu": round(float(hu_score), 4),
            "orb": round(float(orb_score), 4),
            "histogram": round(float(histogram_score), 4) if histogram_score is not None else None,
            "descriptor": round(float(descriptor_score), 4) if descriptor_score is not None else None,
            "vein_density": round(float(density_score), 4) if density_score is not None else None,
        },
    }


def verify_live_profile(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> dict[str, Any]:
    candidates = [stored_profile]
    candidates.extend(stored_profile.get("samples") or [])

    scored_candidates = []
    for index, candidate in enumerate(candidates):
        comparison = _compare_profiles(live_profile, candidate)
        scored_candidates.append((comparison["score"], index, candidate, comparison))

    best_score, best_index, _, best_result = min(scored_candidates, key=lambda item: item[0])
    return {
        **best_result,
        "score": round(float(best_score), 4),
        "matched_sample_index": best_index,
        "live_profile": live_profile,
    }


def _average_components(results: list[dict[str, Any]]) -> dict[str, float | None]:
    component_names = ("geometry", "hu", "orb", "histogram", "descriptor", "vein_density")
    averaged = {}
    for name in component_names:
        values = [result["components"].get(name) for result in results if result.get("components", {}).get(name) is not None]
        averaged[name] = round(float(np.mean(np.array(values, dtype=np.float32))), 4) if values else None
    return averaged


def verify_multiframe(frames_bgr: list[np.ndarray], stored_profile: dict[str, Any]) -> dict[str, Any]:
    live_profile = build_identification_profile(frames_bgr)
    sample_results = [verify_live_profile(sample, stored_profile) for sample in live_profile["samples"]]
    fused_result = verify_live_profile(live_profile, stored_profile)

    ranked_results = sorted([fused_result, *sample_results], key=lambda item: item["score"])
    best_result = ranked_results[0]
    top_results = ranked_results[: min(2, len(ranked_results))]
    top_mean = float(np.mean(np.array([result["score"] for result in top_results], dtype=np.float32)))
    combined_score = round(float(0.45 * fused_result["score"] + 0.35 * best_result["score"] + 0.20 * top_mean), 4)
    quality_gate_passed = any(result.get("quality_gate_passed") for result in ranked_results)
    score_gate_passed = combined_score <= config.MATCH_THRESHOLD

    return {
        "match": bool(quality_gate_passed and score_gate_passed),
        "score": combined_score,
        "threshold": config.MATCH_THRESHOLD,
        "quality_gate_passed": quality_gate_passed,
        "quality_reason": best_result.get("quality_reason"),
        "components": _average_components([fused_result, *top_results]),
        "live_profile": live_profile,
        "matched_sample_index": best_result.get("matched_sample_index"),
        "sample_results": [
            {
                "score": result["score"],
                "quality_gate_passed": result.get("quality_gate_passed"),
                "matched_sample_index": result.get("matched_sample_index"),
            }
            for result in sample_results
        ],
        "fusion": {
            "fused_score": fused_result["score"],
            "best_score": best_result["score"],
            "top_mean_score": round(top_mean, 4),
            "strategy": "0.45*fused + 0.35*best + 0.20*top_mean",
        },
        "valid_sample_count": live_profile["sample_count"],
        "captured_frame_count": live_profile["captured_frame_count"],
        "rejected_samples": live_profile["rejected_samples"],
    }


def verify_multimodal(frame_bgr: np.ndarray, stored_profile: dict) -> dict[str, Any]:
    live_profile = build_multimodal_profile(frame_bgr)
    return verify_live_profile(live_profile, stored_profile)


def load_local_templates(path: str | Path = config.TEMPLATE_FILE) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_local_template(user_id: str, profile: dict, path: str | Path = config.TEMPLATE_FILE) -> None:
    path = Path(path)
    templates = load_local_templates(path)
    templates[user_id] = profile
    path.write_text(json.dumps(templates, indent=2, ensure_ascii=False), encoding="utf-8")
