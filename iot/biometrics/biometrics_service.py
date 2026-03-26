from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config


_GABOR_KERNEL_CACHE: dict[tuple[int, int, float, float, float], list[np.ndarray]] = {}


def _encode_image_base64(image_bgr: np.ndarray, quality: int = 72) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _create_clahe() -> cv2.CLAHE:
    grid = max(2, int(config.NOIR_CLAHE_GRID_SIZE))
    return cv2.createCLAHE(
        clipLimit=float(config.NOIR_CLAHE_CLIP_LIMIT),
        tileGridSize=(grid, grid),
    )


def _elliptic_kernel(size: int) -> np.ndarray:
    odd_size = max(3, int(size))
    if odd_size % 2 == 0:
        odd_size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_size, odd_size))


def _log_hu(hu_moments: np.ndarray) -> list[float]:
    values: list[float] = []
    for value in hu_moments.flatten():
        if value == 0:
            values.append(0.0)
        else:
            values.append(float(-np.sign(value) * np.log10(abs(value))))
    return values


def _preprocess_frame_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    return _create_clahe().apply(gray)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _mask_candidates(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_inv = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        4,
    )
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        4,
    )
    return [otsu_inv, otsu, adaptive_inv, adaptive]


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _elliptic_kernel(7), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _elliptic_kernel(5), iterations=1)
    return mask


def _center_distance_ratio(point: tuple[float, float], shape: tuple[int, int]) -> float:
    height, width = shape[:2]
    cx = width / 2.0
    cy = height / 2.0
    max_distance = max(float(np.hypot(cx, cy)), 1.0)
    return float(np.hypot(point[0] - cx, point[1] - cy) / max_distance)


def _pick_hand_contour(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_mask: np.ndarray | None = None
    best_contour: np.ndarray | None = None
    best_score = -1.0
    frame_area = float(max(gray.shape[0] * gray.shape[1], 1))

    for candidate in _mask_candidates(gray):
        mask = _postprocess_mask(candidate)
        contour = _largest_contour(mask)
        if contour is None:
            continue

        area = float(cv2.contourArea(contour))
        if area < config.MIN_HAND_AREA:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area else 0.0
        moments = cv2.moments(contour)
        center = (
            (moments["m10"] / moments["m00"]) if moments["m00"] else gray.shape[1] / 2.0,
            (moments["m01"] / moments["m00"]) if moments["m00"] else gray.shape[0] / 2.0,
        )
        area_ratio = area / frame_area
        if area_ratio >= 0.92:
            continue

        score = area * max(solidity, 0.15) * (1.15 - min(area_ratio, 0.85))
        score *= 1.15 - min(_center_distance_ratio(center, gray.shape), 1.0)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_contour = contour

    if best_mask is None or best_contour is None:
        raise ValueError("Aucune paume exploitable detectee dans le cadre.")
    return best_mask, best_contour


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a.astype(np.float32) - b.astype(np.float32)
    cb = c.astype(np.float32) - b.astype(np.float32)
    denominator = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denominator <= 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(ab, cb) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _cluster_points(points: list[dict[str, float]], min_delta_x: float = 18.0) -> list[dict[str, float]]:
    if not points:
        return []
    points = sorted(points, key=lambda item: item["x"])
    clusters: list[list[dict[str, float]]] = [[points[0]]]
    for point in points[1:]:
        if abs(point["x"] - clusters[-1][-1]["x"]) <= min_delta_x:
            clusters[-1].append(point)
        else:
            clusters.append([point])

    merged = []
    for cluster in clusters:
        merged.append(
            {
                "x": float(np.mean([point["x"] for point in cluster])),
                "y": float(np.mean([point["y"] for point in cluster])),
                "depth": float(np.max([point["depth"] for point in cluster])),
            }
        )
    return merged


def _find_finger_valleys(contour: np.ndarray) -> list[tuple[int, int]]:
    moments = cv2.moments(contour)
    centroid_x = (moments["m10"] / moments["m00"]) if moments["m00"] else float(contour[:, 0, 0].mean())
    centroid_y = (moments["m01"] / moments["m00"]) if moments["m00"] else float(contour[:, 0, 1].mean())

    if len(contour) < 4:
        return []

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return []

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return []

    candidates: list[dict[str, float]] = []
    for defect in defects[:, 0]:
        start_idx, end_idx, far_idx, depth = defect
        if depth < 1800:
            continue

        start = contour[start_idx][0]
        end = contour[end_idx][0]
        far = contour[far_idx][0]
        if far[1] >= centroid_y:
            continue

        angle = _angle_degrees(start, far, end)
        if angle >= 105.0:
            continue

        candidates.append({"x": float(far[0]), "y": float(far[1]), "depth": float(depth)})

    merged = _cluster_points(candidates)
    if len(merged) < 2:
        return []

    left_candidates = [item for item in merged if item["x"] < centroid_x]
    right_candidates = [item for item in merged if item["x"] > centroid_x]

    if left_candidates and right_candidates:
        left = max(left_candidates, key=lambda item: item["depth"])
        right = max(right_candidates, key=lambda item: item["depth"])
        return [(int(left["x"]), int(left["y"])), (int(right["x"]), int(right["y"]))]

    merged = sorted(merged, key=lambda item: item["x"])
    return [
        (int(merged[0]["x"]), int(merged[0]["y"])),
        (int(merged[-1]["x"]), int(merged[-1]["y"])),
    ]


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.transform(points[np.newaxis, :, :], matrix)[0]


def _transform_point(matrix: np.ndarray, point: tuple[int, int]) -> tuple[float, float]:
    transformed = _transform_points(matrix, np.array([[point[0], point[1]]], dtype=np.float32))[0]
    return float(transformed[0]), float(transformed[1])


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


def _extract_geometry(
    contour: np.ndarray,
    frame_shape: tuple[int, int],
    valleys: list[tuple[int, int]],
) -> dict[str, Any]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area else 0.0
    extent = area / float(max(w * h, 1))
    aspect_ratio = float(w / h) if h else 0.0
    roi_area = float(max(frame_shape[0] * frame_shape[1], 1))
    area_ratio = area / roi_area

    moments = cv2.moments(contour)
    hu_log = _log_hu(cv2.HuMoments(moments))
    center_x = (moments["m10"] / moments["m00"]) if moments["m00"] else x + (w / 2.0)
    center_y = (moments["m01"] / moments["m00"]) if moments["m00"] else y + (h / 2.0)

    centroid_distance = _center_distance_ratio((center_x, center_y), frame_shape)

    convexity_defects = 0
    finger_peaks = 0
    if len(contour) >= 4:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is not None and len(hull_indices) >= 4:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                finger_peaks = int(defects.shape[0])
                for defect in defects[:, 0]:
                    if defect[3] > 2200:
                        convexity_defects += 1

    valley_span_ratio = 0.0
    if len(valleys) >= 2:
        valley_span = float(np.hypot(valleys[1][0] - valleys[0][0], valleys[1][1] - valleys[0][1]))
        valley_span_ratio = valley_span / max(float(w), 1.0)

    return {
        "area": round(area, 4),
        "perimeter": round(perimeter, 4),
        "aspect_ratio": round(aspect_ratio, 6),
        "extent": round(extent, 6),
        "area_ratio": round(area_ratio, 6),
        "hull_area": round(hull_area, 4),
        "solidity": round(solidity, 6),
        "center_distance_ratio": round(float(centroid_distance), 6),
        "border_touch_count": int(_border_touch_count(x, y, w, h, frame_shape)),
        "convexity_defects": int(convexity_defects),
        "finger_peaks": int(finger_peaks),
        "valley_span_ratio": round(float(valley_span_ratio), 6),
        "hu": [round(value, 6) for value in hu_log],
    }


def _normalize_palm_roi(
    gray_frame: np.ndarray,
    hand_mask: np.ndarray,
    contour: np.ndarray,
    valleys: list[tuple[int, int]],
) -> dict[str, Any]:
    x, y, w, h = cv2.boundingRect(contour)

    def build_from_crop(
        rotated_gray: np.ndarray,
        rotated_mask: np.ndarray,
        crop_x1: int,
        crop_y1: int,
        crop_x2: int,
        crop_y2: int,
        rotation_degrees: float,
        alignment_method: str,
        aligned_valleys: list[list[int]],
    ) -> dict[str, Any]:
        cropped_gray = rotated_gray[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped_gray.size == 0 or cropped_mask.size == 0:
            raise ValueError("ROI palmaire vide apres alignement.")

        size = max(64, int(config.PALM_CODE_SIZE))
        normalized_gray = cv2.resize(cropped_gray, (size, size), interpolation=cv2.INTER_LINEAR)
        normalized_mask = cv2.resize(cropped_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        normalized_mask = np.where(normalized_mask > 0, 255, 0).astype(np.uint8)
        return {
            "normalized_gray": normalized_gray,
            "normalized_mask": normalized_mask,
            "rotation_degrees": round(float(rotation_degrees), 4),
            "valleys": aligned_valleys,
            "crop_rect": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
            "rotated_gray": rotated_gray,
            "alignment_method": alignment_method,
        }

    if len(valleys) >= 2:
        left, right = sorted(valleys[:2], key=lambda item: item[0])
    else:
        left = (x + int(w * 0.30), y + int(h * 0.18))
        right = (x + int(w * 0.70), y + int(h * 0.18))

    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    center = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)

    rotation = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_gray = cv2.warpAffine(
        gray_frame,
        rotation,
        (gray_frame.shape[1], gray_frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask = cv2.warpAffine(
        hand_mask,
        rotation,
        (hand_mask.shape[1], hand_mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_contour = _transform_points(rotation, contour.reshape(-1, 2).astype(np.float32)).reshape(-1, 1, 2).astype(np.int32)
    left_r = _transform_point(rotation, left)
    right_r = _transform_point(rotation, right)
    span = float(np.hypot(right_r[0] - left_r[0], right_r[1] - left_r[1]))
    if span < 18.0:
        raise ValueError("Alignement palmaire impossible: points anatomiques instables.")

    crop_width = int(max(span * config.PALM_ROI_WIDTH_RATIO, 64))
    crop_height = int(max(span * config.PALM_ROI_HEIGHT_RATIO, 80))
    crop_center_x = (left_r[0] + right_r[0]) / 2.0
    crop_center_y = (left_r[1] + right_r[1]) / 2.0 + (span * 0.92)
    x1 = max(int(crop_center_x - (crop_width / 2.0)), 0)
    y1 = max(int(crop_center_y - (crop_height / 2.0)), 0)
    x2 = min(x1 + crop_width, rotated_gray.shape[1])
    y2 = min(y1 + crop_height, rotated_gray.shape[0])
    normalized = build_from_crop(
        rotated_gray,
        rotated_mask,
        x1,
        y1,
        x2,
        y2,
        angle,
        "valley_alignment",
        [list(left), list(right)],
    )
    mask_fill_ratio = np.count_nonzero(normalized["normalized_mask"]) / float(max(normalized["normalized_mask"].size, 1))
    if mask_fill_ratio >= 0.05:
        normalized["rotated_contour"] = rotated_contour
        return normalized

    fallback_x = max(int(x - w * 0.10), 0)
    fallback_y = max(int(y + h * 0.12), 0)
    fallback_w = min(int(w * 1.20), gray_frame.shape[1] - fallback_x)
    fallback_h = min(int(h * 0.78), gray_frame.shape[0] - fallback_y)
    fallback = build_from_crop(
        gray_frame,
        hand_mask,
        fallback_x,
        fallback_y,
        fallback_x + fallback_w,
        fallback_y + fallback_h,
        0.0,
        "bbox_fallback",
        [list(left), list(right)],
    )
    fallback["rotated_contour"] = contour
    return fallback


def _enhance_palm_lines(normalized_gray: np.ndarray, normalized_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = _create_clahe().apply(normalized_gray)
    blackhat_small = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, _elliptic_kernel(config.NOIR_BLACKHAT_SMALL))
    blackhat_large = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, _elliptic_kernel(config.NOIR_BLACKHAT_LARGE))
    enhanced = cv2.addWeighted(blackhat_small, 0.6, blackhat_large, 0.4, 0.0)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced = _create_clahe().apply(enhanced)
    enhanced = cv2.bitwise_and(enhanced, normalized_mask)
    binary_lines = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        max(3, int(config.NOIR_ADAPTIVE_BLOCK_SIZE) | 1),
        int(config.NOIR_ADAPTIVE_C),
    )
    binary_lines = cv2.bitwise_and(binary_lines, normalized_mask)
    return enhanced, binary_lines


def _gabor_kernels() -> list[np.ndarray]:
    key = (
        int(config.PALM_CODE_ORIENTATIONS),
        int(config.PALM_GABOR_KERNEL_SIZE),
        float(config.PALM_GABOR_SIGMA),
        float(config.PALM_GABOR_LAMBDA),
        float(config.PALM_GABOR_GAMMA),
    )
    kernels = _GABOR_KERNEL_CACHE.get(key)
    if kernels is not None:
        return kernels

    kernels = []
    orientation_count = max(4, int(config.PALM_CODE_ORIENTATIONS))
    kernel_size = max(7, int(config.PALM_GABOR_KERNEL_SIZE))
    if kernel_size % 2 == 0:
        kernel_size += 1
    for index in range(orientation_count):
        theta = np.pi * index / orientation_count
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            float(config.PALM_GABOR_SIGMA),
            theta,
            float(config.PALM_GABOR_LAMBDA),
            float(config.PALM_GABOR_GAMMA),
            0,
            ktype=cv2.CV_32F,
        )
        kernels.append(kernel)
    _GABOR_KERNEL_CACHE[key] = kernels
    return kernels


def _block_reduce_2d(array: np.ndarray, block_size: int) -> np.ndarray:
    block_size = max(1, int(block_size))
    height = (array.shape[0] // block_size) * block_size
    width = (array.shape[1] // block_size) * block_size
    reduced = array[:height, :width]
    return reduced.reshape(height // block_size, block_size, width // block_size, block_size).mean(axis=(1, 3))


def _block_reduce_3d(array: np.ndarray, block_size: int) -> np.ndarray:
    block_size = max(1, int(block_size))
    _, height, width = array.shape
    height = (height // block_size) * block_size
    width = (width // block_size) * block_size
    reduced = array[:, :height, :width]
    return reduced.reshape(reduced.shape[0], height // block_size, block_size, width // block_size, block_size).mean(axis=(2, 4))


def _extract_orientation_template(
    enhanced_gray: np.ndarray,
    normalized_mask: np.ndarray,
) -> dict[str, Any]:
    image_f = enhanced_gray.astype(np.float32) / 255.0
    responses = []
    for kernel in _gabor_kernels():
        responses.append(np.abs(cv2.filter2D(image_f, cv2.CV_32F, kernel)))
    response_stack = np.stack(responses, axis=0)

    block_size = max(2, int(config.PALM_CODE_BLOCK_SIZE))
    block_responses = _block_reduce_3d(response_stack, block_size)
    block_mask = _block_reduce_2d((normalized_mask > 0).astype(np.float32), block_size) >= 0.35

    best_index = np.argmax(block_responses, axis=0).astype(np.uint8)
    best_response = np.max(block_responses, axis=0)
    second_response = np.partition(block_responses, -2, axis=0)[-2]

    active_blocks = int(np.count_nonzero(block_mask))
    line_strength = float(best_response[block_mask].mean()) if active_blocks else 0.0
    orientation_confidence = (
        float(((best_response - second_response) / np.maximum(best_response, 1e-6))[block_mask].mean())
        if active_blocks
        else 0.0
    )

    histogram = np.zeros(max(4, int(config.PALM_CODE_ORIENTATIONS)), dtype=np.float32)
    if active_blocks:
        valid_codes = best_index[block_mask]
        histogram = np.bincount(valid_codes, minlength=histogram.size).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)

    return {
        "orientation_code": best_index.astype(int).tolist(),
        "orientation_mask": block_mask.astype(np.uint8).tolist(),
        "orientation_histogram": [round(float(value), 6) for value in histogram.tolist()],
        "active_blocks": active_blocks,
        "line_strength": round(line_strength, 6),
        "orientation_confidence": round(orientation_confidence, 6),
    }


def _extract_histogram(image: np.ndarray, mask: np.ndarray | None) -> list[float]:
    histogram = cv2.calcHist([image], [0], mask, [32], [0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return [round(float(value), 6) for value in histogram.tolist()]


def _quality_score(
    mask_fill_ratio: float,
    active_blocks: int,
    line_strength: float,
    orientation_confidence: float,
    sharpness: float,
) -> float:
    fill_component = min(mask_fill_ratio / 0.42, 1.0)
    block_component = min(active_blocks / 220.0, 1.0)
    line_component = min(line_strength / 0.18, 1.0)
    confidence_component = min(orientation_confidence / 0.18, 1.0)
    sharpness_component = min(sharpness / 220.0, 1.0)
    return round(
        float(
            0.18 * fill_component
            + 0.24 * block_component
            + 0.24 * line_component
            + 0.14 * confidence_component
            + 0.20 * sharpness_component
        ),
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
            "min_active_blocks": config.ENROLLMENT_MIN_PALM_ACTIVE_BLOCKS,
            "min_line_strength": config.ENROLLMENT_MIN_PALM_LINE_STRENGTH,
            "min_orientation_confidence": config.ENROLLMENT_MIN_PALM_ORIENTATION_CONFIDENCE,
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
        "min_active_blocks": config.MIN_PALM_ACTIVE_BLOCKS,
        "min_line_strength": config.MIN_PALM_LINE_STRENGTH,
        "min_orientation_confidence": config.MIN_PALM_ORIENTATION_CONFIDENCE,
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
        "active_blocks": quality["keypoints"] >= int(rules["min_active_blocks"]),
        "line_strength": quality["line_strength"] >= float(rules["min_line_strength"]),
        "orientation_confidence": quality["orientation_confidence"] >= float(rules["min_orientation_confidence"]),
        "sharpness": quality["sharpness"] >= float(rules["min_sharpness"]),
        "quality_score": quality["score"] >= float(rules["min_capture_quality"]),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if mode == "enrollment" and failed_checks:
        non_critical = {
            "mask_fill_ratio",
            "active_blocks",
            "line_strength",
            "orientation_confidence",
            "quality_score",
        }
        if set(failed_checks).issubset(non_critical):
            failed_checks = []
    reason_map = {
        "hand_area_ratio": "main trop dominante dans le cadre",
        "mask_fill_ratio": "paume absente ou hors cadrage",
        "extent": "silhouette trop diffuse pour une paume",
        "solidity": "contour de main incoherent",
        "aspect_ratio": "orientation de main non exploitable",
        "center_distance_ratio": "main trop decalee du centre",
        "border_touch_count": "main coupee par le bord",
        "active_blocks": "lignes palmaires insuffisantes",
        "line_strength": "contraste des lignes palmaires insuffisant",
        "orientation_confidence": "orientation des lignes trop ambigue",
        "sharpness": "image trop floue",
        "quality_score": "qualite palmaire insuffisante",
    }
    reason = reason_map.get(failed_checks[0], "capture palmaire invalide") if failed_checks else ""
    return {
        "valid": not failed_checks,
        "failed_checks": failed_checks,
        "reason": reason,
        "checks": checks,
        "mode": mode,
    }


def _orientation_visualization(code: list[list[int]], mask: list[list[int]]) -> np.ndarray:
    code_array = np.array(code, dtype=np.uint8)
    mask_array = np.array(mask, dtype=np.uint8) > 0
    hue_step = max(1, 180 // max(int(config.PALM_CODE_ORIENTATIONS), 1))
    hue = (code_array * hue_step).astype(np.uint8)
    saturation = np.where(mask_array, 255, 30).astype(np.uint8)
    value = np.where(mask_array, 255, 40).astype(np.uint8)
    hsv = cv2.merge((hue, saturation, value))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _render_processed_image(
    frame_bgr: np.ndarray,
    contour: np.ndarray,
    valleys: list[tuple[int, int]],
    normalized_gray: np.ndarray,
    orientation_template: dict[str, Any],
    quality: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    original = frame_bgr.copy()
    cv2.drawContours(original, [contour], -1, (0, 255, 255), 2)
    for valley in valleys:
        cv2.circle(original, valley, 8, (255, 160, 0), -1)

    orientation_map = _orientation_visualization(
        orientation_template["orientation_code"],
        orientation_template["orientation_mask"],
    )
    orientation_map = cv2.resize(
        orientation_map,
        (normalized_gray.shape[1], normalized_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    palm_bgr = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    palm_view = cv2.addWeighted(palm_bgr, 0.45, orientation_map, 0.55, 0.0)

    left = cv2.resize(original, (360, 270), interpolation=cv2.INTER_AREA)
    right = cv2.resize(palm_view, (360, 270), interpolation=cv2.INTER_LINEAR)
    composite = np.hstack((left, right))
    color = (0, 255, 0) if validation["valid"] else (0, 0, 255)
    cv2.putText(
        composite,
        f"Score {quality['score']:.2f}",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        composite,
        f"Blocs {quality['keypoints']}  Lignes {quality['line_strength']:.2f}",
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        composite,
        "ROI palmaire alignee",
        (385, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return _encode_image_base64(composite)


def _profile_texture_density(profile: dict[str, Any]) -> float:
    palm_quality = profile.get("palmprint", {}).get("quality", {})
    if palm_quality.get("texture_density") is not None:
        return float(palm_quality["texture_density"])
    if profile.get("surface_texture", {}).get("density") is not None:
        return float(profile["surface_texture"]["density"])
    if profile.get("vein_pattern", {}).get("density") is not None:
        return float(profile["vein_pattern"]["density"])
    return 0.0


def _profile_texture_fill_ratio(profile: dict[str, Any]) -> float:
    if profile.get("surface_texture", {}).get("binary_fill_ratio") is not None:
        return float(profile["surface_texture"]["binary_fill_ratio"])
    if profile.get("vein_pattern", {}).get("binary_fill_ratio") is not None:
        return float(profile["vein_pattern"]["binary_fill_ratio"])
    return 0.0


def generate_biometric_key(profile: dict[str, Any]) -> str:
    palmprint = profile.get("palmprint", {})
    geometry = palmprint.get("geometry", {})
    histogram = palmprint.get("orientation_histogram") or palmprint.get("intensity_histogram") or []
    payload = {
        "aspect_ratio": round(float(geometry.get("aspect_ratio", 0.0)), 4),
        "solidity": round(float(geometry.get("solidity", 0.0)), 4),
        "extent": round(float(geometry.get("extent", 0.0)), 4),
        "valley_span_ratio": round(float(geometry.get("valley_span_ratio", 0.0)), 4),
        "finger_peaks": int(geometry.get("finger_peaks", 0)),
        "histogram": [round(float(value), 4) for value in histogram],
        "texture_density": round(_profile_texture_density(profile), 5),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _analyze_profile(
    frame_bgr: np.ndarray,
    mode: str = "scan",
    enforce_validation: bool | None = None,
) -> dict[str, Any]:
    if enforce_validation is None:
        enforce_validation = mode != "enrollment"

    gray_frame = _preprocess_frame_gray(frame_bgr)
    hand_mask, contour = _pick_hand_contour(gray_frame)
    valleys = _find_finger_valleys(contour)
    geometry = _extract_geometry(contour, gray_frame.shape, valleys)
    normalized = _normalize_palm_roi(gray_frame, hand_mask, contour, valleys)
    enhanced_gray, binary_lines = _enhance_palm_lines(normalized["normalized_gray"], normalized["normalized_mask"])
    orientation_template = _extract_orientation_template(enhanced_gray, normalized["normalized_mask"])

    hand_pixels = max(int(np.count_nonzero(normalized["normalized_mask"])), 1)
    line_pixels = int(np.count_nonzero(binary_lines))
    texture_density = line_pixels / float(hand_pixels)
    sharpness = float(cv2.Laplacian(enhanced_gray, cv2.CV_32F).var())
    mask_fill_ratio = hand_pixels / float(max(normalized["normalized_mask"].size, 1))
    quality = {
        "hand_area": geometry["area"],
        "keypoints": orientation_template["active_blocks"],
        "mask_fill_ratio": round(mask_fill_ratio, 4),
        "texture_density": round(texture_density, 4),
        "line_strength": orientation_template["line_strength"],
        "orientation_confidence": orientation_template["orientation_confidence"],
        "sharpness": round(sharpness, 4),
        "score": _quality_score(
            mask_fill_ratio,
            orientation_template["active_blocks"],
            orientation_template["line_strength"],
            orientation_template["orientation_confidence"],
            sharpness,
        ),
    }
    validation = _capture_validation(geometry, quality, mode=mode)
    quality["validation"] = validation
    if enforce_validation and not validation["valid"]:
        raise ValueError(f"Capture palmaire invalide: {validation['reason']}")

    profile = {
        "schema_version": "4.0",
        "sensor": {
            "camera": "raspberry-pi-noir-v2",
            "preprocessing": [
                "grayscale",
                "clahe",
                "hand_contour",
                "anatomical_roi_alignment",
                "gabor_orientation_code",
            ],
        },
        "modalities": ["palmprint", "hand_geometry", "finger_geometry"],
        "palmprint": {
            "geometry": geometry,
            "intensity_histogram": _extract_histogram(enhanced_gray, normalized["normalized_mask"]),
            "orientation_histogram": orientation_template["orientation_histogram"],
            "orientation_code": orientation_template["orientation_code"],
            "orientation_mask": orientation_template["orientation_mask"],
            "quality": quality,
            "orb_signature": [],
            "descriptor_rows": [],
            "alignment": {
                "rotation_degrees": normalized["rotation_degrees"],
                "valleys": normalized["valleys"],
                "roi_size": [int(normalized["normalized_gray"].shape[1]), int(normalized["normalized_gray"].shape[0])],
                "method": normalized.get("alignment_method", "valley_alignment"),
            },
        },
        "surface_texture": {
            "density": round(texture_density, 6),
            "binary_fill_ratio": round(line_pixels / float(max(binary_lines.size, 1)), 6),
        },
        "finger_geometry": {
            "estimated_finger_gaps": geometry["convexity_defects"],
            "estimated_finger_peaks": geometry["finger_peaks"],
        },
    }
    profile["biometric_key"] = generate_biometric_key(profile)
    processed_jpeg_base64 = _render_processed_image(
        frame_bgr,
        contour,
        valleys,
        enhanced_gray,
        orientation_template,
        quality,
        validation,
    )
    return {
        "profile": profile,
        "processed_jpeg_base64": processed_jpeg_base64,
    }


def build_multimodal_profile(
    frame_bgr: np.ndarray,
    mode: str = "scan",
    enforce_validation: bool | None = None,
) -> dict[str, Any]:
    return _analyze_profile(frame_bgr, mode=mode, enforce_validation=enforce_validation)["profile"]


def analyze_hand_frame(
    frame_bgr: np.ndarray,
    mode: str = "scan",
    enforce_validation: bool | None = None,
) -> dict[str, Any]:
    return _analyze_profile(frame_bgr, mode=mode, enforce_validation=enforce_validation)


def _mean_numeric(values: list[float | int]) -> float:
    return float(np.mean(np.array(values, dtype=np.float32)))


def _merge_orientation_templates(samples: list[dict[str, Any]]) -> tuple[list[list[int]], list[list[int]], list[float]]:
    codes = np.stack(
        [np.array(sample["palmprint"]["orientation_code"], dtype=np.uint8) for sample in samples],
        axis=0,
    )
    masks = np.stack(
        [np.array(sample["palmprint"]["orientation_mask"], dtype=np.uint8) > 0 for sample in samples],
        axis=0,
    )
    orientation_count = max(4, int(config.PALM_CODE_ORIENTATIONS))
    votes = np.stack(
        [np.sum((codes == index) & masks, axis=0) for index in range(orientation_count)],
        axis=0,
    )
    fused_mask = masks.any(axis=0)
    fused_code = np.argmax(votes, axis=0).astype(np.uint8)
    fused_code[~fused_mask] = 0

    histogram = np.zeros(orientation_count, dtype=np.float32)
    if np.count_nonzero(fused_mask):
        valid_codes = fused_code[fused_mask]
        histogram = np.bincount(valid_codes, minlength=orientation_count).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)

    return (
        fused_code.astype(int).tolist(),
        fused_mask.astype(np.uint8).tolist(),
        [round(float(value), 6) for value in histogram.tolist()],
    )


def _merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    base = samples[0]
    fused_code, fused_mask, fused_histogram = _merge_orientation_templates(samples)
    geometry = {
        "area": round(_mean_numeric([sample["palmprint"]["geometry"]["area"] for sample in samples]), 4),
        "perimeter": round(_mean_numeric([sample["palmprint"]["geometry"]["perimeter"] for sample in samples]), 4),
        "aspect_ratio": round(_mean_numeric([sample["palmprint"]["geometry"]["aspect_ratio"] for sample in samples]), 6),
        "extent": round(_mean_numeric([sample["palmprint"]["geometry"]["extent"] for sample in samples]), 6),
        "area_ratio": round(_mean_numeric([sample["palmprint"]["geometry"]["area_ratio"] for sample in samples]), 6),
        "hull_area": round(_mean_numeric([sample["palmprint"]["geometry"]["hull_area"] for sample in samples]), 4),
        "solidity": round(_mean_numeric([sample["palmprint"]["geometry"]["solidity"] for sample in samples]), 6),
        "center_distance_ratio": round(_mean_numeric([sample["palmprint"]["geometry"]["center_distance_ratio"] for sample in samples]), 6),
        "border_touch_count": int(round(_mean_numeric([sample["palmprint"]["geometry"]["border_touch_count"] for sample in samples]))),
        "convexity_defects": int(round(_mean_numeric([sample["palmprint"]["geometry"]["convexity_defects"] for sample in samples]))),
        "finger_peaks": int(round(_mean_numeric([sample["palmprint"]["geometry"]["finger_peaks"] for sample in samples]))),
        "valley_span_ratio": round(_mean_numeric([sample["palmprint"]["geometry"]["valley_span_ratio"] for sample in samples]), 6),
        "hu": [
            round(_mean_numeric([sample["palmprint"]["geometry"]["hu"][index] for sample in samples]), 6)
            for index in range(len(base["palmprint"]["geometry"]["hu"]))
        ],
    }
    histogram_length = len(base["palmprint"]["intensity_histogram"])
    best_sample = max(samples, key=lambda sample: sample["palmprint"]["quality"]["score"])

    fused = {
        "schema_version": "4.0",
        "sensor": base["sensor"],
        "modalities": base["modalities"],
        "palmprint": {
            "geometry": geometry,
            "intensity_histogram": [
                round(_mean_numeric([sample["palmprint"]["intensity_histogram"][index] for sample in samples]), 6)
                for index in range(histogram_length)
            ],
            "orientation_histogram": fused_histogram,
            "orientation_code": fused_code,
            "orientation_mask": fused_mask,
            "quality": {
                "hand_area": round(_mean_numeric([sample["palmprint"]["quality"]["hand_area"] for sample in samples]), 4),
                "keypoints": int(round(_mean_numeric([sample["palmprint"]["quality"]["keypoints"] for sample in samples]))),
                "mask_fill_ratio": round(_mean_numeric([sample["palmprint"]["quality"]["mask_fill_ratio"] for sample in samples]), 4),
                "texture_density": round(_mean_numeric([sample["palmprint"]["quality"]["texture_density"] for sample in samples]), 4),
                "line_strength": round(_mean_numeric([sample["palmprint"]["quality"]["line_strength"] for sample in samples]), 6),
                "orientation_confidence": round(_mean_numeric([sample["palmprint"]["quality"]["orientation_confidence"] for sample in samples]), 6),
                "sharpness": round(_mean_numeric([sample["palmprint"]["quality"]["sharpness"] for sample in samples]), 4),
                "score": round(_mean_numeric([sample["palmprint"]["quality"]["score"] for sample in samples]), 4),
                "validation": best_sample["palmprint"]["quality"].get("validation", {}),
            },
            "orb_signature": [],
            "descriptor_rows": [],
            "alignment": best_sample["palmprint"].get("alignment", {}),
        },
        "surface_texture": {
            "density": round(_mean_numeric([_profile_texture_density(sample) for sample in samples]), 6),
            "binary_fill_ratio": round(_mean_numeric([_profile_texture_fill_ratio(sample) for sample in samples]), 6),
        },
        "finger_geometry": {
            "estimated_finger_gaps": geometry["convexity_defects"],
            "estimated_finger_peaks": geometry["finger_peaks"],
        },
    }
    fused["biometric_key"] = generate_biometric_key(fused)
    return fused


def build_enrollment_profile(frames_bgr: list[np.ndarray]) -> dict[str, Any]:
    samples = []
    rejected_samples = []
    for index, frame in enumerate(frames_bgr, start=1):
        try:
            samples.append(build_multimodal_profile(frame, mode="enrollment", enforce_validation=False))
        except Exception as exc:
            rejected_samples.append({"sample_index": index, "reason": str(exc)})

    if not samples:
        raise ValueError("Aucun echantillon palmaire exploitable pour l'enrolement.")

    fused = _merge_samples(samples) if len(samples) > 1 else dict(samples[0])
    fused["samples"] = samples
    fused["sample_count"] = len(samples)
    fused["captured_frame_count"] = len(frames_bgr)
    fused["rejected_samples"] = rejected_samples
    fused["sample_keys"] = [sample["biometric_key"] for sample in samples]
    fused["fusion_mode"] = "anatomical_roi_compcode_consensus"
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

    fused = _merge_samples(samples) if len(samples) > 1 else dict(samples[0])
    fused["samples"] = samples
    fused["sample_count"] = len(samples)
    fused["captured_frame_count"] = len(frames_bgr)
    fused["rejected_samples"] = rejected_samples
    fused["fusion_mode"] = "scan_best_orientation_template"
    fused["sample_keys"] = [sample["biometric_key"] for sample in samples]
    fused["biometric_key"] = hashlib.sha256("|".join(fused["sample_keys"]).encode("utf-8")).hexdigest()
    return fused


def _relative_score(value_a: float, value_b: float) -> float:
    denominator = max(abs(value_a), abs(value_b), 1.0)
    return min(abs(value_a - value_b) / denominator, 1.0)


def _orientation_code_score(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> float:
    live_code = np.array(live_profile["palmprint"]["orientation_code"], dtype=np.int16)
    ref_code = np.array(stored_profile["palmprint"]["orientation_code"], dtype=np.int16)
    live_mask = np.array(live_profile["palmprint"]["orientation_mask"], dtype=np.uint8) > 0
    ref_mask = np.array(stored_profile["palmprint"]["orientation_mask"], dtype=np.uint8) > 0

    overlap = live_mask & ref_mask
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count == 0:
        return 1.0

    orientation_count = max(4, int(config.PALM_CODE_ORIENTATIONS))
    delta = np.abs(live_code - ref_code)
    circular_delta = np.minimum(delta, orientation_count - delta).astype(np.float32)
    return float(np.mean(circular_delta[overlap] / max(orientation_count / 2.0, 1.0)))


def _compare_legacy_profiles(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> dict[str, Any]:
    live_geometry = live_profile["palmprint"]["geometry"]
    ref_geometry = stored_profile["palmprint"]["geometry"]
    geometry_score = float(
        np.mean(
            np.array(
                [
                    _relative_score(live_geometry["aspect_ratio"], ref_geometry.get("aspect_ratio", 0.0)),
                    _relative_score(live_geometry["solidity"], ref_geometry.get("solidity", 0.0)),
                    _relative_score(live_geometry["extent"], ref_geometry.get("extent", 0.0)),
                    _relative_score(live_geometry["valley_span_ratio"], ref_geometry.get("valley_span_ratio", 0.0)),
                ],
                dtype=np.float32,
            )
        )
    )

    live_hist = np.array(live_profile["palmprint"].get("intensity_histogram", []), dtype=np.float32)
    ref_hist = np.array(stored_profile["palmprint"].get("intensity_histogram", []), dtype=np.float32)
    histogram_score = float(np.mean(np.abs(live_hist - ref_hist))) if live_hist.size and ref_hist.size else 1.0

    live_hu = np.array(live_geometry.get("hu", []), dtype=np.float32)
    ref_hu = np.array(ref_geometry.get("hu", []), dtype=np.float32)
    if live_hu.size and ref_hu.size:
        hu_denominator = np.maximum(np.maximum(np.abs(live_hu), np.abs(ref_hu)), 1.0)
        hu_score = float(np.mean(np.abs(live_hu - ref_hu) / hu_denominator))
    else:
        hu_score = 1.0

    texture_score = _relative_score(_profile_texture_density(live_profile), _profile_texture_density(stored_profile))
    palm_score = float(0.45 * geometry_score + 0.30 * hu_score + 0.20 * histogram_score + 0.05 * texture_score)
    live_quality = live_profile["palmprint"].get("quality", {})
    live_validation = live_quality.get("validation", {})
    quality_gate_passed = live_validation.get("valid", True) and live_quality.get("score", 0.0) >= config.MIN_CAPTURE_QUALITY

    return {
        "match": bool(quality_gate_passed and palm_score <= config.MATCH_THRESHOLD),
        "score": round(palm_score, 4),
        "threshold": config.MATCH_THRESHOLD,
        "quality_gate_passed": quality_gate_passed,
        "quality_reason": live_validation.get("reason"),
        "components": {
            "orientation": None,
            "geometry": round(geometry_score, 4),
            "histogram": round(histogram_score, 4),
            "texture_density": round(texture_score, 4),
            "alignment": round(hu_score, 4),
        },
    }


def _compare_profiles(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> dict[str, Any]:
    if not stored_profile.get("palmprint", {}).get("orientation_code"):
        return _compare_legacy_profiles(live_profile, stored_profile)

    live_geometry = live_profile["palmprint"]["geometry"]
    ref_geometry = stored_profile["palmprint"]["geometry"]
    orientation_score = _orientation_code_score(live_profile, stored_profile)

    live_hist = np.array(live_profile["palmprint"]["orientation_histogram"], dtype=np.float32)
    ref_hist = np.array(stored_profile["palmprint"]["orientation_histogram"], dtype=np.float32)
    histogram_score = float(np.mean(np.abs(live_hist - ref_hist)))

    geometry_score = float(
        np.mean(
            np.array(
                [
                    _relative_score(live_geometry["aspect_ratio"], ref_geometry.get("aspect_ratio", 0.0)),
                    _relative_score(live_geometry["solidity"], ref_geometry.get("solidity", 0.0)),
                    _relative_score(live_geometry["extent"], ref_geometry.get("extent", 0.0)),
                    _relative_score(live_geometry["valley_span_ratio"], ref_geometry.get("valley_span_ratio", 0.0)),
                    _relative_score(live_geometry["finger_peaks"], ref_geometry.get("finger_peaks", 0.0)),
                ],
                dtype=np.float32,
            )
        )
    )
    texture_score = _relative_score(_profile_texture_density(live_profile), _profile_texture_density(stored_profile))
    alignment_score = _relative_score(
        live_profile["palmprint"].get("alignment", {}).get("rotation_degrees", 0.0),
        stored_profile["palmprint"].get("alignment", {}).get("rotation_degrees", 0.0),
    )

    palm_score = float(
        0.68 * orientation_score
        + 0.16 * histogram_score
        + 0.10 * geometry_score
        + 0.04 * texture_score
        + 0.02 * alignment_score
    )

    live_quality = live_profile["palmprint"].get("quality", {})
    live_validation = live_quality.get("validation", {})
    quality_gate_passed = live_validation.get("valid", True) and live_quality.get("score", 0.0) >= config.MIN_CAPTURE_QUALITY
    score_gate_passed = palm_score <= config.MATCH_THRESHOLD

    return {
        "match": bool(quality_gate_passed and score_gate_passed),
        "score": round(palm_score, 4),
        "threshold": config.MATCH_THRESHOLD,
        "quality_gate_passed": quality_gate_passed,
        "quality_reason": live_validation.get("reason"),
        "components": {
            "orientation": round(orientation_score, 4),
            "geometry": round(geometry_score, 4),
            "histogram": round(histogram_score, 4),
            "texture_density": round(texture_score, 4),
            "alignment": round(alignment_score, 4),
        },
    }


def verify_live_profile(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> dict[str, Any]:
    candidates = [stored_profile]
    candidates.extend(stored_profile.get("samples") or [])

    scored_candidates = []
    for index, candidate in enumerate(candidates):
        comparison = _compare_profiles(live_profile, candidate)
        scored_candidates.append((comparison["score"], index, comparison))

    best_score, best_index, best_result = min(scored_candidates, key=lambda item: item[0])
    return {
        **best_result,
        "score": round(float(best_score), 4),
        "matched_sample_index": best_index,
        "live_profile": live_profile,
    }


def _average_components(results: list[dict[str, Any]]) -> dict[str, float | None]:
    component_names = ("orientation", "geometry", "histogram", "texture_density", "alignment")
    averaged = {}
    for name in component_names:
        values = [
            result["components"].get(name)
            for result in results
            if result.get("components", {}).get(name) is not None
        ]
        averaged[name] = round(float(np.mean(np.array(values, dtype=np.float32))), 4) if values else None
    return averaged


def verify_multiframe(frames_bgr: list[np.ndarray], stored_profile: dict[str, Any]) -> dict[str, Any]:
    live_profile = build_identification_profile(frames_bgr)
    return verify_identification_profile(live_profile, stored_profile)


def verify_identification_profile(live_profile: dict[str, Any], stored_profile: dict[str, Any]) -> dict[str, Any]:
    sample_results = [verify_live_profile(sample, stored_profile) for sample in live_profile["samples"]]
    fused_result = verify_live_profile(live_profile, stored_profile)

    ranked_results = sorted([fused_result, *sample_results], key=lambda item: item["score"])
    best_result = ranked_results[0]
    top_results = ranked_results[: min(2, len(ranked_results))]
    top_mean = float(np.mean(np.array([result["score"] for result in top_results], dtype=np.float32)))
    combined_score = round(float(0.25 * fused_result["score"] + 0.55 * best_result["score"] + 0.20 * top_mean), 4)
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
            "strategy": "0.25*fused + 0.55*best + 0.20*top_mean",
        },
        "valid_sample_count": live_profile["sample_count"],
        "captured_frame_count": live_profile["captured_frame_count"],
        "rejected_samples": live_profile["rejected_samples"],
    }


def verify_multimodal(frame_bgr: np.ndarray, stored_profile: dict[str, Any]) -> dict[str, Any]:
    live_profile = build_multimodal_profile(frame_bgr)
    return verify_live_profile(live_profile, stored_profile)


def load_local_templates(path: str | Path = config.TEMPLATE_FILE) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_local_template(user_id: str, profile: dict[str, Any], path: str | Path = config.TEMPLATE_FILE) -> None:
    path = Path(path)
    templates = load_local_templates(path)
    templates[user_id] = profile
    path.write_text(json.dumps(templates, indent=2, ensure_ascii=False), encoding="utf-8")
