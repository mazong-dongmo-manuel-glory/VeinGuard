from __future__ import annotations

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
    x1 = int(width * 0.2)
    y1 = int(height * 0.12)
    x2 = int(width * 0.8)
    y2 = int(height * 0.92)
    return frame_bgr[y1:y2, x1:x2].copy()


def segment_hand(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    roi = _central_roi(frame_bgr)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi, mask, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < config.MIN_HAND_AREA:
        return roi, mask, None
    return roi, mask, contour


def _extract_geometry(contour: np.ndarray) -> dict[str, Any]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area else 0.0
    aspect_ratio = (w / h) if h else 0.0

    moments = cv2.moments(contour)
    hu_log = _log_hu(cv2.HuMoments(moments))

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
        "hull_area": round(hull_area, 4),
        "solidity": round(solidity, 6),
        "convexity_defects": int(defects_count),
        "finger_peaks": int(finger_peaks),
        "hu": [round(value, 6) for value in hu_log],
    }


def _extract_orb_signature(roi_bgr: np.ndarray, contour: np.ndarray | None) -> dict[str, Any]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask = None
    if contour is not None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    orb = cv2.ORB_create(nfeatures=128)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    if descriptors is None or len(keypoints) == 0:
        return {"vector": [0.0] * 32, "keypoints": 0}

    vector = descriptors.mean(axis=0).astype(np.float32)
    return {
        "vector": [round(float(value), 4) for value in vector.tolist()],
        "keypoints": len(keypoints),
    }


def build_multimodal_profile(frame_bgr: np.ndarray) -> dict[str, Any]:
    roi, mask, contour = segment_hand(frame_bgr)
    if contour is None:
        raise ValueError("Aucune paume exploitable detectee dans la ROI.")

    geometry = _extract_geometry(contour)
    orb = _extract_orb_signature(roi, contour)
    quality = {
        "hand_area": geometry["area"],
        "keypoints": orb["keypoints"],
        "mask_fill_ratio": round(float(np.count_nonzero(mask)) / float(mask.size), 4),
    }

    return {
        "schema_version": "2.0",
        "modalities": ["palmprint", "finger_geometry"],
        "palmprint": {
            "geometry": geometry,
            "orb_signature": orb["vector"],
            "quality": quality,
        },
        "finger_geometry": {
            "estimated_finger_gaps": geometry["convexity_defects"],
            "estimated_finger_peaks": geometry["finger_peaks"],
        },
    }


def _relative_score(value_a: float, value_b: float) -> float:
    denominator = max(abs(value_b), 1e-6)
    return abs(value_a - value_b) / denominator


def verify_multimodal(frame_bgr: np.ndarray, stored_profile: dict) -> dict[str, Any]:
    live_profile = build_multimodal_profile(frame_bgr)

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

    palm_score = 0.55 * geometry_score + 0.20 * hu_score + 0.25 * orb_score
    total_score = palm_score

    return {
        "match": total_score <= config.MATCH_THRESHOLD,
        "score": round(float(total_score), 4),
        "threshold": config.MATCH_THRESHOLD,
        "components": {
            "geometry": round(float(geometry_score), 4),
            "hu": round(float(hu_score), 4),
            "orb": round(float(orb_score), 4),
        },
        "live_profile": live_profile,
    }


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
