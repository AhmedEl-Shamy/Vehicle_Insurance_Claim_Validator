from pathlib import Path
import json
import hashlib
import tempfile
import urllib.request
from urllib.parse import urlparse, unquote
from typing import List, Literal

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field, TypeAdapter
from ultralytics import YOLO


# ========================================
# PYDANTIC OUTPUT SCHEMA
# ========================================

class ImageDamageResult(BaseModel):
    """Validated schema for one detected damage instance."""

    image: str = Field(..., description="Input image filename")
    damage_type: str = Field(..., min_length=1, description="Detected type of vehicle damage")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Damage detection confidence")
    bbox_xyxy: List[float] = Field(
        ..., min_length=4, max_length=4,
        description="Damage bounding box [x1, y1, x2, y2]"
    )
    affected_part: str = Field(
        ..., min_length=1,
        description="Vehicle part affected by the detected damage"
    )
    severity: Literal["minor", "moderate", "severe"] = Field(
        ..., description="Damage severity based on bounding-box area ratio"
    )
    severity_area_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Ratio of damage bounding-box area to image area"
    )


class ImageDetectionSegmentation:
    """
    Combined damage detection and vehicle-part segmentation model.
    Designed for inference and later FastAPI integration.
    It does not train or retrain models.
    """

    DAMAGE_MODEL_PATH = "assets/ai_models/damage_detection/best.pt"
    SEGMENTATION_MODEL_PATH = "assets/ai_models/segmentation/best.pt"

    def __init__(
        self,
        damage_model=None,
        segmentation_model=None,
        device=None,
        damage_confidence=0.25,
        segmentation_confidence=0.25,
        nms_iou=0.7,
    ):
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.damage_model = damage_model
        self.segmentation_model = segmentation_model
        self.damage_confidence = damage_confidence
        self.segmentation_confidence = segmentation_confidence
        # YOLO NMS IoU threshold. Lower values suppress overlapping boxes more aggressively.
        self.nms_iou = nms_iou

    # ========================================
    # SAVE MODEL
    # ========================================

    @staticmethod
    def save_model(model, output_path):
        """Save an already loaded Ultralytics YOLO model."""
        if model is None:
            raise ValueError("Model is None. Nothing to save.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            model.save(str(output_path))
        except AttributeError as exc:
            raise RuntimeError(
                "The provided model does not support the save() method."
            ) from exc

        return output_path

    # ========================================
    # LOAD MODELS
    # ========================================

    @classmethod
    def load_model(
        cls,
        damage_model_path=None,
        segmentation_model_path=None,
        device=None,
        damage_confidence=0.25,
        segmentation_confidence=0.25,
        nms_iou=0.7,
    ):
        """Load the trained damage-detection and vehicle-part-segmentation models."""
        damage_model_path = Path(
            damage_model_path if damage_model_path is not None else cls.DAMAGE_MODEL_PATH
        )
        segmentation_model_path = Path(
            segmentation_model_path
            if segmentation_model_path is not None
            else cls.SEGMENTATION_MODEL_PATH
        )

        if not damage_model_path.exists():
            raise FileNotFoundError(f"Damage model not found: {damage_model_path}")
        if not segmentation_model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found: {segmentation_model_path}"
            )

        damage_model = YOLO(str(damage_model_path))
        print("Damage model loaded:", damage_model_path)

        segmentation_model = YOLO(str(segmentation_model_path))
        print("Segmentation model loaded:", segmentation_model_path)

        selected_device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Device:", selected_device)

        return cls(
            damage_model=damage_model,
            segmentation_model=segmentation_model,
            device=selected_device,
            damage_confidence=damage_confidence,
            segmentation_confidence=segmentation_confidence,
            nms_iou=nms_iou,
        )

    # ========================================
    # DAMAGE DETECTION
    # ========================================

    def detect_damage(self, image_path):
        """Run damage detection on one image."""
        if self.damage_model is None:
            raise RuntimeError(
                "Damage model is not loaded. Call load_model() first."
            )

        result = self.damage_model.predict(
            source=str(image_path),
            conf=self.damage_confidence,
            iou=self.nms_iou,
            device=self.device,
            verbose=False,
        )[0]

        height, width = result.orig_shape
        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = bbox_area / max(1.0, width * height)

            if area_ratio <= 0.03:
                severity = "minor"
            elif area_ratio <= 0.10:
                severity = "moderate"
            else:
                severity = "severe"

            detections.append({
                "damage_type": str(self.damage_model.names[class_id]),
                "confidence": round(confidence, 4),
                "bbox_xyxy": [
                    round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)
                ],
                "severity": severity,
                "severity_area_ratio": round(area_ratio, 4),
                "_image_width": int(width),
                "_image_height": int(height),
            })

        return detections

    # ========================================
    # VEHICLE PART SEGMENTATION
    # ========================================

    def segment_parts(self, image_path):
        """Run vehicle-part segmentation."""
        if self.segmentation_model is None:
            raise RuntimeError(
                "Segmentation model is not loaded. Call load_model() first."
            )

        result = self.segmentation_model.predict(
            source=str(image_path),
            conf=self.segmentation_confidence,
            device=self.device,
            verbose=False,
        )[0]

        parts = []
        if result.boxes is None or result.masks is None:
            return parts

        for index, (box, polygon) in enumerate(zip(result.boxes, result.masks.xy)):
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            part_name = str(self.segmentation_model.names[class_id])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            parts.append({
                "part_instance_id": index,
                "part": part_name,
                "part_confidence": round(confidence, 4),
                "part_bbox": [
                    round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)
                ],
                "mask_polygon": np.asarray(polygon, dtype=float).tolist(),
            })

        return parts

    # ========================================
    # BBOX HELPERS
    # ========================================

    @staticmethod
    def bbox_area(bbox):
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def bbox_intersection(a, b):
        return (
            max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        )

    @staticmethod
    def bbox_center(bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def center_proximity(a, b, width, height):
        diagonal = np.hypot(width, height)
        if diagonal <= 0:
            return 0.0
        distance = np.hypot(a[0] - b[0], a[1] - b[1])
        return max(0.0, 1.0 - distance / diagonal)

    # ========================================
    # POLYGON / MASK
    # ========================================

    @staticmethod
    def rasterize(polygon, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        if polygon is not None and len(polygon) >= 3:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [points], 1)
        return mask

    # ========================================
    # DAMAGE -> PART MATCHING
    # ========================================

    def match_damage_to_part(self, damage, parts):
        height = damage["_image_height"]
        width = damage["_image_width"]
        damage_bbox = tuple(damage["bbox_xyxy"])
        damage_area = self.bbox_area(damage_bbox)

        if damage_area <= 0 or not parts:
            return "unknown"

        x1, y1, x2, y2 = [int(round(v)) for v in damage_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        damage_mask = np.zeros((height, width), dtype=np.uint8)
        damage_mask[y1:y2, x1:x2] = 1
        damage_center = self.bbox_center(damage_bbox)
        candidates = []

        for part in parts:
            if part["part"] == "object":
                continue

            part_mask = self.rasterize(part["mask_polygon"], height, width)
            if part_mask.sum() == 0:
                continue

            intersection = float(np.logical_and(damage_mask, part_mask).sum())
            union = float(np.logical_or(damage_mask, part_mask).sum())
            in_part = intersection / damage_area
            iou = intersection / union if union > 0 else 0.0
            proximity = self.center_proximity(
                damage_center, self.bbox_center(part["part_bbox"]), width, height
            )
            containment = (
                self.bbox_intersection(damage_bbox, tuple(part["part_bbox"]))
                / damage_area
            )
            part_confidence = float(part["part_confidence"])

            score = (
                0.45 * in_part
                + 0.20 * iou
                + 0.15 * proximity
                + 0.10 * part_confidence
                + 0.10 * containment
            )
            candidates.append((score, part))

        if not candidates:
            return "unknown"

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_part = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else None
        margin = None if second_score is None else best_score - second_score

        if best_score < 0.20:
            return "unknown"
        if margin is not None and margin < 0.08:
            return "unknown"

        return best_part["part"]

    # ========================================
    # ANNOTATED IMAGE
    # ========================================

    def create_annotated_image(self, image_path, detections, output_path):
        """Create and save an annotated image with damage bounding boxes."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        for damage in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in damage["bbox_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f'{damage["damage_type"]} {damage["confidence"]:.2f}'
            cv2.putText(
                image, label, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path

    # ========================================
    # FINAL JSON
    # ========================================

    def build_final_json(self, image_path, detections):
        final = []
        for damage in detections:
            final.append({
                "image": Path(image_path).name,
                "damage_type": damage["damage_type"],
                "confidence": damage["confidence"],
                "bbox_xyxy": damage["bbox_xyxy"],
                "affected_part": damage.get("affected_part", "unknown") or "unknown",
                "severity": damage["severity"],
                "severity_area_ratio": damage["severity_area_ratio"],
            })

        validated_results = TypeAdapter(List[ImageDamageResult]).validate_python(final)
        return [item.model_dump() for item in validated_results]

    # ========================================
    # JSON VALIDATION
    # ========================================

    @staticmethod
    def validate_json_file(json_path):
        """
        Validate a saved JSON file:
        existence, non-empty content, no semicolons, valid JSON,
        and a top-level list.
        """
        json_path = Path(json_path)

        if not json_path.exists():
            return {"valid": False, "message": f"JSON file not found: {json_path}"}
        if not json_path.is_file():
            return {"valid": False, "message": f"JSON path is not a file: {json_path}"}

        try:
            raw_content = json_path.read_text(encoding="utf-8")
        except Exception as exc:
            return {"valid": False, "message": f"Could not read JSON file: {exc}"}

        if not raw_content.strip():
            return {"valid": False, "message": "JSON file is empty."}
        if ";" in raw_content:
            return {
                "valid": False,
                "message": "Invalid JSON: semicolon ';' is not allowed.",
            }

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            return {
                "valid": False,
                "message": (
                    f"Invalid JSON syntax: {exc.msg} "
                    f"(line {exc.lineno}, column {exc.colno})"
                ),
            }

        if not isinstance(data, list):
            return {
                "valid": False,
                "message": "Invalid JSON structure: top-level JSON must be a list [].",
            }

        return {"valid": True, "message": "JSON file is valid.", "data": data}

    # ========================================
    # URL HELPERS
    # ========================================

    @staticmethod
    def _is_http_url(value) -> bool:
        parsed = urlparse(str(value))
        return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)

    @staticmethod
    def _download_url_to_temp(image_url):
        """Download one HTTP(S) image to a temporary local file for OpenCV/YOLO."""
        parsed = urlparse(str(image_url))
        original_name = Path(unquote(parsed.path)).name or "image"
        suffix = Path(original_name).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
            suffix = ".jpg"

        digest = hashlib.sha256(str(image_url).encode("utf-8")).hexdigest()[:12]
        temp_dir = tempfile.TemporaryDirectory(prefix="claim_image_")
        local_path = (
            Path(temp_dir.name)
            / f"{digest}_{Path(original_name).stem[:40]}{suffix}"
        )

        try:
            request = urllib.request.Request(
                str(image_url),
                headers={"User-Agent": "VehicleInsuranceClaimValidator/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.lower().startswith("image/"):
                    raise ValueError(
                        f"URL did not return an image (Content-Type: {content_type})"
                    )
                content = response.read()

            if not content:
                raise ValueError("Downloaded image is empty.")

            local_path.write_bytes(content)
            return temp_dir, local_path, original_name
        except Exception:
            temp_dir.cleanup()
            raise

    # ========================================
    # MAIN FUNCTIONS FOR FASTAPI
    # ========================================

    def extract_information(self, image_path, output_dir="outputs") -> dict:
        """Process one local image or HTTP(S) image URL."""
        if self._is_http_url(image_path):
            temp_dir, local_path, original_name = self._download_url_to_temp(image_path)
            try:
                result = self._extract_information_local(
                    local_path, output_dir=output_dir, display_name=original_name
                )
                result["image"] = original_name
                for detection in result.get("results", []):
                    detection["image"] = original_name
                return result
            finally:
                temp_dir.cleanup()

        return self._extract_information_local(image_path, output_dir=output_dir)

    def extract_information_batch(self, images, output_dir="outputs") -> dict:
        """Process multiple images; return per-image details and a combined detection list."""
        if images is None:
            image_list = []
        elif isinstance(images, (str, Path)):
            image_list = [images]
        else:
            image_list = [item for item in images if item is not None]

        per_image_results = []
        combined_results = []

        for image in image_list:
            try:
                result = self.extract_information(image, output_dir=output_dir)
            except Exception as exc:
                result = {"status": "ERROR", "image": str(image), "message": str(exc)}

            per_image_results.append(result)

            if result.get("status") == "SUCCESS":
                for detection in result.get("results", []):
                    combined_results.append(dict(detection))

        successful = sum(
            result.get("status") == "SUCCESS" for result in per_image_results
        )

        return {
            "status": (
                "SUCCESS"
                if successful == len(per_image_results) and successful > 0
                else "PARTIAL_SUCCESS" if successful > 0 else "ERROR"
            ),
            "total_images": len(per_image_results),
            "successful_images": successful,
            "failed_images": len(per_image_results) - successful,
            "per_image_results": per_image_results,
            "results": combined_results,
        }

    def _extract_information_local(
        self, image_path, output_dir="outputs", display_name=None
    ) -> dict:
        """Process one local image, save annotated image and per-image JSON."""
        image_path = Path(image_path)

        if not image_path.exists():
            return {
                "status": "ERROR",
                "image": display_name or image_path.name,
                "message": "Image not found.",
            }

        try:
            damages = self.detect_damage(image_path)
            parts = self.segment_parts(image_path)

            for damage in damages:
                damage["affected_part"] = self.match_damage_to_part(damage, parts)

            final_json = self.build_final_json(image_path, damages)
            if display_name:
                for item in final_json:
                    item["image"] = display_name

            output_dir = Path(output_dir)

            image_output = (
                output_dir / "images" / f"{image_path.stem}_annotated.jpg"
            )
            annotated_path = self.create_annotated_image(
                image_path, damages, image_output
            )

            json_output = output_dir / "json" / f"{image_path.stem}.json"
            json_output.parent.mkdir(parents=True, exist_ok=True)

            with open(json_output, "w", encoding="utf-8") as file:
                json.dump(final_json, file, indent=2, ensure_ascii=False)

            json_validation = self.validate_json_file(json_output)
            if not json_validation["valid"]:
                return {
                    "status": "ERROR",
                    "image": display_name or image_path.name,
                    "annotated_image": str(annotated_path),
                    "json_file": str(json_output),
                    "message": "JSON validation failed.",
                    "json_validation": json_validation,
                }

            return {
                "status": "SUCCESS",
                "image": display_name or image_path.name,
                "annotated_image": str(annotated_path),
                "json_file": str(json_output),
                "json_validation": {
                    "valid": True,
                    "message": "JSON file is valid.",
                },
                "results": final_json,
            }

        except Exception as exc:
            return {
                "status": "ERROR",
                "image": display_name or image_path.name,
                "message": str(exc),
            }
