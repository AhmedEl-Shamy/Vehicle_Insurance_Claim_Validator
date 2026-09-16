from pathlib import Path
import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ImageDetectionSegmentation:
    """
    Combined Damage Detection + Vehicle Part Segmentation model.

    This class is designed for inference and later FastAPI integration.
    It does not train or retrain models.
    """

    def __init__(
        self,
        damage_model_path="assets/ai_models/damage_detection/best.pt",
        segmentation_model_path="assets/ai_models/segmentation/best.pt",
        device=None,
        damage_confidence=0.25,
        segmentation_confidence=0.25,
    ):

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.damage_model_path = Path(
            damage_model_path
        )

        self.segmentation_model_path = Path(
            segmentation_model_path
        )

        self.damage_confidence = (
            damage_confidence
        )

        self.segmentation_confidence = (
            segmentation_confidence
        )

        self.damage_model = None
        self.segmentation_model = None

        self.load_models()

    # =====================================================
    # LOAD MODELS
    # =====================================================
    def load_models(self):
        """
        Load the already-trained checkpoints.
        """

        if not self.damage_model_path.exists():
            raise FileNotFoundError(
                f"Damage model not found: "
                f"{self.damage_model_path}"
            )

        if not self.segmentation_model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found: "
                f"{self.segmentation_model_path}"
            )

        # Damage detection model
        self.damage_model = YOLO(
            str(self.damage_model_path)
        )

        print(
            "✅ Damage model loaded:",
            self.damage_model_path
        )

        # Vehicle part segmentation model
        self.segmentation_model = YOLO(
            str(self.segmentation_model_path)
        )

        print(
            "✅ Segmentation model loaded:",
            self.segmentation_model_path
        )

        print(
            "Device:",
            self.device
        )

    # =====================================================
    # SAVE MODELS
    # =====================================================
    def save_models(self):
        """
        Inference only.
        Models are loaded from existing checkpoints.
        """
        pass

    # =====================================================
    # DAMAGE DETECTION
    # =====================================================
    def detect_damage(self, image_path):
        """
        Run damage detection on one image.
        """

        result = self.damage_model.predict(
            source=str(image_path),
            conf=self.damage_confidence,
            device=self.device,
            verbose=False
        )[0]

        height, width = result.orig_shape

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            x1, y1, x2, y2 = [
                float(v)
                for v in box.xyxy[0].tolist()
            ]

            # Severity proxy from bbox area
            bbox_area = (
                max(0.0, x2 - x1)
                *
                max(0.0, y2 - y1)
            )

            area_ratio = (
                bbox_area
                / max(
                    1.0,
                    width * height
                )
            )

            if area_ratio <= 0.03:
                severity = "minor"

            elif area_ratio <= 0.10:
                severity = "moderate"

            else:
                severity = "severe"

            detections.append({
                "damage_type": str(
                    self.damage_model.names[class_id]
                ),

                "confidence": round(
                    confidence,
                    4
                ),

                "bbox_xyxy": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ],

                "severity": severity,

                "severity_area_ratio": round(
                    area_ratio,
                    4
                ),

                "_image_width": int(width),
                "_image_height": int(height)
            })

        return detections

    # =====================================================
    # VEHICLE PART SEGMENTATION
    # =====================================================
    def segment_parts(self, image_path):
        """
        Run vehicle-part segmentation.
        """

        result = self.segmentation_model.predict(
            source=str(image_path),
            conf=self.segmentation_confidence,
            device=self.device,
            verbose=False
        )[0]

        parts = []

        if (
            result.boxes is None
            or result.masks is None
        ):
            return parts

        for index, (
            box,
            polygon
        ) in enumerate(
            zip(
                result.boxes,
                result.masks.xy
            )
        ):

            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            part_name = str(
                self.segmentation_model.names[
                    class_id
                ]
            )

            x1, y1, x2, y2 = [
                float(v)
                for v in box.xyxy[0].tolist()
            ]

            parts.append({
                "part_instance_id": index,

                "part": part_name,

                "part_confidence": round(
                    confidence,
                    4
                ),

                "part_bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ],

                "mask_polygon": np.asarray(
                    polygon,
                    dtype=float
                ).tolist()
            })

        return parts

    # =====================================================
    # BBOX HELPERS
    # =====================================================
    @staticmethod
    def bbox_area(bbox):
        return (
            max(
                0.0,
                bbox[2] - bbox[0]
            )
            *
            max(
                0.0,
                bbox[3] - bbox[1]
            )
        )

    @staticmethod
    def bbox_intersection(a, b):
        return (
            max(
                0.0,
                min(a[2], b[2])
                -
                max(a[0], b[0])
            )
            *
            max(
                0.0,
                min(a[3], b[3])
                -
                max(a[1], b[1])
            )
        )

    @staticmethod
    def bbox_center(bbox):
        return (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        )

    @staticmethod
    def center_proximity(
        a,
        b,
        width,
        height
    ):

        diagonal = np.hypot(
            width,
            height
        )

        if diagonal <= 0:
            return 0.0

        distance = np.hypot(
            a[0] - b[0],
            a[1] - b[1]
        )

        return max(
            0.0,
            1.0 - distance / diagonal
        )

    # =====================================================
    # POLYGON / MASK
    # =====================================================
    @staticmethod
    def rasterize(
        polygon,
        height,
        width
    ):

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        if polygon is not None and len(polygon) >= 3:

            points = np.asarray(
                polygon,
                dtype=np.int32
            ).reshape(-1, 1, 2)

            cv2.fillPoly(
                mask,
                [points],
                1
            )

        return mask

    # =====================================================
    # DAMAGE -> PART MATCHING
    # =====================================================
    def match_damage_to_part(
        self,
        damage,
        parts
    ):

        height = damage["_image_height"]
        width = damage["_image_width"]

        damage_bbox = tuple(
            damage["bbox_xyxy"]
        )

        damage_area = self.bbox_area(
            damage_bbox
        )

        if damage_area <= 0 or not parts:
            return "unknown"

        x1, y1, x2, y2 = [
            int(round(v))
            for v in damage_bbox
        ]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        damage_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        damage_mask[
            y1:y2,
            x1:x2
        ] = 1

        damage_center = self.bbox_center(
            damage_bbox
        )

        candidates = []

        for part in parts:

            # object is NOT a valid affected part
            if part["part"] == "object":
                continue

            part_mask = self.rasterize(
                part["mask_polygon"],
                height,
                width
            )

            if part_mask.sum() == 0:
                continue

            intersection = float(
                np.logical_and(
                    damage_mask,
                    part_mask
                ).sum()
            )

            union = float(
                np.logical_or(
                    damage_mask,
                    part_mask
                ).sum()
            )

            in_part = (
                intersection
                / damage_area
            )

            iou = (
                intersection / union
                if union > 0
                else 0.0
            )

            proximity = (
                self.center_proximity(
                    damage_center,
                    self.bbox_center(
                        part["part_bbox"]
                    ),
                    width,
                    height
                )
            )

            containment = (
                self.bbox_intersection(
                    damage_bbox,
                    tuple(
                        part["part_bbox"]
                    )
                )
                /
                damage_area
            )

            part_confidence = float(
                part["part_confidence"]
            )

            score = (
                0.45 * in_part
                +
                0.20 * iou
                +
                0.15 * proximity
                +
                0.10 * part_confidence
                +
                0.10 * containment
            )

            candidates.append(
                (
                    score,
                    part
                )
            )

        if not candidates:
            return "unknown"

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        best_score, best_part = candidates[0]

        second_score = (
            candidates[1][0]
            if len(candidates) > 1
            else None
        )

        margin = (
            None
            if second_score is None
            else best_score - second_score
        )

        if best_score < 0.20:
            return "unknown"

        if (
            margin is not None
            and margin < 0.08
        ):
            return "unknown"

        return best_part["part"]

    # =====================================================
    # ANNOTATED IMAGE
    # =====================================================
    def create_annotated_image(
        self,
        image_path,
        detections,
        output_path
    ):
        """
        Create and save an annotated image
        with damage bounding boxes.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        for damage in detections:

            x1, y1, x2, y2 = [
                int(round(v))
                for v in damage["bbox_xyxy"]
            ]

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Label
            label = (
                f'{damage["damage_type"]} '
                f'{damage["confidence"]:.2f}'
            )

            cv2.putText(
                image,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        return output_path

    # =====================================================
    # FINAL JSON
    # =====================================================
    def build_final_json(
        self,
        image_path,
        detections
    ):

        final = []

        for damage in detections:

            final.append({
                "image": Path(
                    image_path
                ).name,

                "damage_type": damage[
                    "damage_type"
                ],

                "confidence": damage[
                    "confidence"
                ],

                "bbox_xyxy": damage[
                    "bbox_xyxy"
                ],

                "affected_part": (
                    damage.get(
                        "affected_part",
                        "unknown"
                    )
                    or "unknown"
                ),

                "severity": damage[
                    "severity"
                ],

                "severity_area_ratio":
                    damage[
                        "severity_area_ratio"
                    ]
            })

        return final

    # =====================================================
    # MAIN FUNCTION FOR FASTAPI
    # =====================================================
    def extract_information(
        self,
        image_path,
        output_dir="outputs"
    ) -> dict:
        """
        Main function that FastAPI will call.

        Returns:
            analyzed image path
            final JSON
            raw segmentation information
        """

        image_path = Path(image_path)

        if not image_path.exists():

            return {
                "status": "ERROR",
                "image": image_path.name,
                "message": "Image not found."
            }

        try:

            # -----------------------------------------
            # 1. Damage Detection
            # -----------------------------------------
            damages = self.detect_damage(
                image_path
            )

            # -----------------------------------------
            # 2. Part Segmentation
            # -----------------------------------------
            parts = self.segment_parts(
                image_path
            )

            # -----------------------------------------
            # 3. Damage -> Part Matching
            # -----------------------------------------
            for damage in damages:

                damage["affected_part"] = (
                    self.match_damage_to_part(
                        damage,
                        parts
                    )
                )

            # -----------------------------------------
            # 4. Final JSON
            # -----------------------------------------
            final_json = self.build_final_json(
                image_path,
                damages
            )

            # -----------------------------------------
            # 5. Save Annotated Image
            # -----------------------------------------
            output_dir = Path(output_dir)

            image_output = (
                output_dir
                / "images"
                / f"{image_path.stem}_annotated.jpg"
            )

            annotated_path = (
                self.create_annotated_image(
                    image_path,
                    damages,
                    image_output
                )
            )

            # -----------------------------------------
            # 6. Save JSON
            # -----------------------------------------
            json_output = (
                output_dir
                / "json"
                / f"{image_path.stem}.json"
            )

            json_output.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                json_output,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    final_json,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            return {
                "status": "SUCCESS",

                "image": image_path.name,

                "annotated_image": annotated_path,

                "json_file": str(
                    json_output
                ),

                "results": final_json
            }

        except Exception as e:

            return {
                "status": "ERROR",

                "image": image_path.name,

                "message": str(e)
            }