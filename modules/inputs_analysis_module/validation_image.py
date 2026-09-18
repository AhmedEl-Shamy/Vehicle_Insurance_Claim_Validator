from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


class ValidationImage:

    VEHICLE_MODEL_PATH = "assets/ai_models/yolo11n.pt"

    def __init__(
        self,
        vehicle_model=None,
        maniqa_metric=None
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.thresholds = {
            "MANIQA": 15,
            "BLUR": 40,
            "CONTRAST": 30,
            "BRIGHTNESS_MIN": 35,
            "BRIGHTNESS_MAX": 220,
            "MIN_WIDTH": 512,
            "MIN_HEIGHT": 384,
        }

        # The object holds the models
        self.vehicle_model = vehicle_model
        self.maniqa_metric = maniqa_metric

    # =====================================================
    # SAVE MODEL
    # =====================================================
    @staticmethod
    def save_model(
        model_name: str = "yolo11n.pt"
    ):
        """
        Save/copy model if needed.

        This is not needed for normal inference because
        the model checkpoint already exists.
        """
        pass

    # =====================================================
    # LOAD MODEL
    # =====================================================
    @classmethod
    def load_model(cls):

        vehicle_model = YOLO(
            cls.VEHICLE_MODEL_PATH
        )

        maniqa_metric = None

        try:
            import pyiqa

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            maniqa_metric = pyiqa.create_metric(
                "maniqa",
                device=torch.device(device)
            )

            print(
                "✅ MANIQA loaded successfully"
            )

        except Exception as e:

            print(
                "❌ MANIQA could not be loaded:",
                e
            )

        return cls(
            vehicle_model=vehicle_model,
            maniqa_metric=maniqa_metric
        )

    # =====================================================
    # IMAGE LOADING
    # =====================================================
    @staticmethod
    def load_image(image_path):

        image_path = Path(image_path)

        try:

            return Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise ValueError(
                f"Cannot read image "
                f"{image_path.name}: {e}"
            )

    # =====================================================
    # IMAGE QUALITY
    # =====================================================
    def calculate_quality(
        self,
        image_path
    ):

        image = self.load_image(
            image_path
        )

        image_array = np.array(
            image
        )

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        height, width = (
            gray.shape[:2]
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        brightness = float(
            gray.mean()
        )

        contrast = float(
            gray.std()
        )

        return {

            "width": int(width),

            "height": int(height),

            "blur_score": round(
                blur_score,
                2
            ),

            "brightness": round(
                brightness,
                2
            ),

            "contrast": round(
                contrast,
                2
            ),

            "resolution_ok": (
                width
                >= self.thresholds[
                    "MIN_WIDTH"
                ]
                and
                height
                >= self.thresholds[
                    "MIN_HEIGHT"
                ]
            ),

            "blur_ok": (
                blur_score
                >= self.thresholds[
                    "BLUR"
                ]
            ),

            "brightness_ok": (
                self.thresholds[
                    "BRIGHTNESS_MIN"
                ]
                <= brightness
                <= self.thresholds[
                    "BRIGHTNESS_MAX"
                ]
            ),

            "contrast_ok": (
                contrast
                >= self.thresholds[
                    "CONTRAST"
                ]
            ),
        }

    # =====================================================
    # MANIQA
    # =====================================================
    def calculate_maniqa(
        self,
        image_path
    ):

        if self.maniqa_metric is None:

            print(
                "❌ MANIQA model is NOT loaded."
            )

            return None

        try:

            result = self.maniqa_metric(
                str(image_path)
            )

            score = float(
                result.item()
                * 100.0
            )

            return round(
                score,
                2
            )

        except Exception as e:

            print(
                "❌ MANIQA error:",
                e
            )

            return None

    # =====================================================
    # VEHICLE DETECTION
    # =====================================================
    def check_vehicle(
        self,
        image_path
    ):

        if self.vehicle_model is None:

            return {
                "vehicle_detected": False,
                "vehicle_count": 0,
                "vehicle_confidence": 0.0
            }

        result = self.vehicle_model.predict(
            source=str(image_path),
            conf=0.05,
            device=self.device,
            verbose=False
        )[0]

        vehicle_count = 0
        best_confidence = 0.0

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls.item()
                )

                # COCO class 2 = car
                if class_id == 2:

                    vehicle_count += 1

                    confidence = float(
                        box.conf.item()
                    )

                    best_confidence = max(
                        best_confidence,
                        confidence
                    )

        return {

            "vehicle_detected": (
                vehicle_count > 0
            ),

            "vehicle_count": (
                vehicle_count
            ),

            "vehicle_confidence": round(
                best_confidence,
                4
            )
        }

    # =====================================================
    # MAIN FUNCTION
    # =====================================================
    def extract_information(
        self,
        image_path
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            return {
                "status": "ERROR",
                "image": image_path.name,
                "message": "Image not found."
            }

        try:

            quality = (
                self.calculate_quality(
                    image_path
                )
            )

            maniqa_score = (
                self.calculate_maniqa(
                    image_path
                )
            )

            quality[
                "maniqa_score"
            ] = maniqa_score

            maniqa_ok = (
                maniqa_score is not None
                and
                maniqa_score
                >= self.thresholds[
                    "MANIQA"
                ]
            )

            quality[
                "maniqa_ok"
            ] = maniqa_ok

            vehicle_result = (
                self.check_vehicle(
                    image_path
                )
            )

            quality.update(
                vehicle_result
            )

            technical_pass = (
                quality["resolution_ok"]
                and
                quality["blur_ok"]
                and
                quality["brightness_ok"]
                and
                quality["contrast_ok"]
            )

            final_valid = (
                technical_pass
                and
                maniqa_ok
                and
                quality[
                    "vehicle_detected"
                ]
            )

            failure_reasons = []

            if not quality[
                "resolution_ok"
            ]:
                failure_reasons.append(
                    "resolution"
                )

            if not quality[
                "blur_ok"
            ]:
                failure_reasons.append(
                    "blur"
                )

            if not quality[
                "brightness_ok"
            ]:
                failure_reasons.append(
                    "brightness"
                )

            if not quality[
                "contrast_ok"
            ]:
                failure_reasons.append(
                    "contrast"
                )

            if not maniqa_ok:
                failure_reasons.append(
                    "MANIQA"
                )

            if not quality[
                "vehicle_detected"
            ]:
                failure_reasons.append(
                    "vehicle_presence"
                )

            return {

                "status": (
                    "VALID"
                    if final_valid
                    else "INVALID"
                ),

                "image": image_path.name,

                "validation": quality,

                "failure_reasons":
                    failure_reasons
            }

        except Exception as e:

            return {
                "status": "ERROR",
                "image": image_path.name,
                "message": str(e)
            }
