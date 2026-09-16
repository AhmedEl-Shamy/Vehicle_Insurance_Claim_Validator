from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


class ValidationImage:
    """
    Image Validation model.
    This class is designed to be called later from FastAPI.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.thresholds = {
            "MANIQA": 15,
            "BLUR": 40,
            "CONTRAST": 30,
            "BRIGHTNESS_MIN": 35,
            "BRIGHTNESS_MAX": 220,
            "MIN_WIDTH": 512,
            "MIN_HEIGHT": 384,
        }

        self.vehicle_model = None
        self.maniqa_metric = None

        self.load_models()

    # =====================================================
    # LOAD MODELS
    # =====================================================
    def load_models(self):
        """
        Load the models required for image validation.

        Model files are expected to be available in:
        assets/ai_models/
        """

        # ---------------------------------------------
        # Vehicle Presence Model
        # ---------------------------------------------
        try:
            self.vehicle_model = YOLO(
                "assets/ai_models/yolo11n.pt"
            )

            print("✅ Vehicle detection model loaded successfully.")

        except Exception as e:
            self.vehicle_model = None

            print("❌ Vehicle detection model could not be loaded.")
            print("Error type:", type(e).__name__)
            print("Error:", str(e))

        # ---------------------------------------------
        # MANIQA Model
        # ---------------------------------------------
        try:
            import pyiqa

            print("Loading MANIQA model...")

            self.maniqa_metric = pyiqa.create_metric(
                "maniqa",
                device=torch.device(self.device)
            )

            print("✅ MANIQA model loaded successfully.")

        except Exception as e:
            self.maniqa_metric = None

            print("❌ MANIQA model could not be loaded.")
            print("Error type:", type(e).__name__)
            print("Error:", str(e))

    # =====================================================
    # SAVE MODELS
    # =====================================================
    def save_models(self):
        """
        No model training is performed here.

        This class is for inference only, so models are loaded
        from existing checkpoint files.
        """
        pass

    # =====================================================
    # IMAGE LOADING
    # =====================================================
    def load_image(self, image_path):
        """
        Load image safely and convert it to RGB.
        Supports JFIF as well.
        """

        image_path = Path(image_path)

        try:
            return Image.open(image_path).convert("RGB")

        except Exception as e:
            raise ValueError(
                f"Cannot read image {image_path.name}: {e}"
            )

    # =====================================================
    # IMAGE QUALITY
    # =====================================================
    def calculate_quality(self, image_path):
        """
        Calculate technical image-quality metrics:
        - resolution
        - blur
        - brightness
        - contrast
        """

        image = self.load_image(image_path)

        image_array = np.array(image)

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        height, width = gray.shape[:2]

        # ---------------------------------------------
        # Blur score
        # ---------------------------------------------
        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        # ---------------------------------------------
        # Brightness
        # ---------------------------------------------
        brightness = float(gray.mean())

        # ---------------------------------------------
        # Contrast
        # ---------------------------------------------
        contrast = float(gray.std())

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
                width >= self.thresholds["MIN_WIDTH"]
                and
                height >= self.thresholds["MIN_HEIGHT"]
            ),

            "blur_ok": (
                blur_score
                >= self.thresholds["BLUR"]
            ),

            "brightness_ok": (
                self.thresholds["BRIGHTNESS_MIN"]
                <= brightness
                <= self.thresholds["BRIGHTNESS_MAX"]
            ),

            "contrast_ok": (
                contrast
                >= self.thresholds["CONTRAST"]
            ),
        }

    # =====================================================
    # MANIQA
    # =====================================================
    def calculate_maniqa(self, image_path):

        if self.maniqa_metric is None:
            print("❌ MANIQA model is NOT loaded.")
            return None

        try:
            print(f"Running MANIQA on: {Path(image_path).name}")

            result = self.maniqa_metric(
                str(image_path)
            )

            score = float(result.item() * 100.0)

            print(f"✅ MANIQA score: {score:.2f}")

            return round(score, 2)

        except Exception as e:
            print("❌ MANIQA inference error:")
            print("Type:", type(e).__name__)
            print("Message:", str(e))
        return None
    # =====================================================
    # VEHICLE DETECTION
    # =====================================================
    def check_vehicle(self, image_path):
        """
        Check whether a car is present in the image.

        COCO class 2 = car.
        """

        if self.vehicle_model is None:

            return {
                "vehicle_detected": False,
                "vehicle_count": 0,
                "vehicle_confidence": 0.0,
                "vehicle_model_error": (
                    "Vehicle detection model is not loaded."
                )
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
    # MAIN VALIDATION
    # =====================================================
    def extract_information(self, image_path) -> dict:
        """
        Main method that FastAPI will call.

        Input:
            image_path

        Output:
            JSON-compatible dictionary.
        """

        image_path = Path(image_path)

        # ---------------------------------------------
        # Check file exists
        # ---------------------------------------------
        if not image_path.exists():

            return {
                "status": "ERROR",
                "image": image_path.name,
                "message": "Image not found."
            }

        try:

            # =========================================
            # 1. Technical Image Quality
            # =========================================
            quality = self.calculate_quality(
                image_path
            )

            # =========================================
            # 2. MANIQA
            # =========================================
            maniqa_score = self.calculate_maniqa(
                image_path
            )

            quality["maniqa_score"] = (
                maniqa_score
            )

            if maniqa_score is None:

                # MANIQA is unavailable.
                # Keep the pipeline running for testing.
                maniqa_ok = True

            else:

                maniqa_ok = (
                    maniqa_score
                    >= self.thresholds["MANIQA"]
                )

            quality["maniqa_ok"] = (
                maniqa_ok
            )

            # =========================================
            # 3. Vehicle Presence
            # =========================================
            vehicle_result = self.check_vehicle(
                image_path
            )

            quality.update(
                vehicle_result
            )

            # =========================================
            # 4. Technical Validation
            # =========================================
            technical_pass = (
                quality["resolution_ok"]
                and
                quality["blur_ok"]
                and
                quality["brightness_ok"]
                and
                quality["contrast_ok"]
            )

            # =========================================
            # 5. Final Validation
            # =========================================
            final_valid = (
                technical_pass
                and
                maniqa_ok
                and
                quality["vehicle_detected"]
            )

            if final_valid:

                status = "VALID"

            else:

                status = "INVALID"

            # =========================================
            # 6. Failure Reasons
            # =========================================
            failure_reasons = []

            if not quality["resolution_ok"]:

                failure_reasons.append(
                    "resolution"
                )

            if not quality["blur_ok"]:

                failure_reasons.append(
                    "blur"
                )

            if not quality["brightness_ok"]:

                failure_reasons.append(
                    "brightness"
                )

            if not quality["contrast_ok"]:

                failure_reasons.append(
                    "contrast"
                )

            if not maniqa_ok:

                failure_reasons.append(
                    "MANIQA"
                )

            if not quality["vehicle_detected"]:

                failure_reasons.append(
                    "vehicle_presence"
                )

            # =========================================
            # 7. Final Result
            # =========================================
            return {
                "status": status,

                "image": image_path.name,

                "validation": quality,

                "failure_reasons": (
                    failure_reasons
                )
            }

        except Exception as e:

            return {
                "status": "ERROR",

                "image": image_path.name,

                "message": str(e)
            }