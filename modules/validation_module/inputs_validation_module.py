"""Validation orchestration for claim inputs, including multiple images."""
from pathlib import Path
import json

from modules.validation_module.ai_image_detector import AIImageDetector
from modules.validation_module.validation_image import ValidationImage
from modules.validation_module.valid_invoice_police import ValidInvoicePolice


class InputsValidationModule:
    def __init__(self, ai_image_detector, validation_image_model, validation_invoice_police):
        self.ai_image_detector = ai_image_detector
        self.validation_image_model = validation_image_model
        self.validation_invoice_police = validation_invoice_police

    @classmethod
    def load_module(cls):
        # Fix: instantiate and pass the same, correctly named variable.
        ai_image_detector = AIImageDetector()
        validation_image_model = ValidationImage.load_model()
        validation_invoice_police = ValidInvoicePolice.load_model()
        return cls(
            ai_image_detector=ai_image_detector,
            validation_image_model=validation_image_model,
            validation_invoice_police=validation_invoice_police,
        )

    @staticmethod
    def _normalize_images(images):
        """Accept either one path/URL or an iterable of paths/URLs."""
        if images is None:
            return []
        if isinstance(images, (str, Path)):
            return [str(images)]
        try:
            return [str(image) for image in images if image is not None]
        except TypeError:
            return [str(images)]

    @staticmethod
    def _to_dict(result):
        """Support model methods that return a dict or a JSON string."""
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        raise TypeError(
            "Expected extract_information() to return a dict or JSON object string."
        )

    @staticmethod
    def _status(result):
        return str(result.get("status", "ERROR")).strip().lower()

    @staticmethod
    def _vehicle_detected(result):
        if "vehicle_detected" in result:
            return bool(result["vehicle_detected"])
        if "vehicle" in result:
            return bool(result["vehicle"])
        validation = result.get("validation", {})
        return bool(validation.get("vehicle_detected", False)) if isinstance(validation, dict) else False

    @staticmethod
    def _vehicle_confidence(result):
        keys = ("vehicle_confidence", "vehicle_detected_confidence", "max_confidence")
        for key in keys:
            value = result.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        validation = result.get("validation", {})
        if isinstance(validation, dict):
            for key in keys:
                value = validation.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0

    def _validate_each_image(self, image_paths):
        """Validate every image separately; an individual error won't stop the batch."""
        results = []
        for image_path in image_paths:
            try:
                result = self._to_dict(
                    self.validation_image_model.extract_information(image_path)
                )
                result.setdefault("image", Path(image_path).name)
                result["status"] = str(result.get("status", "ERROR")).upper()
                result.setdefault("failure_reasons", [])
            except Exception as exc:
                result = {
                    "image": Path(image_path).name,
                    "status": "ERROR",
                    "failure_reasons": [f"Image validation error: {exc}"],
                }
            results.append(result)
        return results

    def get_validation_info(self, images, invoice, police_report) -> dict:
        """Return aggregate validation fields plus per-image results."""
        image_paths = self._normalize_images(images)
        if not image_paths:
            raise ValueError("At least one image path or URL must be provided.")

        per_image_results = self._validate_each_image(image_paths)

        # A claim can continue when at least one image passes quality validation.
        quality_valid_results = [
            item for item in per_image_results if self._status(item) == "valid"
        ]
        vehicle_results = [
            item for item in quality_valid_results if self._vehicle_detected(item)
        ]

        invoice_police_data = self._to_dict(
            self.validation_invoice_police.extract_information(
                invoice_path=invoice,
                police_report_path=police_report,
            )
        )
        ai_score, ai_details = self.ai_image_detector.predict_batch_from_urls(
            image_urls=image_paths
        )

        image_validation_info = {
            "image_quality_status": "valid" if quality_valid_results else "invalid",
            "vehicle_detected": bool(vehicle_results),
            "vehicle_detected_confidence": max(
                (self._vehicle_confidence(item) for item in vehicle_results),
                default=0.0,
            ),
            "total_images": len(per_image_results),
            "valid_images": len(quality_valid_results),
            "invalid_images": len(per_image_results) - len(quality_valid_results),
            "per_image_results": per_image_results,
        }

        return {
            "ai_score": ai_score,
            "ai_images_details": ai_details,
            "image_validation": image_validation_info,
            "police_report_validation": invoice_police_data["police_report"],
            "invoice_validation": invoice_police_data["invoice"],
        }

    def get_validation_scores(self, validation_info) -> dict:
        return {
            "ai_score": validation_info["ai_score"],
            "vehicle_detected_confidence": (
                validation_info["image_validation"]["vehicle_detected_confidence"]
            ),
        }

    def is_valid(self, validation_info) -> bool:
        """Apply the existing claim-level rules to the aggregated results."""
        ai_score = validation_info["ai_score"]
        image_validation = validation_info["image_validation"]

        if image_validation["image_quality_status"] != "valid":
            return False
        if not validation_info["police_report_validation"]["is_valid"]:
            return False
        if not validation_info["invoice_validation"]["is_valid"]:
            return False
        if not image_validation["vehicle_detected"]:
            return False
        if ai_score >= 3.5:
            return False
        return True
