from modules.validation_module.ai_check_model import AICheckModel
from modules.validation_module.validation_image import ValidationImage
import json


class InputsValidationModule:
    def __init__(
            self, 
            ai_check_model: AICheckModel,
            validation_image_model: ValidationImage
    ):
        self.ai_check_model = ai_check_model
        self.validation_image_model = validation_image_model

    @classmethod
    def load_module(cls):
        ai_check_model = AICheckModel.load_model()
        validation_image_model = ValidationImage.load_model()

        return cls(
            ai_check_model =  ai_check_model,
            validation_image_model = validation_image_model
        )

    def get_validation_info(self, images) -> dict:
        """Get the validation information of inputs"""
        image_validation_data = json.loads(self.validation_image_model.extract_information())
        image_validation_info = {
            'image_quality_status': image_validation_data['status'].lower(),
            'vehicle_detected': image_validation_data['vehicle_detected'],
            'vehicle_detected_confidence': image_validation_data['vehicle_confidence']
        }
        validation_info = {
            "ai_score": self.ai_check_model.get_ai_score(),
            "image_validation": image_validation_info
        }
        return validation_info

    def get_validation_scores(self, validation_info) -> dict:
        """Get the validation information of inputs"""
        ai_score = validation_info['ai_score']
        vehicle_detected_confidence = validation_info['image_validation']['vehicle_detected_confidence']

        validation_scores = {
            "ai_score": ai_score,
            "vehicle_detected_confidence": vehicle_detected_confidence
        }
        return validation_scores

    def is_valid(self, validation_info) -> bool:
        """Return False only if the inputs violates our Rules"""
        ai_score = validation_info['ai_score']
        image_status = validation_info['image_validation']['image_quality_status']
        is_car_detected = validation_info['image_validation']['vehicle_detected']

        if image_status != "valid": return False
        if not is_car_detected: return False
        if ai_score >= 3.5 : return False

        return True

    