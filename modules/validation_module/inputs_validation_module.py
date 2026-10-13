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
        validation_info = {
            "ai_validation": self.ai_check_model.extract_information(),
            "image_validation": self.validation_image_model.extract_information()
        }
        return validation_info
    
    def getAIScore(self, validation_info):
        """Return The AI score from validation info (get_validation_info returned value)"""
        ai_validation_info = json.loads(validation_info['ai_validation'])

        # Modify this line to get ai_score
        ai_score = 4
        return ai_score

    def get_image_quality_scores(self, validation_info):
        "Return The Image Quality scores from validation info (get_validation_info returned value)"
        image_quality_info = json.loads(validation_info['image_validation'])

        # Modify these lines to get image_qulaity_score and image_quality_confidence
        image_quality_score = 2
        image_quality_confidence = 0.9
        return image_quality_score, image_quality_confidence


    def get_validation_scores(self, images) -> dict:
        """Get the validation information of inputs"""
        validation_info = self.get_validation_info(images)
        ai_score = self.getAIScore(validation_info=validation_info)
        image_quality_score, image_quality_confidence = self.get_image_quality_scores(validation_info=validation_info)

        validation_scores = {
            "ai_score": ai_score,
            "image_quality_score": image_quality_score,
            "image_quality_confidence": image_quality_confidence
        }
        return validation_scores

    def is_valid(self, validation_scores) -> bool:
        """Return False only if the inputs violates our Rules"""

        # Check if inputs voilates our rules ===> RETURN FALSE

        return True

    