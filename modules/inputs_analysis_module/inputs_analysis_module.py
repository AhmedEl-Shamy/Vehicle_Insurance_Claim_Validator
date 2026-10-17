
from invoice_model import InvoiceModel, SMART_INVOICE_PROMPT
from police_report_model import PoliceReportExtractor
from user_description_model import UserDescriptionModel
from image_detection_segmentation import ImageDetectionSegmentation


class InputsAnalysisModule:
    def __init__(
        self,
        invoice_model: InvoiceModel,
        police_model: PoliceReportExtractor,
        user_description_model: UserDescriptionModel,
        images_model: ImageDetectionSegmentation,
    ):
        self.invoice_model = invoice_model
        self.police_model = police_model
        self.user_description_model = user_description_model
        self.images_model = images_model

    @classmethod
    def load_analysis_module(cls):
        images_model = ImageDetectionSegmentation.load_model(
            segmentation_model_path=ImageDetectionSegmentation.SEGMENTATION_MODEL_PATH,
            damage_model_path=ImageDetectionSegmentation.DAMAGE_MODEL_PATH,
        )

        police_report_model = PoliceReportExtractor.load_model()
        invoice_model = InvoiceModel()
        user_description_model = UserDescriptionModel.load_model()

        return cls(
            invoice_model=invoice_model,
            police_model=police_report_model,
            user_description_model=user_description_model,
            images_model=images_model,
        )

    def extract_information(
        self, images, description, invoice, police_report
    ) -> dict:
        # Process all supplied images, including Supabase URLs.
        image_pipeline = self.images_model.extract_information_batch(
            images=images
        )

        info = {
            "image_pipeline": image_pipeline,
            "user_description_pipeline":
                self.user_description_model.extract_information(
                    user_description=description
                ),
            "invoice_pipeline":
                self.invoice_model.extract_information(
                    invoice=invoice
                ),
            "police_report_pipeline":
                self.police_model.extract(
                    report=police_report
                ).model_dump_json(),
        }

        return info
