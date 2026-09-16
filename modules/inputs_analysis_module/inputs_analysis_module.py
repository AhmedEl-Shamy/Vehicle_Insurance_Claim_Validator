from invoice_model import InvoiceModel
from police_report_model import PoliceReportModel
from user_description_model import UserDescriptionModel
from images_model import ImagesModel

class InputsAnalysisModule:
    def __init__(
            self,
            invoice_model: InvoiceModel,
            police_model: PoliceReportModel,
            user_description_model: UserDescriptionModel,
            images_model: ImagesModel
    ):
        self.invoice_model = invoice_model
        self.police_model = police_model
        self.user_description_model = user_description_model
        self.images_model = images_model

    def extract_information(self, images, description, invoice, police_report) -> dict:
        info = {
            'image_pipeline' : self.images_model.extract_information(images=images),
            'user_description_pipeline': self.user_description_model.extract_information(
                user_description=description
            ),
            'invoice_pipeline': self.invoice_model.extract_information(invoice=invoice),
            'police_report_pipeline': self.police_model.extract_information(report=police_report),
        }
        return info