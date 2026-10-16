from modules.inputs_analysis_module.inputs_analysis_module import InputsAnalysisModule
from modules.consistency_module.consistency_module import ConsistencyModule
from modules.risk_analysis_module.risk_analysis_module import RiskAnalysisModule
from modules.report_generation_module.report_generation_module import ReportModule
from modules.validation_module.inputs_validation_module import InputsValidationModule

import json

class InputsNotValidError(Exception):
    """Raised when the inputs is not valid (Violates our terms)"""

class InsuranceValidator:
    def __init__(
        self,
        analysis_module: InputsAnalysisModule,
        consistency_module: ConsistencyModule,
        risk_module: RiskAnalysisModule,
        report_module: ReportModule,
        validation_module: InputsValidationModule
    ):
        self.analysis_module = analysis_module
        self.consistency_module = consistency_module
        self.risk_module = risk_module
        self.report_module = report_module
        self.validation_module = validation_module

    @classmethod
    def load_validator(cls):
        analysis_module = InputsAnalysisModule.load_analysis_module()
        consistency_module = ConsistencyModule.load_module()
        risk_module = RiskAnalysisModule.load_module()
        report_module = ReportModule.load_module()
        validation_module = InputsValidationModule.load_module()

        return cls(
            analysis_module = analysis_module,
            consistency_module = consistency_module,
            risk_module = risk_module,
            report_module = report_module,
            validation_module = validation_module
        )


    def getReport(self, images, police_report, invoice, description, car_plate_number) -> str:

        # ==================================================
        # Layer 1: Validation Layer
        # ==================================================
        validation_info = self.validation_module.get_validation_info(
            images=images,
            invoice=invoice,
            police_report=police_report
        )
        validation_scores = self.validation_module.get_validation_scores(validation_info=validation_info)

        if self.validation_module.is_valid(validation_info=validation_info):

            # ==================================================
            # Layer 2: Inputs Analysis Layer
            # ==================================================
            inputs_analysis_info = self.analysis_module.extract_information(
                images=images,
                invoice=invoice,
                police_report=police_report,
                description=description,
            )

            image_json = inputs_analysis_info['image_pipeline']
            invoice_json = inputs_analysis_info['invoice_pipeline']
            police_report_json = inputs_analysis_info['police_report_pipeline']
            description_json = inputs_analysis_info['user_description_pipeline']

            # ==================================================
            # Layer 3: Consistency Check Layer
            # ==================================================
            consistency_info = self.consistency_module.check_consistency(
                image_json = image_json,
                description_json = description_json,
                invoice_json = invoice_json,
                police_report_json = police_report_json,
                plate_number = car_plate_number
            )

            # ==================================================
            # Layer 4: Risk Analysis Layer
            # ==================================================
            risk_info = self.risk_module.get_risk_info(
                consistency_info = consistency_info,
                validation_scores = validation_scores
            )
            risk_class = json.loads(risk_info)['predicted_risk']

            # ==================================================
            # Layer 5: Report Generation Layer
            # ==================================================
            report = self.report_module.generate_report(
                validation_scores = validation_scores,
                consistency_result = consistency_info,
                risk_class = risk_class
            )

            return report

        else:
            raise InputsNotValidError
