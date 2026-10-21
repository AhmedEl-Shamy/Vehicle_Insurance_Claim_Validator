import json
import os
import tempfile
import time
from typing import Optional
from dotenv import load_dotenv
import requests
from supabase import Client, create_client

from modules.inputs_analysis_module.inputs_analysis_module import InputsAnalysisModule
from modules.consistency_module.consistency_module import ConsistencyModule
from modules.risk_analysis_module.risk_analysis_module import RiskAnalysisModule
from modules.report_generation_module.report_generation_module import ReportModule
from modules.validation_module.inputs_validation_module import InputsValidationModule

load_dotenv()


class InsuranceValidator:

  def __init__(
      self,
      analysis_module: InputsAnalysisModule,
      consistency_module: ConsistencyModule,
      risk_module: RiskAnalysisModule,
      report_module: ReportModule,
      validation_module: InputsValidationModule,
      supabase_url: Optional[str] = None,
      supabase_key: Optional[str] = None,
  ):
    self.analysis_module = analysis_module
    self.consistency_module = consistency_module
    self.risk_module = risk_module
    self.report_module = report_module
    self.validation_module = validation_module

    # Initialize Supabase
    self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
    self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
    if not self.supabase_url or not self.supabase_key:
      raise ValueError(
          "Supabase URL and API Key must be set via arguments or environment"
          " variables."
      )

    self.supabase: Client = create_client(
        self.supabase_url, self.supabase_key
    )

  @classmethod
  def load_validator(
      cls,
      supabase_url: Optional[str] = None,
      supabase_key: Optional[str] = None,
  ):
    analysis_module = InputsAnalysisModule.load_analysis_module()
    consistency_module = ConsistencyModule.load_module()
    risk_module = RiskAnalysisModule.load_module()
    report_module = ReportModule.load_module()
    validation_module = InputsValidationModule.load_module()

    return cls(
        analysis_module=analysis_module,
        consistency_module=consistency_module,
        risk_module=risk_module,
        report_module=report_module,
        validation_module=validation_module,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
    )

  @staticmethod
  def _download_file(file_url: str, dest_folder: str) -> Optional[str]:
    if not file_url:
      return None
    try:
      response = requests.get(file_url, stream=True, timeout=60)
      response.raise_for_status()
      filename = file_url.split("/")[-1].split("?")[0]
      local_path = os.path.join(dest_folder, filename)

      with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)
      return local_path
    except Exception as e:
      print(f"[ERROR] Failed to download {file_url}: {e}")
      return None

  def _update_stage(
      self, claim_id: str, stage_status: dict, key: str, val: str
  ):
    stage_status[key] = val
    self.supabase.table("claims").update(
        {"pipeline_status": stage_status}
    ).eq("id", claim_id).execute()

  def process_claim(self, claim: dict) -> dict:
    claim_id = claim["id"]
    ref_code = claim.get("claim_reference", claim_id)
    print(f"\n Processing Claim [{ref_code}]...")

    self.supabase.table("claims").update({"status": "Validating"}).eq(
        "id", claim_id
    ).execute()

    validation_errors = []
    invalid_fields = []

    with tempfile.TemporaryDirectory() as temp_dir:
      # Download Artifacts
      invoice_path = self._download_file(claim.get("invoice_url"), temp_dir)
      police_path = self._download_file(
          claim.get("police_report_url"), temp_dir
      )

      image_urls = claim.get("image_urls") or (
          [claim.get("image_url")] if claim.get("image_url") else []
      )
      local_image_paths = []
      for url in image_urls:
        img_p = self._download_file(url, temp_dir)
        if img_p:
          local_image_paths.append(img_p)

      user_description = claim.get("user_description", "")

      # Pre-validation Gatekeeper
      if not image_urls:
        validation_errors.append("No damage photos were provided.")
        invalid_fields.append("Images")
        self.supabase.table("claims").update({
            "status": "Validation Failed",
            "validation_comments": validation_errors,
            "invalid_inputs": invalid_fields,
        }).eq("id", claim_id).execute()
        return {"error": "No image URLs found for this claim."}

      # Layer 1: Inputs Validation Layer

      validation_info = self.validation_module.get_validation_info(
          images=image_urls,
          invoice=claim.get("invoice_url"),
          police_report=claim.get("police_report_url"),
      )
      validation_scores = self.validation_module.get_validation_scores(
          validation_info=validation_info
      )
      image_validation = validation_info["image_validation"]

      if not self.validation_module.is_valid(validation_info=validation_info):
        if image_validation["image_quality_status"] != "valid":
          validation_errors.append("No vehicles detected in images")
          invalid_fields.append("Images")
        if not validation_info["police_report_validation"]["is_valid"]:
          validation_errors.append("Invalid police report")
          invalid_fields.append("Police report")
        if not validation_info["invoice_validation"]["is_valid"]:
          validation_errors.append("Invoice")
          invalid_fields.append("Documents")
        if validation_scores["ai_score"] > 3.5:
          validation_errors.append("AI image detected")
          invalid_fields.append("Images")

        self.supabase.table("claims").update({
            "status": "Validation Failed",
            "validation_comments": validation_errors,
            "invalid_inputs": invalid_fields,
            "ai_image_score": validation_scores["ai_score"],
            "vehicle_confidence": validation_scores[
                "vehicle_detected_confidence"
            ],
            "ai_image_details": validation_info["ai_images_details"],
            "pipeline_status": {
                "narrative": "Skipped",
                "vision": "Skipped",
                "invoice": "Skipped",
                "police": "Skipped",
            },
        }).eq("id", claim_id).execute()

        return {
            "status": "HALTED",
            "claim_id": claim_id,
            "reason": (
                "Validation Failed (Max AI score"
                f" {validation_scores['ai_score']} > 3.5)"
            ),
        }

      # Passed Validation
      self.supabase.table("claims").update({
          "ai_image_score": validation_scores["ai_score"],
          "vehicle_confidence": validation_scores[
              "vehicle_detected_confidence"
          ],
          "ai_image_details": validation_info["ai_images_details"],
      }).eq("id", claim_id).execute()

      pipeline_status = claim.get("pipeline_status") or {
          "police": "PENDING",
          "vision": "PENDING",
          "invoice": "PENDING",
          "narrative": "PENDING",
          "consistency": "PENDING",
          "risk": "PENDING",
          "report": "PENDING",
      }

      self.supabase.table("claims").update({
          "status": "Processing",
          "pipeline_status": pipeline_status,
      }).eq("id", claim_id).execute()

      # Layer 2: Inputs Analysis Layer

      for stage in ["narrative", "invoice", "police", "vision"]:
        self._update_stage(claim_id, pipeline_status, stage, "running")

      try:
        inputs_analysis_info = self.analysis_module.extract_information(
            images=local_image_paths,
            invoice=invoice_path,
            police_report=police_path,
            description=user_description,
        )

        for stage in ["narrative", "invoice", "police", "vision"]:
          self._update_stage(claim_id, pipeline_status, stage, "ok")

      except Exception as e:
        print(f"[ERROR] Monolithic analysis module failed: {e}")
        for stage in ["narrative", "invoice", "police", "vision"]:
          self._update_stage(claim_id, pipeline_status, stage, "error")

        self.supabase.table("claims").update({
            "failed_layer": "Inputs analysis Layer",
            "status": "System Error",
        }).eq("id", claim_id).execute()
        return {"status": "ERROR", "layer": "analysis", "details": str(e)}

    # Save Layer 2 Outputs
    self.supabase.table("claims").update({
        "analysis_results": inputs_analysis_info,
    }).eq("id", claim_id).execute()

    image_json = inputs_analysis_info["image_pipeline"]
    invoice_json = inputs_analysis_info["invoice_pipeline"]
    police_report_json = inputs_analysis_info["police_report_pipeline"]
    description_json = inputs_analysis_info["user_description_pipeline"]
    car_plate_number = claim.get("plate_number", "")

    # Layer 3: Consistency Check Layer
    self._update_stage(claim_id, pipeline_status, "consistency", "running")
    try:
      consistency_info = self.consistency_module.check_consistency(
          image_json=image_json,
          description_json=description_json,
          invoice_json=invoice_json,
          police_report_json=police_report_json,
          plate_number=car_plate_number,
      )
      self._update_stage(claim_id, pipeline_status, "consistency", "ok")
      self.supabase.table("claims").update({
          "consistency_output": consistency_info,
      }).eq("id", claim_id).execute()
    except Exception as e:
      self._update_stage(claim_id, pipeline_status, "consistency", "error")
      self.supabase.table("claims").update({
          "failed_layer": "Consistency layer",
          "status": "System Error",
      }).eq("id", claim_id).execute()
      print(f"[ERROR] Consistency layer error: {e}")
      return {"status": "ERROR", "layer": "consistency", "details": str(e)}


    # Layer 4: Risk Analysis Layer

    self._update_stage(claim_id, pipeline_status, "risk", "running")
    try:
      risk_info = self.risk_module.get_risk_info(
          consistency_info=consistency_info,
          validation_scores=validation_scores,
      )
      risk_class = json.loads(risk_info)["predicted_risk"]
      self._update_stage(claim_id, pipeline_status, "risk", "ok")
      self.supabase.table("claims").update({
          "risk": risk_class,
      }).eq("id", claim_id).execute()
    except Exception as e:
      self._update_stage(claim_id, pipeline_status, "risk", "error")
      self.supabase.table("claims").update({
          "failed_layer": "Risk layer",
          "status": "System Error",
      }).eq("id", claim_id).execute()
      print(f"[ERROR] Risk layer error: {e}")
      return {"status": "ERROR", "layer": "risk", "details": str(e)}

    # Layer 5: Report Generation Layer
    
    self._update_stage(claim_id, pipeline_status, "report", "running")
    try:
      report = self.report_module.generate_report(
          validation_scores=validation_scores,
          consistency_result=consistency_info,
          risk_class=risk_class,
      )
      self._update_stage(claim_id, pipeline_status, "report", "ok")
      self.supabase.table("claims").update({
          "report": report,
      }).eq("id", claim_id).execute()
    except Exception as e:
      self._update_stage(claim_id, pipeline_status, "report", "error")
      self.supabase.table("claims").update({
          "failed_layer": "Report generation layer",
          "status": "System Error",
      }).eq("id", claim_id).execute()
      print(f"[ERROR] Report generation error: {e}")
      return {"status": "ERROR", "layer": "report", "details": str(e)}

    # Complete Pipeline Execution
    self.supabase.table("claims").update({
        "status": "Pending Decision"
    }).eq("id", claim_id).execute()

    print(f" Finished processing Claim [{ref_code}]!")
    return {"status": "SUCCESS", "claim_id": claim_id}

  def start_polling(self, poll_interval: int = 5):
    print(
        " Insurance Validator running. Polling Supabase for"
        " 'Submitted' claims..."
    )
    while True:
      try:
        response = (
            self.supabase.table("claims")
            .select("*")
            .eq("status", "Submitted")
            .execute()
        )
        pending_claims = response.data

        if pending_claims:
          print(f" Found {len(pending_claims)} new claim(s).")
          for claim in pending_claims:
            self.process_claim(claim)
      except Exception as e:
        print(f"[ERROR] Monolith Worker Loop Exception: {e}")

      time.sleep(poll_interval)


if __name__ == "__main__":
  validator = InsuranceValidator.load_validator()
  validator.start_polling(poll_interval=5)