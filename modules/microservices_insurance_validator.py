import json
import os
import tempfile
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests
from supabase import Client, create_client

# Internal Pipeline Modules
from modules.consistency_module.consistency_module import ConsistencyModule
from modules.validation_module.inputs_validation_module import InputsValidationModule
from modules.report_generation_module.report_generation_module import ReportModule
from modules.risk_analysis_module.risk_analysis_module import RiskAnalysisModule

load_dotenv()


# =====================================================================
# 1. MICROSERVICE CLIENT WRAPPERS
# =====================================================================


class PoliceReportClient:

  def __init__(self, service_url: str):
    self.service_url = service_url.rstrip("/")

  def extract(self, file_path: str) -> dict:
    if not file_path:
      raise ValueError("No police report file provided")
    with open(file_path, "rb") as f:
      response = requests.post(f"{self.service_url}/extract", files={"file": f})
      response.raise_for_status()
      return response.json()


class InvoiceModelClient:

  def __init__(self, service_url: str):
    self.service_url = service_url.rstrip("/")

  def extract_information(self, invoice: str) -> dict:
    if not invoice:
      raise ValueError("No invoice file provided")
    with open(invoice, "rb") as f:
      response = requests.post(f"{self.service_url}/extract", files={"file": f})
      response.raise_for_status()
      return response.json()


class UserDescriptionModelClient:

  def __init__(self, service_url: str):
    self.service_url = service_url.rstrip("/")

  def extract_information(self, description: str) -> dict:
    if not description:
      raise ValueError("No user description provided")
    response = requests.post(
        f"{self.service_url}/extract",
        json={"description": description},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


class ImageModelClient:

  def __init__(self, service_url: str):
    self.service_url = service_url.rstrip("/")

  def extract_information(self, images: List[str]) -> dict:
    if not images:
      raise ValueError("No images provided")

    opened_files = [open(img_path, "rb") for img_path in images]
    files_payload = [
        ("files", (path.split("/")[-1], file_obj, "image/jpeg"))
        for path, file_obj in zip(images, opened_files)
    ]

    try:
      response = requests.post(
          f"{self.service_url}/extract", files=files_payload
      )
      response.raise_for_status()
      return response.json()
    finally:
      for file_obj in opened_files:
        file_obj.close()


# =====================================================================
# 2. CENTRAL PIPELINE ORCHESTRATOR
# =====================================================================


class ClaimPipelineWorker:

  def __init__(
      self,
      supabase_url: Optional[str] = None,
      supabase_key: Optional[str] = None,
      police_url: Optional[str] = None,
      invoice_url: Optional[str] = None,
      user_desc_url: Optional[str] = None,
      image_url: Optional[str] = None,
  ):
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

    # Initialize Service Clients
    self.police_client = PoliceReportClient(
        police_url or os.getenv("POLICE_SERVICE_URL", "http://127.0.0.1:8004")
    )
    self.invoice_client = InvoiceModelClient(
        invoice_url or os.getenv("INVOICE_SERVICE_URL", "http://127.0.0.1:8003")
    )
    self.user_desc_client = UserDescriptionModelClient(
        user_desc_url
        or os.getenv("DESCRIPTION_SERVICE_URL", "http://127.0.0.1:8001")
    )
    self.image_client = ImageModelClient(
        image_url or os.getenv("IMAGE_SERVICE_URL", "http://127.0.0.1:8002")
    )

    # Initialize Internal Modules
    self.validation_module = InputsValidationModule.load_module()
    self.consistency_module = ConsistencyModule.load_module()
    self.risk_module = RiskAnalysisModule.load_module()
    self.report_module = ReportModule.load_module()

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

  def _update_stage(self, claim_id: str, stage_status: dict, key: str, val: str):
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

    inputs_analysis_info = {}
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

      # ==================================================
      # Layer 1: Inputs Validation Layer
      # ==================================================
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

      # ==================================================
      # Layer 2: Inputs Analysis Microservices
      # ==================================================
      # 1. User Description Microservice
      self._update_stage(claim_id, pipeline_status, "narrative", "running")
      try:
        res = self.user_desc_client.extract_information(
            description=user_description
        )
        inputs_analysis_info["user_description_pipeline"] = res
        self._update_stage(claim_id, pipeline_status, "narrative", "ok")
      except Exception as e:
        print(f"[ERROR] Narrative service error: {e}")
        self._update_stage(claim_id, pipeline_status, "narrative", "error")
        self.supabase.table("claims").update({
            "failed_layer": "Inputs analysis Layer",
            "status": "System Error",
        }).eq("id", claim_id).execute()
        return {"status": "ERROR", "layer": "narrative", "details": str(e)}

      # 2. Invoice Microservice
      self._update_stage(claim_id, pipeline_status, "invoice", "running")
      try:
        res = self.invoice_client.extract_information(invoice=invoice_path)
        inputs_analysis_info["invoice_pipeline"] = res
        self._update_stage(claim_id, pipeline_status, "invoice", "ok")
      except Exception as e:
        print(f"[ERROR] Invoice service error: {e}")
        self._update_stage(claim_id, pipeline_status, "invoice", "error")
        self.supabase.table("claims").update({
            "failed_layer": "Inputs analysis Layer",
            "status": "System Error",
        }).eq("id", claim_id).execute()
        return {"status": "ERROR", "layer": "invoice", "details": str(e)}

      # 3. Police Report Microservice
      self._update_stage(claim_id, pipeline_status, "police", "running")
      try:
        res = self.police_client.extract(file_path=police_path)
        inputs_analysis_info["police_report_pipeline"] = res
        self._update_stage(claim_id, pipeline_status, "police", "ok")
      except Exception as e:
        print(f"[ERROR] Police report service error: {e}")
        self._update_stage(claim_id, pipeline_status, "police", "error")
        self.supabase.table("claims").update({
            "failed_layer": "Inputs analysis Layer",
            "status": "System Error",
        }).eq("id", claim_id).execute()
        return {"status": "ERROR", "layer": "police", "details": str(e)}

      # 4. Vision Microservice
      self._update_stage(claim_id, pipeline_status, "vision", "running")
      try:
        res = self.image_client.extract_information(images=local_image_paths)
        inputs_analysis_info["image_pipeline"] = res
        self._update_stage(claim_id, pipeline_status, "vision", "ok")
      except Exception as e:
        print(f"[ERROR] Vision service error: {e}")
        self._update_stage(claim_id, pipeline_status, "vision", "error")
        self.supabase.table("claims").update({
            "failed_layer": "Inputs analysis Layer",
            "status": "System Error",
        }).eq("id", claim_id).execute()
        return {"status": "ERROR", "layer": "vision", "details": str(e)}

    # Save Layer 2 Outputs
    self.supabase.table("claims").update({
        "analysis_results": inputs_analysis_info,
    }).eq("id", claim_id).execute()

    image_json = json.dumps(inputs_analysis_info["image_pipeline"])
    invoice_json = inputs_analysis_info["invoice_pipeline"]
    police_report_json = json.dumps(
        inputs_analysis_info["police_report_pipeline"]
    )
    description_json = inputs_analysis_info["user_description_pipeline"]
    car_plate_number = claim.get("plate_number", "")

    # ==================================================
    # Layer 3: Consistency Check Layer
    # ==================================================
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

    # ==================================================
    # Layer 4: Risk Analysis Layer
    # ==================================================
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

    # ==================================================
    # Layer 5: Report Generation Layer
    # ==================================================
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
        " Claim Pipeline Worker running. Polling Supabase for 'Submitted'"
        " claims..."
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
        print(f"[ERROR] Worker Loop Exception: {e}")

      time.sleep(poll_interval)


if __name__ == "__main__":
  worker = ClaimPipelineWorker()
  worker.start_polling(poll_interval=5)