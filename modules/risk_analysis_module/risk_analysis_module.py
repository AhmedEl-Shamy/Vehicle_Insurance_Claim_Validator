import json
import logging
import os
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
)
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


class RiskAnalysisModule:

  def __init__(self, api_key: str = None):
    self.api_key = (
        api_key or os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    )

    self.headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "Insurance Fraud Risk Layer",
    }

  @staticmethod
  def save_model(output_path: str = "outputs/risk_model.json"):
    print(" RiskAnalysisModule configuration verified.")
    return output_path

  @classmethod
  def load_module(cls, api_key: str = None):
    instance = cls(api_key=api_key)
    print(" RiskAnalysisModule loaded successfully.")
    return instance

  def _build_jev_request(self, payload: dict) -> dict:
    return {
        "model": "~typesafe/jev-latest",
        "state": payload,
        "questions": {
            "is_visible_ghost_billing": {
                "type": "choice",
                "instructions": (
                    "Analyze missing parts. Are there visible outer structural"
                    " body parts (e.g., doors, roof, roof panel, fenders,"
                    " quarter panel) billed strictly in the invoice without"
                    " proof in photos/police?"
                ),
                "criteria": {
                    "TRUE": (
                        "Yes, unverified visible outer structural parts exist"
                        " in invoice."
                    ),
                    "FALSE": (
                        "No, all visible parts are verified or discrepancies"
                        " are purely internal/non-structural."
                    ),
                },
            },
            "ai_image_threat": {
                "type": "choice",
                "instructions": (
                    "Inspect 'images_analysis' array in state. Evaluate PoliMi"
                    " score S or ai_level: S >= 1.00 (or ai_level == 3) is HIGH"
                    " risk; 0.05 <= S < 1.00 (or ai_level == 2) is MILD risk; S"
                    " < 0.05 (or ai_level == 1) is NONE."
                ),
                "criteria": {
                    "AI_RISK_HIGH": (
                        "At least one image has PoliMi score S >= 1.00 or"
                        " ai_level == 3."
                    ),
                    "AI_RISK_MILD": (
                        "No image has S >= 1.00, but at least one image has"
                        " 0.05 <= S < 1.00 or ai_level == 2."
                    ),
                    "AI_RISK_NONE": (
                        "All images have S < 0.05 or ai_level == 1 OR"
                        " 'images_analysis' is missing/empty."
                    ),
                },
            },
            "vehicle_detection_status": {
                "type": "choice",
                "instructions": (
                    "Inspect 'images_analysis' array in state. Evaluate vehicle"
                    " detection results from ValidationImage (vehicle_detected"
                    " and vehicle_confidence). Flag NO_VEHICLE_DETECTED if"
                    " vehicle_detected is False or vehicle_confidence < 0.05"
                    " (conf threshold)."
                ),
                "criteria": {
                    "NO_VEHICLE_DETECTED": (
                        "At least one image has vehicle_detected == False or"
                        " vehicle_confidence < 0.05."
                    ),
                    "LOW_CONFIDENCE": (
                        "Vehicle detected, but vehicle_confidence is between"
                        " 0.05 and 0.20."
                    ),
                    "VEHICLE_CONFIRMED": (
                        "All images have vehicle_detected == True and"
                        " vehicle_confidence >= 0.20 OR 'images_analysis' is"
                        " missing/empty."
                    ),
                },
            },
            "risk_level": {
                "type": "choice",
                "instructions": (
                    "Evaluate overall risk level considering text consistency,"
                    " AI image threat (PoliMi Feature), and vehicle detection"
                    " confidence (ValidationImage)."
                ),
                "criteria": {
                    "HIGH": (
                        "ai_image_threat is AI_RISK_HIGH OR"
                        " is_visible_ghost_billing is TRUE OR"
                        " overall_consistency_score < 0.60 OR"
                        " vehicle_detection_status is NO_VEHICLE_DETECTED."
                    ),
                    "MEDIUM": (
                        "ai_image_threat is AI_RISK_MILD OR"
                        " vehicle_detection_status is LOW_CONFIDENCE OR"
                        " internal structural ghost billing exists OR"
                        " overall_consistency_score is between 0.60 and 0.79."
                    ),
                    "LOW": (
                        "ai_image_threat is AI_RISK_NONE AND"
                        " vehicle_detection_status is VEHICLE_CONFIRMED AND no"
                        " ghost billing AND overall_consistency_score >= 0.80."
                    ),
                },
            },
        },
    }

  def evaluate_claim(self, payload: dict) -> dict:
    try:
      req_body = self._build_jev_request(payload)
      response = requests.post(
          DECISIONS_URL, headers=self.headers, json=req_body, timeout=10
      )

      if response.status_code == 200:
        res_data = response.json().get("answers", {})

        risk_choice = res_data.get("risk_level", {}).get("choice", "MEDIUM")
        ai_threat = res_data.get("ai_image_threat", {}).get(
            "choice", "AI_RISK_NONE"
        )
        vehicle_status = res_data.get("vehicle_detection_status", {}).get(
            "choice", "VEHICLE_CONFIRMED"
        )
        jev_has_ghost_visible = (
            res_data.get("is_visible_ghost_billing", {}).get("choice") == "TRUE"
        )

        
        if (
            risk_choice == "HIGH"
            or ai_threat == "AI_RISK_HIGH"
            or jev_has_ghost_visible
            or vehicle_status == "NO_VEHICLE_DETECTED"
        ):
          action = "REJECT_AND_REFER_TO_SIU"
          risk_choice = "HIGH"
        elif (
            risk_choice == "MEDIUM"
            or ai_threat == "AI_RISK_MILD"
            or vehicle_status == "LOW_CONFIDENCE"
        ):
          action = "MANUAL_HUMAN_REVIEW"
          risk_choice = "MEDIUM"
        else:
          action = "APPROVE_OR_FAST_TRACK"
          risk_choice = "LOW"

        return {
            "predicted_risk": risk_choice,
            "recommended_action": action,
            "jev_evaluations": {
                "ai_image_threat": ai_threat,
                "vehicle_detection_status": vehicle_status,
                "visible_ghost_billing_detected": jev_has_ghost_visible,
            },
        }
      else:
        logging.warning(
            f"Jev API returned HTTP {response.status_code}. Executing"
            " Fallback."
        )

    except Exception as e:
      logging.error(
          f"Error communicating with Jev API: {e}. Executing Fallback."
      )

    # Emergency Fallback
    return {
        "predicted_risk": "MEDIUM",
        "recommended_action": "MANUAL_HUMAN_REVIEW",
        "jev_evaluations": {
            "status": "FALLBACK_DUE_TO_API_FAILURE",
            "message": (
                "JEV Service unavailable. Claim safely routed to Manual Human"
                " Review."
            ),
        },
    }

  def get_risk_info(self, consistency_info: dict, validation_scores) -> str:
    payload = {}

    # 1. add consistency_info
    if isinstance(consistency_info, dict):
      payload.update(consistency_info)

    # 2.add validation_scores in array of images_analysis
    if isinstance(validation_scores, list):
      payload["images_analysis"] = validation_scores
    elif isinstance(validation_scores, dict):
      if "images_analysis" in validation_scores:
        payload["images_analysis"] = validation_scores["images_analysis"]
      else:
        payload["images_analysis"] = [validation_scores]

    # 3. Calculating Risk Level and Making Decisions via JEV
    result_dict = self.evaluate_claim(payload)

    # 4.Returning the result as a String, based on the exact matching type hint (-> str)
    return json.dumps(result_dict, indent=2, ensure_ascii=False)
