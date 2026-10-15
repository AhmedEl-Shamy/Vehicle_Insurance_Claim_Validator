import base64
import io
import json
import os
import time
from PIL import Image
import pymupdf
import requests

# OpenRouter API Key
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
)


def resolve_file_path(path: str) -> str:
    """Checks the file path and supports both PDF and image files."""
    if os.path.isdir(path):
        valid_extensions = (".pdf", ".jpg", ".jpeg", ".png", ".webp", "docx")
        files = [
            f
            for f in os.listdir(path)
            if f.lower().endswith(valid_extensions) and not f.startswith(".")
        ]
        if not files:
            raise FileNotFoundError(
                f"No valid PDF or image files found in directory: {path}"
            )
        selected_file = os.path.join(path, files[0])
        return selected_file

    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist at path: {path}")
    return path


def process_and_encode_file(
    file_path: str, max_size=(1024, 1024), quality=85
) -> str:
    """Converts a PDF page or image into a compressed base64 JPEG string."""
    actual_path = resolve_file_path(file_path)

    if actual_path.lower().endswith(".pdf"):
        doc = pymupdf.open(actual_path)
        if len(doc) == 0:
            raise ValueError(f"PDF file is empty: {actual_path}")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        img = Image.open(actual_path)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def validate_input_documents(
    invoice_path: str, police_report_path: str, max_retries: int = 3
) -> dict:
    if not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
        err_msg = "Invalid API key! OpenRouter keys must start with 'sk-or-v1-'"
        return {
            "invoice": {"is_valid": False, "reason": err_msg},
            "police_report": {"is_valid": False, "reason": err_msg},
        }

    try:
        invoice_b64 = process_and_encode_file(invoice_path)
        police_b64 = process_and_encode_file(police_report_path)
    except Exception as e:
        err_msg = f"Error processing input files (PDF/Image): {str(e)}"
        return {
            "invoice": {"is_valid": False, "reason": err_msg},
            "police_report": {"is_valid": False, "reason": err_msg},
        }

    prompt = """
    You are an Input Validation Gate for an insurance system.
    Analyze the two provided documents carefully:
    - Document 1 is claimed to be a Repair/Parts Invoice.
    - Document 2 is claimed to be an Official Police Report.

    Verify if each document matches its claimed type.
    
    Return strictly a JSON object with this exact structure:
    {
        "invoice": {
            "is_valid": true/false,
            "reason": "Provide ONLY the specific rejection reason if is_valid is false. If is_valid is true, this MUST be an empty string ''"
        },
        "police_report": {
            "is_valid": true/false,
            "reason": "Provide ONLY the specific rejection reason if is_valid is false. If is_valid is true, this MUST be an empty string ''"
        }
    }
    """

    payload = {
        "model": "google/gemini-2.5-flash",
        "max_tokens": 1000,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{invoice_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{police_b64}"},
                },
            ],
        }],
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result_text = response.json()["choices"][0]["message"]["content"]
                return json.loads(result_text)
            elif response.status_code in [500, 502, 503, 504]:
                time.sleep(2)
                continue
            else:
                err_msg = f"Server connection error HTTP {response.status_code}: {response.text}"
                return {
                    "invoice": {"is_valid": False, "reason": err_msg},
                    "police_report": {"is_valid": False, "reason": err_msg},
                }

        except Exception as e:
            if attempt == max_retries:
                err_msg = f"Connection failed: {str(e)}"
                return {
                    "invoice": {"is_valid": False, "reason": err_msg},
                    "police_report": {"is_valid": False, "reason": err_msg},
                }
            time.sleep(2)


# ==========================================
# Run Execution
# ==========================================

validation_result = validate_input_documents(
    "/kaggle/input/datasets/nataliemmoris/bta2aaa",
    "/kaggle/input/datasets/nataliemmoris/pppppppppppp/MV_104A_Report_1.pdf",
)


print(json.dumps(validation_result, indent=4, ensure_ascii=False))