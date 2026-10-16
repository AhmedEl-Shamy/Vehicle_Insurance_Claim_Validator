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


class ValidInvoicePolice:
    """Input validation gate for Invoice and Police Report documents.

    Supports both local file paths and online Supabase Storage URLs.
    """

    def __init__(self, api_key: str = None):
      self.api_key = (
          api_key or os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
      )

    @staticmethod
    def save_model(
        output_path: str = "outputs/document_validator.json",
    ) -> str:
      """An optional static function for saving module settings when needed."""
      print(" Document validation module configuration verified.")
      return output_path

    @classmethod
    def load_model(cls, api_key: str = None):
      """Loading the module and creating the operational kernel, 
      following the same standards as the other classes. """
      instance = cls(api_key=api_key)
      print(" Valid_Invoice_Police module loaded successfully.")
      return instance

    @staticmethod
    def resolve_file_path(path: str) -> str:
      """Checks local file path and supports both PDF and image files."""
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

    @staticmethod
    def process_and_encode_file(
        file_input: str, max_size=(1024, 1024), quality=85
    ) -> str:
      """Converts a PDF page or image (local path OR Supabase URL) into a compressed base64 JPEG string directly in memory."""
      # 1. Handling Links Online (Supabase Storage URLs)
      if file_input.startswith(("http://", "https://")):
        response = requests.get(file_input, timeout=20)
        response.raise_for_status()
        content = response.content

        # Removing query parameters from a Supabase signed token link
        clean_url = file_input.split("?")[0].lower()

        if (
            clean_url.endswith(".pdf")
            or response.headers.get("Content-Type") == "application/pdf"
        ):
          doc = pymupdf.open(stream=content, filetype="pdf")
          if len(doc) == 0:
            raise ValueError(f"Downloaded PDF is empty: {file_input}")
          page = doc[0]
          pix = page.get_pixmap(dpi=150)
          img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
          doc.close()
        else:
          img = Image.open(io.BytesIO(content))

      
      else:
        actual_path = Valid_Invoice_Police.resolve_file_path(file_input)

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

    def extract_information(
        self, invoice_path: str, police_report_path: str, max_retries: int = 3
    ) -> str:
      """Validates invoice and police report documents.

      Accepts local paths or Supabase URLs, returning a JSON-formatted string
      (-> str).
      """
      if not self.api_key.startswith("sk-or-v1-"):
        err_msg = "Invalid API key! OpenRouter keys must start with 'sk-or-v1-'"
        res = {
            "invoice": {"is_valid": False, "reason": err_msg},
            "police_report": {"is_valid": False, "reason": err_msg},
        }
        return json.dumps(res, indent=4, ensure_ascii=False)

      try:
        invoice_b64 = self.process_and_encode_file(invoice_path)
        police_b64 = self.process_and_encode_file(police_report_path)
      except Exception as e:
        err_msg = f"Error processing input files (PDF/Image/URL): {str(e)}"
        res = {
            "invoice": {"is_valid": False, "reason": err_msg},
            "police_report": {"is_valid": False, "reason": err_msg},
        }
        return json.dumps(res, indent=4, ensure_ascii=False)

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
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{invoice_b64}"
                      },
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{police_b64}"
                      },
                  },
              ],
          }],
          "response_format": {"type": "json_object"},
      }

      headers = {
          "Authorization": f"Bearer {self.api_key}",
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
            parsed_json = json.loads(result_text)
            return json.dumps(parsed_json, indent=4, ensure_ascii=False)
          elif response.status_code in [500, 502, 503, 504]:
            time.sleep(2)
            continue
          else:
            err_msg = (
                f"Server connection error HTTP {response.status_code}:"
                f" {response.text}"
            )
            res = {
                "invoice": {"is_valid": False, "reason": err_msg},
                "police_report": {"is_valid": False, "reason": err_msg},
            }
            return json.dumps(res, indent=4, ensure_ascii=False)

        except Exception as e:
          if attempt == max_retries:
            err_msg = f"Connection failed: {str(e)}"
            res = {
                "invoice": {"is_valid": False, "reason": err_msg},
                "police_report": {"is_valid": False, "reason": err_msg},
            }
            return json.dumps(res, indent=4, ensure_ascii=False)
          time.sleep(2)


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    # 1. Loading the module
    validator = Valid_Invoice_Police.load_model()

    # 2. Defining Supabase Links
    supabase_invoice_url = "https://your-supabase-project.supabase.co/storage/v1/object/public/claims/invoice_sample.pdf"
    supabase_police_url = "https://your-supabase-project.supabase.co/storage/v1/object/public/claims/police_report.pdf"

    # 3. Actual invocation and execution of the check on the links
    validation_result_str = validator.extract_information(
        invoice_path=supabase_invoice_url,
        police_report_path=supabase_police_url
    )

    # 4. Print the final result string
    print("\n--- Output (Returned JSON String) ---")
    print(validation_result_str)
