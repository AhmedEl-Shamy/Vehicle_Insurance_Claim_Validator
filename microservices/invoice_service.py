import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pyngrok import ngrok
import uvicorn

# Shared model imports
from modules.inputs_analysis_module.gpt_invoice_model import InvoiceModelGPT
from modules.inputs_analysis_module.invoice_model import InvoiceModel

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Invoice Service",
    description="Microservice for invoice data extraction with Gemini and GPT fallback.",
    version="1.0.0",
)

# Initialize primary and fallback models
invoice_model = InvoiceModel()
gpt_invoice_model = InvoiceModelGPT()


@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
  tmp_path = None
  try:
    ext = os.path.splitext(file.filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
      tmp.write(await file.read())
      tmp_path = tmp.name

    # 1. Attempt Primary Extraction (Gemini)
    try:
      print("[INFO] Attempting extraction with Gemini InvoiceModel...")
      return invoice_model.extract_information(invoice=tmp_path)
    except Exception as gemini_err:
      # 2. Fallback to GPT model on primary failure
      print(
          f"[WARNING] Gemini model failed ({gemini_err}). Falling back to GPT"
          " model..."
      )
      return gpt_invoice_model.extract_information(invoice=tmp_path)

  except Exception as e:
    print(f"[ERROR] Both primary and fallback invoice models failed: {e}")
    raise HTTPException(
        status_code=500, detail=f"Invoice extraction failed: {str(e)}"
    ) from e

  finally:
    if tmp_path and os.path.exists(tmp_path):
      os.remove(tmp_path)


@app.get("/health")
async def health_check():
  return {"status": "healthy", "service": "invoice-processing"}


if __name__ == "__main__":
  port = int(os.getenv("INVOICE_SERVICE_PORT", 8003))
  ngrok_token = os.getenv("INVOICE_NGROK_TOKEN", "").strip()

  if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print(f"\n Invoice Service Live At: {public_url}\n")
  else:
    print(
        f"\n No Ngrok token provided. Running locally at:"
        f" http://127.0.0.1:{port}\n"
    )

  uvicorn.run(app, host="0.0.0.0", port=port)