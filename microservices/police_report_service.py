import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from pyngrok import ngrok
import uvicorn

# Import shared model from models directory
from modules.inputs_analysis_module.police_report_model import PoliceReportExtractor

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Police Report Service",
    description="Microservice for extracting structured fraud indicators from official police report PDFs.",
    version="1.0.0",
)

# Initialize and load model into GPU/CPU memory once at startup
PoliceReportExtractor.save_model()  # Setup Phase
police_model = PoliceReportExtractor.load_model()


@app.post("/extract")
async def extract_police_report(file: UploadFile = File(...)):
  # Preserve original extension or default to .pdf
  ext = os.path.splitext(file.filename)[1] or ".pdf"

  # Save uploaded PDF file temporarily for processing
  with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp.write(await file.read())
    tmp_path = tmp.name

  try:
    # Run model inference
    result = police_model.extract(file_path=tmp_path)
    return result.model_dump() if hasattr(result, "model_dump") else result
  finally:
    # Clean up temporary PDF file from disk
    if os.path.exists(tmp_path):
      os.remove(tmp_path)


@app.get("/health")
async def health_check():
  return {"status": "healthy", "service": "police-report"}


if __name__ == "__main__":
  port = int(os.getenv("POLICE_SERVICE_PORT", 8004))
  ngrok_token = os.getenv("POLICE_NGROK_TOKEN", "").strip()

  if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print(f"\n Police Report Service Live At: {public_url}\n")
  else:
    print(
        f"\n No Ngrok token provided. Running locally at:"
        f" http://127.0.0.1:{port}\n"
    )

  uvicorn.run(app, host="0.0.0.0", port=port)