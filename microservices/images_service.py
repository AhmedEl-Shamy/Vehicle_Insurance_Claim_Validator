import os
import tempfile
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from pyngrok import ngrok
import uvicorn

# Import shared model from models directory
from modules.inputs_analysis_module.image_detection_segmentation import ImageDetectionSegmentation

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Vision / Image Processing Service",
    description="Microservice for multi-image fraud detection and object segmentation.",
    version="1.0.0",
)

# Load model into GPU memory once upon server startup
image_model = ImageDetectionSegmentation.load_model()


@app.post("/extract")
async def extract_images(files: List[UploadFile] = File(...)):
  combined_result = None

  for file in files:
    # Preserve original extension or default to .jpg
    ext = os.path.splitext(file.filename)[1] or ".jpg"

    # Save uploaded file temporarily for inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
      tmp.write(await file.read())
      tmp_path = tmp.name

    try:
      # Run inference on single image
      result = image_model.extract_information(image_path=tmp_path)

      if combined_result is None:
        # Initialize response structure with top-level metadata from first image
        combined_result = {
            "status": result.get("status", "SUCCESS"),
            "image": result.get("image", file.filename),
            "annotated_image": result.get("annotated_image", ""),
            "json_file": result.get("json_file", ""),
            "json_validation": result.get("json_validation", {}),
            "results": list(result.get("results", [])),
        }
      else:
        # Append detections from remaining images into the main results array
        if "results" in result and isinstance(result["results"], list):
          combined_result["results"].extend(result["results"])

    finally:
      # Clean up temporary file from disk
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  return combined_result or {}


@app.get("/health")
async def health_check():
  return {"status": "healthy", "service": "image-processing"}


if __name__ == "__main__":
  port = int(os.getenv("IMAGE_SERVICE_PORT", 8002))
  ngrok_token = os.getenv("IMAGE_NGROK_TOKEN", "").strip()

  if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print(f"\n Image Service Live At: {public_url}\n")
  else:
    print(
        f"\n No Ngrok token provided. Running locally at:"
        f" http://127.0.0.1:{port}\n"
    )

  uvicorn.run(app, host="0.0.0.0", port=port)