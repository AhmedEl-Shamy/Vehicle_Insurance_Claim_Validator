import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn

# Import shared model class from the models folder
from modules.inputs_analysis_module.user_description_model import UserDescriptionModel

# Load environment variables
load_dotenv()

app = FastAPI(title="User Description Service", version="1.0.0")

# Initialize and load model
UserDescriptionModel.save_model()
user_description_model = UserDescriptionModel.load_model()


class DescriptionRequest(BaseModel):
  description: str


@app.post("/extract")
async def extract_description(payload: DescriptionRequest):
  result = user_description_model.extract_information(
      description=payload.description
  )
  return result.model_dump() if hasattr(result, "model_dump") else result


if __name__ == "__main__":
  port = int(os.getenv("DESCRIPTION_SERVICE_PORT", 8001))
  ngrok_token = os.getenv("DESCRIPTION_NGROK_TOKEN", "").strip()

  if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print(f"\n Description Service Live At: {public_url}\n")
  else:
    print(f"\n Running locally at: http://127.0.0.1:{port}\n")

  uvicorn.run(app, host="0.0.0.0", port=port)