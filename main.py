from fastapi import FastAPI
from pydantic import BaseModel
import json
from modules.inputs_analysis_module.user_description_model import UserDescriptionModel

app = FastAPI()

class UserInputsSchema(BaseModel):
    description: str

@app.get("/")
async def root():
    return {"message": "Welcome to Vehicle Insurance Claim Validator"}

# For Testing Only
@app.post("/extract_description")
async def extract_description(inputs: UserInputsSchema):
    inputs_dict = inputs.model_dump()
    model = UserDescriptionModel.load_model()
    json_str = model.extract_information(description=inputs_dict["description"])
    response = json.loads(json_str)
    return response