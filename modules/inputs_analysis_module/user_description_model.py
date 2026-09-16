from transformers import AutoModelForCausalLM, AutoTokenizer
from assets.pydantic_schema.user_description_info_shcema import UserDescriptionInfoSchema
import re

class UserDescriptionModel:
    LOCAL_MODEL_PATH = "assets/ai_models/qwen2.5-1.5B"

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def save_model(model_name: str="Qwen/Qwen2.5-1.5B-Instruct"):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(UserDescriptionModel.LOCAL_MODEL_PATH)
        tokenizer.save_pretrained(UserDescriptionModel.LOCAL_MODEL_PATH)

    @classmethod
    def load_model(cls):
        model = AutoModelForCausalLM.from_pretrained(cls.LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(cls.LOCAL_MODEL_PATH)
        return cls(model=model, tokenizer=tokenizer)

    def getPromptMessages(accidentDescription: str):
        return [
            {
                "role": "system",
                "content": '\n'.join([
                    "You are an NLP data extraction assistant.",
                    "You will be provided with an accident description.",
                    "Extract the required information only from the provided accident description.",
                    "Do not guess or invent missing information.",
                    "You must use the `extract_accident_details` tool to return the extracted accident information.",
                ])
            },
            {
                "role": "user",
                "content": '\n'.join([
                    "## Accident Description",
                    accidentDescription,
                ])
            }
        ]

    def prepareChatTemplate(self, description: str):
        data_extraction_tool = {
            "type": "function",
            "function": {
                "name": "extract_accident_details",
                "description": "Extract structured accident details from the accident description.",
                "parameters": UserDescriptionInfoSchema.model_json_schema(),
            }
        }

        inputs = self.tokenizer.apply_chat_template(
            self.getPromptMessages(accidentDescription=description),
            tools=[data_extraction_tool],
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self.model.device)

        return inputs

    def generateResponse (self, description: str):
        inputs = self.prepareChatTemplate(description=description)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        return response

    def extract_response(self, response: str):
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.DOTALL)
        if not match:
            raise ValueError("No <tool_call> block found in response")

        json_str = match.group(1)
        return json_str

    def extract_information(self, description: str) -> str:
        response = self.generateResponse(description=description)
        json_str = self.extract_response(response)
        return json_str
