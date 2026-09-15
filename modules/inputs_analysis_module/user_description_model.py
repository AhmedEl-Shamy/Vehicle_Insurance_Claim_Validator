from transformers import AutoModelForCausalLM, AutoTokenizer
from assets.pydantic_schema.user_description_info_shcema import UserDescriptionInfoSchema


class UserDescriptionModel:

    def save_model(self):
        pass

    def load_model(self):
        pass

    def extract_information(self, user_description: str) -> str:
        pass
