import json
import re
from typing import List, Optional
import torch
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from jsonrepair import repair_json


# Pydantic Schema
class VehicleDetails(BaseModel):
    make: Optional[str] = Field(None, description="Car manufacturer")
    model: Optional[str] = Field(None, description="Car model")
    year: Optional[int] = Field(None, description="Car manufacturing year")
    plate_number: Optional[str] = Field(None, description="Plate number or VIN")
    damaged_parts: List[str] = Field(
        default_factory=list, 
        description="List of specific damaged components (e.g. ['front bumper', 'window glass'])"
    )
    damaged_area: List[str] = Field(
        default_factory=list, 
        description="List of general areas of damage (e.g. ['driver side', 'front end'])"
    )
    driver_details: Optional[str] = Field(None, description="Full name of the vehicle driver without commas")


class Vehicle(BaseModel):
    vehicle_info: VehicleDetails


class PoliceReport(BaseModel):
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    vehicles: List[Vehicle] = Field(default_factory=list)


# Main Extractor Class
class PoliceReportExtractor:
    def __init__(self, 
        model=None, processor=None, 
        model_path: str = "assets/ai_models/Qwen2.5-VL-7B-Instruct"):
    
        self.model_path = model_path
        self.model = model
        self.processor = processor

    @staticmethod
    def save_model(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", model_path: str = "assets/ai_models/Qwen2.5-VL-7B-Instruct") -> None:
        """
        Downloading the model and saving a copy to a specific local folder.
        """
        print(f"Downloading model and processor weights for: {model_id}...")
        
        # Download from Hugging Face
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"Saving model to local directory: {model_path}...")
        # Saving explicitly to the local folder
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        print("Download and local save complete.")

    @classmethod
    def load_model(cls, model_path: str = "assets/ai_models/Qwen2.5-VL-7B-Instruct") -> None:
        """
        Function 2: Loads the 4-bit quantized model and processor into VRAM from the specified model_path.
        """
        print(f"Loading from {model_path} into memory with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
        loaded_processor = AutoProcessor.from_pretrained(model_path)
        print("Model and processor successfully loaded.")

        return cls(model= loaded_model, processor= loaded_processor)

    def extract_and_validate(self, raw: str) -> PoliceReport:
        """
        Post-processing logic to strip markdown fences, repair JSON structure, 
        and validate outputs against Pydantic schema.
        """
        # Strip markdown fences
        clean = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.IGNORECASE)
        clean = re.sub(r'\s*```$', '', clean.strip())

        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in output:\n{raw}")

        json_str = match.group(0)

        # Robust JSON parsing with repair_json fallback -->
        try:
            raw_dict = json.loads(json_str)
        except Exception:
            # If standard parsing fails, repair syntax errors dynamically
            repaired_str = repair_json(json_str)
            raw_dict = json.loads(repaired_str)

        # Fallback fixes if the model still nests location or string arrays
        if isinstance(raw_dict.get("location"), dict):
            loc = raw_dict["location"]
            raw_dict["location"] = ", ".join(str(v) for v in loc.values() if v)

        for veh in raw_dict.get("vehicles", []):
            info = veh.get("vehicle_info", {})
            if isinstance(info, dict):
                if isinstance(info.get("damaged_parts"), str):
                    info["damaged_parts"] = [p.strip() for p in info["damaged_parts"].split(",") if p.strip()]
                if isinstance(info.get("damaged_area"), str):
                    info["damaged_area"] = [a.strip() for a in info["damaged_area"].split(",") if a.strip()]

        return PoliceReport(**raw_dict)

    def extract(self, file_path: str, dpi: int = 100) -> PoliceReport:
        """
        Function 3: Runs inference on a target PDF report (supports single or multi-page documents).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded! Call `load_model()` before running inference.")

        # Convert all PDF pages into PIL images
        images = convert_from_path(file_path, dpi=dpi)

        # Build vision content block dynamically for all converted pages
        user_content = []
        for img in images:
            user_content.append({
                "type": "image",
                "image": img,
                "max_pixels": 501760
            })

        user_content.append({
            "type": "text",
            "text": (
                "Extract the accident information from this police report. "
                "Ensure drivers are correctly matched to their respective vehicles."
            )
        })

        schema_json = json.dumps(PoliceReport.model_json_schema(), indent=2)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise data extraction assistant. "
                    "Your only job is to read a police accident report and return a single valid JSON object "
                    "that strictly adheres to the provided JSON Schema.\n\n"
                    f"JSON SCHEMA:\n{schema_json}\n\n"
                    "RULES:\n"
                    "1. Output ONLY valid JSON — no markdown fences, no explanation, no preamble.\n"
                    "2. Ensure data types match the schema strictly (damaged_parts and damaged_area MUST be arrays/lists of strings).\n"
                    "3. Do not include officer name, badge number, supervisor, charges, or report status.\n"
                    "4. Match the driver's name to the correct vehicle.\n"
                    "5. Driver name shouldn't include commas\n"
                )
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        # Model Inference Execution
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1536)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse and return validated Pydantic object
        return self.extract_and_validate(output_text)


