import os
import re
import json
import pymupdf
from PIL import Image
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from google import genai
from google.genai import types
from json_repair import repair_json

# ============================================================
# 1. SHARED SYSTEM TAXONOMY (NO CHANGES)
# ============================================================
VehicleType = Literal[
    "PASSENGER CAR", "TRUCK", "MULTIPURPOSE PASSENGER VEHICLE (MPV)",
    "BUS", "INCOMPLETE VEHICLE", "OTHER"
]

BodyClass = Literal[
    "Cargo Van", "Convertible/Cabriolet", "Coupe", "Crossover Utility Vehicle (CUV)",
    "Hatchback/Liftback/Notchback", "Incomplete", "Incomplete - Chassis Cab (Number of Cab Unknown)",
    "Incomplete - Chassis Cab (Single Cab)", "Incomplete - Cutaway", "Incomplete - Motor Home Chassis",
    "Incomplete - Stripped Chassis", "Minivan", "Pickup", "Roadster", "Sedan/Saloon",
    "Sport Utility Truck (SUT)", "Sport Utility Vehicle (SUV)/Multi-Purpose Vehicle (MPV)",
    "Truck", "Van", "Wagon", "Other"
]

CarPart = Literal[
    "Bumper", "Front Bumper", "Rear Bumper", "Bumper Reinforcement", "Bumper Support",
    "Bumper Fascia", "Lower Valance", "Hood", "Grille", "Headlight", "Taillight",
    "Fender", "Quarter Panel","Front Door","Rear Door", "Door", "Door Handle", "Door Jamb", "Side Mirror",
    "Windshield", "Side Window", "Rear Window", "Windshield Header", "Rear Window Header",
    "Rear Window Ledge", "Roof", "Roof Rail", "Roof Header", "Roof Rack", "Headliner",
    "Rocker Panel", "Molding", "Weatherstrip", "A-Pillar", "B-Pillar", "C-Pillar", "D-Pillar", "Other Pillar",
    "Trunk Lid", "Rear Hatch", "Tailgate", "Spoiler", "Body Trim", "Running Board",
    "Truck Cab", "Pickup Bed", "Bedside Panel", "Bed Rail", "Vehicle Canopy",
    "Wheel", "Tire", "Rim", "Wheel Cover", "Wheel Hub", "Wheel Well", "Axle",
    "Suspension", "Control Arm", "Differential", "Drivetrain", "Frame", "Frame Rail",
    "Crossmember", "Radiator", "Radiator Support", "Radiator Mounting Bracket",
    "Skid Plate", "Underbody", "Seat", "License Plate", "Trailer Hitch","Park Sensor", "Sensor Cap", "Radar", "Other"
]

OperationType = Literal["Replace", "Repair","Refinish", "Blend", "Not Specified", "Exclude"]

ALLOWED_CAR_PARTS = set(CarPart.__args__)

# ============================================================
# 2. SCHEMAS 
# ============================================================
class ExtractedCarInfo(BaseModel):
    car_make: Optional[str] = Field(None, description="Vehicle Make (e.g. Toyota, Dodge).")
    car_model: Optional[str] = Field(None, description="Vehicle Model (e.g. Camry, Ram).")
    car_model_year: Optional[int] = Field(None, description="4-digit Model Year.")
    vehicle_type: VehicleType = Field("PASSENGER CAR", description="Mapped Vehicle Type enum.")
    body_class: BodyClass = Field("Sedan/Saloon", description="Mapped Body Class enum.")

class ExtractedRepairedPart(BaseModel):
    original_description: str = Field(..., description="EXACT line item text printed on the invoice.")
    standard_part: CarPart = Field(..., description="Mapped part corresponding directly to system CarPart taxonomy.")
    operation: OperationType = Field(..., description="Action code: Replace, Repair, or Blend.")

    @field_validator('operation', mode='before')
    @classmethod
    def parse_and_validate_operation(cls, v: str) -> str:
        v_clean = str(v).strip().lower()
        if any(x in v_clean for x in ["r&i", "r/i", "rbi", "rai", "remove"]):
            return "Exclude"
        if any(x in v_clean for x in ["repl", "replace", "repi", "rep/l"]):
            return "Replace"
        elif any(x in v_clean for x in ["blnd", "blend", "bind"]):
            return "Blend"
        elif any(x in v_clean for x in ["rpr", "repair"]):
            return "Repair"
        else:
            return "Exclude"

    @field_validator('standard_part', mode='before')
    @classmethod
    def auto_fix_part_typos(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            return "Other"
        
        v_clean = v.strip()
        typo_map = {
            "tailight": "Taillight", "tail light": "Taillight",
            "head light": "Headlight", "front bumper": "Front Bumper",
            "rear bumper": "Rear Bumper", "door shell": "Door",
            "rocker panel": "Rocker Panel", "quarter panel": "Quarter Panel",
            "rocker molding": "Rocker Panel", "stone guard": "Body Trim",
            "door w/strip": "Door Jamb", "belt molding": "Body Trim",

            "auto park sensor": "Park Sensor", 
            "park sensor": "Park Sensor", 
            "reverse sensor": "Park Sensor",
            "reverse sensor cap": "Sensor Cap",
            "sensor cap": "Sensor Cap"
        }
        
        if v_clean.lower() in typo_map:
            return typo_map[v_clean.lower()]
            
        formatted = v_clean.title()
        if formatted in ALLOWED_CAR_PARTS:
            return formatted
        return "Other"

class CleanInvoiceExtraction(BaseModel):
    car_info: ExtractedCarInfo = Field(..., description="Header vehicle meta information extracted from the invoice.")
    repaired_parts: List[ExtractedRepairedPart] = Field(default_factory=list, description="List of physical repaired/replaced/blended parts.")

    @model_validator(mode='after')
    def strict_labor_and_supplies_filter(self):
        filtered_list = []
        section_headers = ["rear door", "front door", "quarter panel", "rear lamps", "pillars", "rear bumper", "pillars, rocker & floor"]
        blacklisted_terms = [
            "coat", "spray", "mask", "sand", "reset", "scan", "waste", "hazardous",
            "calibrate", "labor", "buff", "undercoat", "deadner", "oil", "fluid",
            "washer", "cleaner", "coolant", "spark plug", "grease", "chemical",
            "zip tie", "filter", "shim", "seal kit", "syn", "synthetic", "drum",
            "75w", "5w40", "dot4", "r&i", "r/i", "subl", "pre and post scan", "clear coat"
        ]
        seen_descriptions = set()

        for part in self.repaired_parts:
            desc_clean = part.original_description.strip().lower()

            if part.operation == "Exclude":
                continue
            if desc_clean in section_headers:
                continue
            if any(term in desc_clean for term in blacklisted_terms):
                continue
            if desc_clean in seen_descriptions:
                continue
            
            seen_descriptions.add(desc_clean)
            filtered_list.append(part)
            
        self.repaired_parts = filtered_list
        return self

# ============================================================
# 3. PROMPT GENERATION 
# ============================================================
SCHEMA_STR = json.dumps(CleanInvoiceExtraction.model_json_schema(), indent=2)

SMART_INVOICE_PROMPT = f"""
You are a Spatial OCR Document Engine for Automotive Estimates.

CRITICAL EXECUTION LOGIC (STRICT STEP-BY-STEP AUDITING):

STEP 1: IDENTIFY TABLE COLUMNS
Locate the table header: [Line] [Oper] [Description] ...
The 'Oper' column is strictly positioned between [Line] and [Description].

STEP 2: ROW-BY-ROW COLUMN AUDIT (DO NOT SKIP ANY CHARACTERS):
For EVERY numbered row, extract the EXACT raw text inside the 'Oper' column bounds.
Apply the following strict filters:

A. EMPTY/BLANK Oper Cell:
   - If the cell under 'Oper' is empty, space, or missing (e.g., "O/H bumper assy"): DISCARD ROW IMMEDIATELY. Do not invent an operation.

B. ALLOWED OPER VALUES (Exact Match):
   - 'Repl' or 'Repi' -> Map operation to "Replace"
   - 'Rpr' or 'Repair' -> Map operation to "Repair"
   - 'Blnd' or 'Bind' -> Map operation to "Blend"
   - 'Refn' or 'Paint' -> Map operation to "Refinish"

C. STRICT DISCARD VALUES:
   - 'R&I', 'R/I', 'RBI', 'RAI', 'Refn', 'Subl' -> DISCARD ROW IMMEDIATELY.

STEP 3: CATEGORY & DESCRIPTION CLEANUP
- Ignore section header rows (e.g., REAR BUMPER, VEHICLE DIAGNOSTICS).
- Ignore non-physical items/scans/fees (Pre-repair scan, Post-repair scan, Clear coat).
-- Extract physical sensors (e.g., Reverse sensor, Park sensor, Radar) as valid parts under 'Park Sensor' or 'Sensor Cap'.
Return ONLY valid raw JSON matching {SCHEMA_STR}. Do not include any extra introductory text.
- Do NOT discard body parts that are being refinished or blended (e.g. "Blend rt. roof rail" is VALID).
- IGNORE paint materials/supplies and fees (e.g., Pre-repair scan, Post-repair scan, Clear coat, Flex additive, Hazardous waste).
- Extract physical sensors (e.g., Reverse sensor, Park sensor, Radar) as valid parts under 'Park Sensor' or 'Sensor Cap'.
Return ONLY valid raw JSON matching {SCHEMA_STR}. Do not include any extra introductory text.
"""

# ============================================================
# 4. REQUIRED MODEL CLASS STRUCTURE
# ============================================================
class InvoiceModel:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.prompt = SMART_INVOICE_PROMPT
        self.client = None
        
        if self.api_key:
            self.load_models()

    def load_models(self):
        # Load your models from this path "assets/ai_models/{your_model}"
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def save_models(self):
        # Save your models to this path "assets/ai_models/{your_model}"
        pass

    def load_document_as_images(self, file_path: str, dpi: int = 200) -> List[Image.Image]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at path: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        images = []

        print("=" * 70)
        print(f"PROCESSING DOCUMENT: {os.path.basename(file_path)}")
        print("=" * 70)

        if ext == ".pdf":
            doc = pymupdf.open(file_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                print(f"  --> Extracted Page {page_idx + 1}")
            doc.close()
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            print("  --> Image Loaded Successfully")
        else:
            raise ValueError("Unsupported file extension.")
            
        return images

    @staticmethod
    def _extract_json(raw_text: str) -> dict:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
        text = match.group(1).strip() if match else raw_text.strip()
        
        try:
            return json.loads(text)
        except Exception:
            pass

        try:
            repaired_string = repair_json(text)
            return json.loads(repaired_string)
        except Exception:
            cleaned = re.sub(r',\s*([\}\]])', r'\1', text)
            cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
            return json.loads(cleaned)

    def extract_information(self, invoice: Union[str, List[Image.Image]], dpi: int = 200) -> str:
        if not self.client:
            self.load_models()
            if not self.client:
                raise ValueError("Gemini Client is not initialized. Please provide a valid API key.")

        
        if isinstance(invoice, str):
            images = self.load_document_as_images(invoice, dpi=dpi)
        elif isinstance(invoice, list):
            images = invoice
        else:
            raise ValueError("Input must be either a file path (str) or a list of PIL Images.")

        if not images:
            raise ValueError("No document images provided for processing.")

        contents = list(images)
        contents.append(self.prompt)

        print(f" RUNNING {self.model_name.upper()} VISION INFERENCE...")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )

        output_text = response.text
        raw_json = self._extract_json(output_text)
        validated_output = CleanInvoiceExtraction(**raw_json)
        
        
        return json.dumps(validated_output.model_dump(), indent=2, ensure_ascii=False)


# # ============================================================
# # HOW TO USE
# # ============================================================
# if __name__ == "__main__":
#     API_KEY = "YOUR_GEMINI_API_KEY"
#     API_KEY = "AQ.Ab8RN6J0Dphy_bRfVE1MSuosATvxv9WUqa94N-rZ9AYcSlYTyQ"
#     model = InvoiceModel(api_key=API_KEY)
#     file_path = "path/to/invoice.jpg"
    
#     if os.path.exists(file_path):
#         result_str: str = model.extract_information(invoice=file_path)
#         print("\n EXTRACTION SUCCESSFUL:")
#         print(result_str)
