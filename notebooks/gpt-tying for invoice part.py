!pip install pymupdf
!pip install json_repair
import os
import re
import time
import json
import io
import base64
import pymupdf
from PIL import Image
from typing import List, Optional, Literal, Union, get_args
from pydantic import BaseModel, Field, field_validator, model_validator
from openai import OpenAI
from json_repair import repair_json

# ============================================================
# 1. SHARED SYSTEM TAXONOMY 
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
    "Fender", "Quarter Panel", "Front Door", "Rear Door", "Door", "Door Handle", "Door Jamb", "Side Mirror",
    "Windshield", "Side Window", "Rear Window", "Windshield Header", "Rear Window Header",
    "Rear Window Ledge", "Roof", "Roof Rail", "Roof Header", "Roof Rack", "Headliner",
    "Rocker Panel", "Molding", "Weatherstrip", "A-Pillar", "B-Pillar", "C-Pillar", "D-Pillar", "Other Pillar",
    "Trunk Lid", "Rear Hatch", "Tailgate", "Spoiler", "Body Trim", "Running Board",
    "Truck Cab", "Pickup Bed", "Bedside Panel", "Bed Rail", "Vehicle Canopy",
    "Wheel", "Tire", "Rim", "Wheel Cover", "Wheel Hub", "Wheel Well", "Axle",
    "Suspension", "Control Arm", "Differential", "Drivetrain", "Frame", "Frame Rail",
    "Crossmember", "Radiator", "Radiator Support", "Radiator Mounting Bracket",
    "Skid Plate", "Underbody", "Seat", "License Plate", "Trailer Hitch", "Park Sensor", "Sensor Cap", "Radar", "Other"
]

OperationType = Literal["Replace", "Repair", "Refinish", "Blend", "Not Specified", "Exclude"]

ALLOWED_CAR_PARTS = set(get_args(CarPart))

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
    operation: OperationType = Field(..., description="Action code: Replace, Repair, Refinish, or Blend.")

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
        elif any(x in v_clean for x in ["refn", "refinish", "paint"]):
            return "Refinish"
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
        section_headers = {
            "rear door", "front door", "quarter panel", "rear lamps", 
            "pillars", "rear bumper", "pillars, rocker & floor"
        }
        blacklisted_phrases = [
            "clear coat", "pre and post scan", "pre-repair scan", "post-repair scan",
            "hazardous waste", "flex additive", "paint supplies", "shop supplies",
            "labor", "calibrate", "undercoat", "deadner", "zip tie", "seal kit"
        ]
        seen_descriptions = set()

        for part in self.repaired_parts:
            desc_clean = part.original_description.strip().lower()

            if part.operation == "Exclude":
                continue
            if desc_clean in section_headers:
                continue
            if any(phrase in desc_clean for phrase in blacklisted_phrases):
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
You are an Advanced Spatial OCR & Document Parsing Engine for Automotive & Mechanical Invoices/Estimates.

YOUR GOAL: Extract vehicle meta information and a list of physical replaced, repaired, or refinished parts/components from ANY invoice format (Collision Estimates, Standard Invoices, Mechanical Service Sheets).

---

### STEP 1: VEHICLE INFORMATION EXTRACTION (car_info)
Locate vehicle details from the header, document metadata, or job description:
- Extract: car_make, car_model, car_model_year (4 digits), vehicle_type, body_class.
- Map vehicle_type and body_class strictly to the standard allowed Enums.
- Examples in documents: "Toyota Camry Sedan (2004)", "2020 ACURA MDX AWD", "BMW 3 SERIES TOURING (E91)", "BMW motorcycle".

---

### STEP 2: DYNAMIC TABLE PARSING (MATCH ONE OF 3 PATTERNS)

PATTERN A: Collision Estimate Table (Has explicit 'Oper' / 'Operation' column)
- Applicable when: The table contains columns like [Line] [Oper] [Description] ...
- Operation rules from 'Oper' column:
  * 'Repl' / 'Repi' -> "Replace"
  * 'Rpr' / 'Repair' -> "Repair"
  * 'Blnd' / 'Blend' / 'Bind' -> "Blend"
  * 'Refn' / 'Paint' -> "Refinish"
  * 'R&I', 'R/I', 'RBI', 'RAI', 'Subl', 'Setup', '*' -> DISCARD ROW.

PATTERN B: Single Standard Invoice Table (No 'Oper' column, mixed items)
- Applicable when: The table lists items under [Item Name] / [Description] without an 'Oper' column.
- Operation rules:
  * Inspect the text for operation keywords (e.g., "(Replacement)", "Repair", "Paint").
  * If a physical part is listed without an operation keyword, set operation to "Replace" by default.

PATTERN C: Multi-Section Service Sheet (Separated 'Parts' and 'Labor' sections)
- Applicable when: The invoice has separate blocks for "Parts" and "Labor".
- Operation rules:
  * EXTRACT ONLY from the "Parts" table block. Completely IGNORE the "Labor" table block.
  * Set operation to "Replace" for all physical components/filters/parts listed in the Parts section.

---

### STEP 3: STRICT EXCLUSION & FILTERING RULES
MUST DISCARD/EXCLUDE ALL OF THE FOLLOWING NON-PARTS:
1. Labor Lines: "Installation Labor", "Paint Labor", "Body Labor", "Unibody Pull", "Remove/Install", "Reset Electronics", or lines inside Labor tables.
2. Materials & Fees: "Hazardous Waste", "Paint Supplies", "Clear Coat / Add for Clear Coat", "Chemicals and Cleaners", "Shop Supplies", "Over Spray Masking".
3. Scans & Calibrations: "Pre/Post Scan", "Recalibrate Lane Changer", "Wheel Alignment".
4. Category Headers: "REAR DOOR", "FRONT DOOR", "QUARTER PANEL", "REAR LAMPS", "PILLARS", "REAR BUMPER", "MISCELLANEOUS", "PARTS", "LABOR".
- DISCARD ROUTINE MAINTENANCE & CONSUMABLE ITEMS:
* Filters: Oil filter, air filter, cabin filter, fuel filter.
* Ignition & Small Wear-and-Tear Parts: Spark plugs, shims, O-rings, washers, gaskets, drain plug seals.
* Fluids & Chemicals: Engine oil, brake fluid, coolant, hydraulic fluid, cleaners, degreasers.
* Routine Wear Items: Wiper blades, light bulbs, brake pads/rotors (if listed as routine service).
---

### STEP 4: CAR PART TAXONOMY MAPPING
Map every extracted physical item to the closest system `CarPart` Enum (e.g., "Front Bumper Cover" -> "Front Bumper", "Fender Liner" -> "Wheel Well" or "Fender", "Rocker Molding" -> "Rocker Panel", "Oil Filter" / "Spark Plug" -> "Other").

Return ONLY valid raw JSON matching {SCHEMA_STR}. Do not include any extra introductory text.
"""

# ============================================================
# 4. MODEL CLASS (gpt-6-luna)
# ============================================================
class InvoiceModel:
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL", None)
    client: Optional[OpenAI] = None

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-6-luna", base_url: Optional[str] = None):
        if api_key:
            InvoiceModel.api_key = api_key
        if base_url:
            InvoiceModel.base_url = base_url

        self.model_name = model_name
        self.prompt = SMART_INVOICE_PROMPT

        if InvoiceModel.api_key and not InvoiceModel.client:
            InvoiceModel.load_models()

    @classmethod
    def load_models(cls, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if api_key:
            cls.api_key = api_key
        if base_url:
            cls.base_url = base_url
        if not cls.api_key:
            cls.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")

        if cls.api_key:
            kwargs = {"api_key": cls.api_key}
            if cls.base_url:
                kwargs["base_url"] = cls.base_url
            cls.client = OpenAI(**kwargs)

    @staticmethod
    def save_models():
        pass

    @staticmethod
    def load_document_as_images(file_path: str, dpi: int = 200) -> List[Image.Image]:
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
            repaired_str = repair_json(text)
            return json.loads(repaired_str)
        except Exception:
            pass

        cleaned = re.sub(r',\s*([\}\]])', r'\1', text)
        cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
        return json.loads(repair_json(cleaned))

    def extract_information(self, invoice: Union[str, List[Image.Image]], dpi: int = 200, max_retries: int = 5) -> str:
        if not self.client:
            self.load_models()
            if not self.client:
                raise ValueError("OpenAI/OpenRouter Client is not initialized. Please provide a valid API key.")

        if isinstance(invoice, str):
            images = self.load_document_as_images(invoice, dpi=dpi)
        elif isinstance(invoice, list):
            images = invoice
        else:
            raise ValueError("Input must be either a file path (str) or a list of PIL Images.")

        if not images:
            raise ValueError("No document images provided for processing.")

        
        user_content = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })

        user_content.append({
            "type": "text",
            "text": self.prompt
        })

        print(f" RUNNING {self.model_name.upper()} VISION INFERENCE...")

        delay = 4
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": user_content}
                    ],
                    temperature=1.0
                )

                output_text = response.choices[0].message.content
                raw_json = self._extract_json(output_text)
                validated_output = CleanInvoiceExtraction(**raw_json)

                return json.dumps(validated_output.model_dump(), indent=2, ensure_ascii=False)

            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg:
                    print(f"⚠️ سيرفر {self.model_name} مشغول (503/429). محاولة {attempt}/{max_retries} بعد {delay} ثوانٍ...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e

        raise RuntimeError(f"تعذر استخراج البيانات عبر {self.model_name} بسبب استمرار الضغط العالي على السيرفر.")


# ============================================================
# HOW TO USE
# ============================================================
if __name__ == "__main__":
    
    API_KEY = ""
    file_path = "/kaggle/input/datasets/nataliemmoris/test-5/Toyota_Camry_Invoice.pdf"

    
    model = InvoiceModel(
        api_key=API_KEY, 
        model_name="gpt-6-luna",
        # base_url="https://openrouter.ai/api/v1"  
    )

    if os.path.exists(file_path):
        result_str: str = model.extract_information(invoice=file_path)
        print("\n EXTRACTION SUCCESSFUL:")
        print(result_str)