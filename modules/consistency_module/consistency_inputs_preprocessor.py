import json
import re


class PlateNotFoundError(ValueError):
    """The plate number is empty or not in the police report."""

class ConsistencyInputsPreprocessor:
    @staticmethod
    def get_clear_parts(parts):
        IGNORED_PARTS = ["unknown", "", "n/a", "none"]
        clear_parts = []

        for part in parts:
            part = part.lower().strip()
            if (part not in clear_parts) and (part not in IGNORED_PARTS):
                clear_parts.append(part)

        return clear_parts

    @staticmethod
    def user_description_preprocessing(input_description_json: str):
        "Get damaged parts list and car infromation from User Description JSON"

        data = json.loads(input_description_json)['arguments']

        parts = data['car_damage']['damage_parts']
        clear_parts = ConsistencyInputsPreprocessor.get_clear_parts(parts)

        car_info = data['car_info']

        return clear_parts, car_info

    @staticmethod
    def image_preprocessing(image_json: str) -> list:
        "Get clear damage parts from image JSON"

        d = json.loads(image_json)['results']
        parts = []

        for item in d:
            part = item['affected_part']
            parts.append(part)

        clear_parts = ConsistencyInputsPreprocessor.get_clear_parts(parts)
        return clear_parts

    @staticmethod
    def get_side(text: str, use_corner_codes: bool = False, driver_side: str = None):
        """
        Return
        "left" or "right" from the text, or None if there is no side or both sides appear.
        Handles words (left, right), short forms (LT, RT, LH, RH, lft, rgt) and
        written forms like L/H, R.H. and L-H.
        Optional: corner codes (LF, RF, LR, RR) and driver/passenger words.
        driver_side: "left" or "right". Set it only if the driver side is the same for all cars.
        """

        LEFT_WORDS = {"left", "lt", "lh", "lft", "lhs"}
        RIGHT_WORDS = {"right", "rt", "rh", "rgt", "rght", "rhs"}
        CORNER_CODES = {"lf": "left", "lr": "left", "rf": "right", "rr": "right"}
        DRIVER_WORDS = {"driver", "drvr", "drv"}
        PASSENGER_WORDS = {"passenger", "psgr", "pssngr"}

        t = text.lower()
        t = re.sub(r"\bl[\s./-]+h\b", " left ", t)  # L/H  L.H.  L-H
        t = re.sub(r"\br[\s./-]+h\b", " right ", t)  # R/H  R.H.  R-H
        tokens = set(re.split(r"[^a-z]+", t))

        is_left = bool(tokens & LEFT_WORDS)
        is_right = bool(tokens & RIGHT_WORDS)

        if use_corner_codes:
            is_left = is_left or any(CORNER_CODES.get(w) == "left" for w in tokens)
            is_right = is_right or any(CORNER_CODES.get(w) == "right" for w in tokens)

        if driver_side in ("left", "right"):
            passenger_side = "right" if driver_side == "left" else "left"
            for words, side in ((DRIVER_WORDS, driver_side), (PASSENGER_WORDS, passenger_side)):
                if tokens & words:
                    if side == "left":
                        is_left = True
                    else:
                        is_right = True

        if is_left == is_right:
            return None
        return "left" if is_left else "right"

    @staticmethod
    def normalize_plate(plate) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(plate).upper())

    @staticmethod
    def strip_part_location(name: str) -> str:
        """Remove a location detail at the end of the part name, like 'lower corner' or 'upper edge'."""
        name = name.strip().lower()
        return re.sub(r"\s+(lower|upper)\s+(corner|edge)$", "", name) or name

    @staticmethod
    def police_report_preprocessing(report_json, plate_number):
        """Get the clean damaged parts list and the car info of the vehicle with the given plate."""
        data = json.loads(report_json) if isinstance(report_json, str) else report_json

        wanted = ConsistencyInputsPreprocessor.normalize_plate(plate_number) if plate_number else ""
        if not wanted:
            raise PlateNotFoundError("The plate number is empty.")

        for v in data.get("vehicles", []):
            info = v["vehicle_info"]
            if ConsistencyInputsPreprocessor.normalize_plate(info.get("plate_number", "")) != wanted:
                continue

            parts = ConsistencyInputsPreprocessor.get_clear_parts(
                [ConsistencyInputsPreprocessor.strip_part_location(p) for p in info.get("damaged_parts", [])]
            )
            car_info = {
                "car_make": info.get("make"),
                "car_model": info.get("model"),
                "car_model_year": info.get("year"),
            }
            return parts, car_info

        raise PlateNotFoundError("No vehicle in the police report matches the given plate number.")

    @staticmethod
    def invoice_preprocessing(invoice_json: str, ignore_operations=()):
        """Get the clean repaired parts list (with the side if written) and the car info from the invoice JSON."""
        data = json.loads(invoice_json)

        parts = []
        for p in data['repaired_parts']:
            if p.get('operation') in ignore_operations:
                continue
            orig = p.get('original_description', '').strip()
            name = p['standard_part']
            if name == 'Other':
                parts.append(orig)
                continue
            side = ConsistencyInputsPreprocessor.get_side(orig)
            parts.append(f"{name} ({side})" if side else name)

        return ConsistencyInputsPreprocessor.get_clear_parts(parts), data['car_info']

    @staticmethod
    def get_preprocessed_model_input(
            pipeline_name_a: str,
            pipeline_name_b: str,
            parts_a: list,
            parts_b: list
    ) -> str:

        data = {
            f'{pipeline_name_a}_module_parts': parts_a,
            f'{pipeline_name_b}_module_parts': parts_b,
        }

        return json.dumps(data, ensure_ascii=False)