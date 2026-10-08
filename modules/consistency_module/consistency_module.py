import json

from huggingface_hub import InferenceClient
from pydantic import ValidationError
from datetime import time
import re
from json_repair import repair_json
import os
import time

from consistency_schemas import ConsistencySchemas
from consistency_inputs_preprocessor import ConsistencyInputsPreprocessor, PlateNotFoundError


class ConsistencyModule:
    BASE_MODEL_NAME = "openai/gpt-oss-20b"

    def __init__(self, client=None):
        self.client =client

    # ================================================================================
    # Load Model Function
    # ================================================================================
    @classmethod
    def load_model(cls):
        hf_token = os.getenv("CONSISTENCY_HF_TOKEN")
        client = InferenceClient(
            model=cls.BASE_MODEL_NAME,
            provider='groq',
            token=hf_token
        )
        return cls(client=client)

    # ================================================================================
    # Get Prompt Messages Function
    # ================================================================================
    def get_prompt_messages(
            self,
            pipeline_name_a,
            pipeline_name_b,
            comparison_description,
            input_schema,
            input_data,
            additional_instructions=(),
    ):

        system_lines = [
            "You compare the damaged vehicle parts reported by two sources of a vehicle insurance claim.",
            f"The input is a JSON with two lists of part names: {pipeline_name_a}_module_parts and {pipeline_name_b}_module_parts.",
            "A part name may include a side (left or right).",
            "",
            "Fill the tool fields in this order:",
            f"1. matches: pairs of one part from {pipeline_name_a} and one part from {pipeline_name_b} that are the same vehicle part.",
            f'  - Each item in matches is an object with exactly two fields: "{pipeline_name_a}" and "{pipeline_name_b}". Do not rename these fields.',
            "   - Follow these steps to find the matches:",
            f"       - Pick each part in resource {pipeline_name_a}, then search for a part in the other resource {pipeline_name_b} that matches the picked part in meaning and its location in the vehicle.",
            f"       - If you find 2 parts in {pipeline_name_a} and {pipeline_name_b} match, add this pair to matches; if you don't find any matches, leave it empty.",
            "   - Different parts never match, even if both are damaged or close to each other.",
            "   - Make Sure ANY pairs in matches MUST BE EQUIVALENT IN MEANING and reference the EXACTLY SAME PART IN THE SAME LOCATION IN the vehicle",
            f"   - YOU MUST LEAVE matches EMPTY if there are no EQUIVALENT parts between the sources {pipeline_name_a} and {pipeline_name_b}",
            "   - Match only listed parts. Do not infer damage.",
            "   - Match only when sure. A part appears in at most one pair.",
            "   - Copy names exactly as written in the input.",
            "   - If no pair qualifies, return an empty list.",
            f"2. consistent_parts: matched parts whose sides do not conflict. Use the {pipeline_name_b} name.",
            f"3. inconsistent_parts: matched parts where one name says left and the other says right. Use the {pipeline_name_b} name.",
            "   - If either name has no side, the part is consistent.",
            "",
            "Every matched part appears in exactly one list: consistent_parts or inconsistent_parts. Do not list unmatched parts.",
        ]

        if additional_instructions:
            system_lines += ["", "Additional rules for this comparison:"]
            system_lines += [f"- {rule}" for rule in additional_instructions]

        system_lines += ["", "Return the result using the `calculate_consistency_score` tool."]

        return [
            {"role": "system", "content": "\n".join(system_lines)},
            {"role": "user", "content": "\n".join([
                "## Comparison",
                f"{pipeline_name_a} vs {pipeline_name_b}: {comparison_description}",
                "",
                "## Input Data Schema",
                input_schema,
                "",
                "## Input Data",
                input_data,
            ])},
        ]

    # ================================================================================
    # Generate Response Function
    # ================================================================================
    def generate_response(
            self,
            prompt,
            consistency_schema,
            temperature=0.01
    ):
        """
        Send the prompt to the LLM and return the model's tool call as a JSON string.
        Return None if the provider response is malformed."""

        time.sleep(30)

        prepare_consistency_analysis = {
            "type": "function",
            "function": {
                "name": "calculate_consistency_score",
                "description": "Calculate the consistency score.",
                "parameters": consistency_schema.model_json_schema()
            },
        }

        completion = self.client.chat.completions.create(
            messages=prompt,
            tools=[prepare_consistency_analysis],
            tool_choice="required",
            max_tokens=8192,
            temperature=temperature,
            extra_body={"reasoning_effort": "medium"},
        )

        message = completion.choices[0].message

        if message.tool_calls:
            call = message.tool_calls[0]
            args = call.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            return json.dumps(
                {"name": call.function.name, "arguments": args},
                ensure_ascii=False,
            )

        return message.content

    # ================================================================================
    # Validate Matches Function
    # ================================================================================
    def validate_matches(
            self,
            parts_a,
            parts_b,
            matches,
            name_a,
            name_b
    ):
        """
        Return the matches that use a part name not found in the source lists.

        Names are compared in lower case. An empty result means all names are valid.
        Args:
            parts_a, parts_b: Part names of each source.
            matches: Match objects returned by the model.
            name_a, name_b: Field names of the two sources in each match.
        """
        set_a = {p.lower() for p in parts_a}
        set_b = {p.lower() for p in parts_b}

        bad = []
        for m in matches:
            a = getattr(m, name_a).strip().lower()
            b = getattr(m, name_b).strip().lower()
            if a not in set_a or b not in set_b:
                bad.append(m)

        return bad

    # ================================================================================
    # Compute Missing Function
    # ================================================================================
    def compute_missing(self, parts_a, parts_b, matches, name_a, name_b):
        """
        Returns the List of missing parts.
        """

        matched_a = {getattr(m, name_a).strip().lower() for m in matches}
        matched_b = {getattr(m, name_b).strip().lower() for m in matches}

        return (
                [{"part": p, "only_in": name_a}
                 for p in parts_a if p.lower() not in matched_a]
                + [{"part": p, "only_in": name_b}
                   for p in parts_b if p.lower() not in matched_b]
        )

    # ================================================================================
    # Run With Retry Function
    # ================================================================================
    def run_with_retry(
            self,
            messages,
            schema,
            name_a,
            name_b,
            parts_a,
            parts_b,
            max_tries=3,
    ):
        """
        Ask the model for the matches and retry if the names are not in the data.

        Always returns a result: the first valid response, or the last parsed
        response if all tries have invalid names, or an empty result if no
        response could be parsed.
        """
        msgs = list(messages)
        r = schema()
        parsed = False

        for i in range(max_tries):

            if i == 0:
                temp = None
            else:
                temp = 0.3
                print("\nSolving some issues in the response ...")

            raw = self.generate_response(
                prompt=msgs,
                consistency_schema=schema,
                temperature=temp,
            )
            if raw is None:
                continue
            response = repair_json(raw)

            # Validation of Schema and JSON syntax
            try:
                data = json.loads(response)
                r = schema.model_validate(data["arguments"])
                parsed = True
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
                print(f"\nCould not parse the response. Raw: {raw[:500]}\n")
                continue

            bad = self.validate_matches(
                parts_a=parts_a,
                parts_b=parts_b,
                matches=r.matches,
                name_a=name_a,
                name_b=name_b,
            )

            if not bad:
                return r  # Good Response

            bad_names = [
                f"{getattr(m, name_a)} / {getattr(m, name_b)}" for m in bad
            ]

            msgs = list(messages) + [{
                "role": "user",
                "content": (
                        "Some names in matches do not exist in the data: "
                        + "; ".join(bad_names)
                        + f". Use only these names. {name_a}: {parts_a}. {name_b}: {parts_b}."
                ),
            }]

        if parsed:
            print("\nWARNING: returning the last response. Some match names are not in the data.\n")
        else:
            print("\nWARNING: no valid response from the model. Returning an empty result.\n")
        return r

    # ================================================================================
    # Not Comparable Result Function
    # ================================================================================
    def not_comparable_result(self, reason: str) -> dict:
        return {
            "status": "not_comparable",
            "reason": reason,
            "matches": [],
            "consistent_parts": [],
            "inconsistent_parts": [],
            "missings": [],
            "consistency_score": None,
        }

    # ================================================================================
    # Consistency Scores FUNCTIONS
    # ================================================================================

    def calculate_pw_consistency_score(self, result_json, is_car_info_consistent=True) -> float:
        consisteny_parts = len(result_json['consistent_parts'])
        inconsistent_parts = len(result_json['inconsistent_parts'])
        missing_parts = len(result_json['missings'])

        consistency_score = consisteny_parts / (consisteny_parts + inconsistent_parts + 0.25 * missing_parts + 0.00001)

        # if the car infromation are wrong ===> we compare deferent cars ===> consistency = 0
        consistency_score *= int(is_car_info_consistent)

        return consistency_score

    def calculate_overall_consistency_score(self, scores: list) -> float:
        return sum(scores) / len(scores)

    # ================================================================================
    # COMPARISON FUNCTIONS
    # ================================================================================

    def _norm(self, x) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()

    def _is_empty(self, x) -> bool:
        return x is None or self._norm(x) == ""

    def _same_year(self, a, b) -> bool:
        try:
            return int(float(a)) == int(float(b))
        except (TypeError, ValueError):
            return self._norm(a) == self._norm(b)

    def _same_name(self, a, b) -> bool:
        """Same make or model,even if one side has extra words (trim). Example: 'camry' and 'camry se'."""
        ta, tb = set(self._norm(a).split()), set(self._norm(b).split())
        if ta <= tb or tb <= ta:
            return True
        return self._norm(a).replace(" ", "") == self._norm(b).replace(" ", "")  # F-150 و F150

    def car_info_differences(self, car_info_a, car_info_b) -> list:
        """Return the fields that clearly differ. A field that is empty in either source is skipped."""
        checks = (("car_make", self._same_name), ("car_model", self._same_name), ("car_model_year", self._same_year))
        different = []
        for field, same in checks:
            a, b = car_info_a.get(field), car_info_b.get(field)
            if self._is_empty(a) or self._is_empty(b):
                continue
            if not same(a, b):
                different.append(field)
        return different

    def compare_car_info(self, car_info_a, car_info_b) -> bool:
        """Return True if the two sources can describe the same car."""
        return not self.car_info_differences(car_info_a, car_info_b)

    def compare_image_description(self, image_json, description_json) -> dict:

        comparison_description = (
            "image_module_parts lists the vehicle parts where a damage detection model found damage in the claim photos. "
            "user_description_module_parts lists the vehicle parts the claimant reported as damaged in the written description of the accident. "
            "Image part names are written with underscores instead of spaces. "
            "Both lists can include the side of the part (left or right) as a word in the name."
        )

        input_schema = ConsistencySchemas.get_input_schema(
            pipeline_a_name='image',
            pipeline_b_name='user_description'
        )
        input_schema = json.dumps(input_schema.model_json_schema(), ensure_ascii=False)

        user_description_parts, _ = ConsistencyInputsPreprocessor.user_description_preprocessing(description_json)
        image_parts = ConsistencyInputsPreprocessor.image_preprocessing(image_json)

        input_data = ConsistencyInputsPreprocessor.get_preprocessed_model_input(
            pipeline_name_a='image',
            pipeline_name_b='user_description',
            parts_a=image_parts,
            parts_b=user_description_parts
        )

        additional_instructions = [
            "Treat underscores in part names as spaces when you compare names.",
        ]

        prompt = self.get_prompt_messages(
            pipeline_name_a='image',
            pipeline_name_b='user_description',
            comparison_description=comparison_description,
            input_schema=input_schema,
            input_data=input_data,
            additional_instructions=additional_instructions,
        )

        consistency_schema = ConsistencySchemas.make_schema(
            name_a='image',
            name_b='user_description',
            is_car_info_applied=False
        )

        response = self.run_with_retry(
            messages=prompt,
            schema=consistency_schema,
            name_a='image',
            name_b='user_description',
            parts_a=image_parts,
            parts_b=user_description_parts
        )

        missings = self.compute_missing(
            parts_a=image_parts,
            parts_b=user_description_parts,
            matches=response.matches,
            name_a='image',
            name_b='user_description'
        )

        final_response = response.model_dump()
        final_response['missings'] = missings

        consistency_score = self.calculate_pw_consistency_score(final_response)
        final_response['consistency_score'] = round(consistency_score, 3)

        return final_response

    # ================================================================================
    # Image and Description Comparison Function
    # ================================================================================

    def compare_image_invoice(self, image_json, invoice_json) -> dict:
        COMPARISON_DESCRIPTION = (
            "image_module_parts lists the vehicle parts where a damage detection model found damage in the claim photos. "
            "invoice_module_parts lists the vehicle parts repaired or replaced according to the repair invoice. "
            "Image part names are written with underscores instead of spaces, and include the side (left or right) in the name. "
            "Invoice part names are standard part names. The side, when the invoice gives one, is in parentheses."
        )

        ADDITIONAL_INSTRUCTIONS = [
            "Treat underscores in part names as spaces when you compare names.",
        ]

        input_schema = ConsistencySchemas.get_input_schema(
            pipeline_a_name='image',
            pipeline_b_name='invoice',
            pipeline_b_description=(
                "Names of the vehicle parts repaired or replaced according to the repair invoice. "
                "Names are lowercase standard part names. "
                "When the invoice gives a side, it is added at the end in parentheses: (left) or (right). "
                "A name with no parentheses has no side on the invoice."
            )
        )
        input_schema = json.dumps(input_schema.model_json_schema(), ensure_ascii=False)

        invoice_parts, _ = ConsistencyInputsPreprocessor.invoice_preprocessing(invoice_json)
        image_parts = ConsistencyInputsPreprocessor.image_preprocessing(image_json)

        input_data = ConsistencyInputsPreprocessor.get_preprocessed_model_input(
            pipeline_name_a='image',
            pipeline_name_b='invoice',
            parts_a=image_parts,
            parts_b=invoice_parts
        )

        prompt = self.get_prompt_messages(
            pipeline_name_a='image',
            pipeline_name_b='invoice',
            comparison_description=COMPARISON_DESCRIPTION,
            input_schema=input_schema,
            input_data=input_data,
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
        )

        consistency_schema = ConsistencySchemas.make_schema(
            name_a='image',
            name_b='invoice',
            is_car_info_applied=False
        )

        response = self.run_with_retry(
            messages=prompt,
            schema=consistency_schema,
            name_a='image',
            name_b='invoice',
            parts_a=image_parts,
            parts_b=invoice_parts
        )

        missings = self.compute_missing(
            parts_a=image_parts,
            parts_b=invoice_parts,
            matches=response.matches,
            name_a='image',
            name_b='invoice'
        )

        final_response = response.model_dump()
        final_response['missings'] = missings

        consistency_score = self.calculate_pw_consistency_score(final_response)
        final_response['consistency_score'] = round(consistency_score, 3)

        return final_response


    # ================================================================================
    # Image and Police Report Comparison Function
    # ================================================================================
    def compare_image_police_report(self, image_json, police_report_json, plate_number) -> dict:
        name_a, name_b = 'image', 'police_report'

        comparison_description = (
            "Compare the damage found in the claim photos with the damage that the police report records "
            "for the claimant's vehicle."
        )

        additional_instructions = [
            "Treat underscores in part names as spaces when you compare names.",
            "Compare the side word (left or right) in an image name with the side word in a police report name.",
        ]

        input_schema = ConsistencySchemas.get_input_schema(
            pipeline_a_name=name_a,
            pipeline_b_name=name_b,
            pipeline_a_description=(
                "Names of the vehicle parts where the damage detection model found damage in the claim photos. "
                "Names are lowercase. Words in a name are separated by underscores or spaces. "
                "A name can include the position (front or rear) and the side (left or right) as words."
            ),
            pipeline_b_description=(
                "Names of the vehicle parts that the police report records as damaged for the claimant's vehicle. "
                "Names are lowercase free text. "
                "A name can include the position (front or rear) and the side (left or right) as words."
            ),
        )
        input_schema = json.dumps(input_schema.model_json_schema(), ensure_ascii=False)

        image_parts = ConsistencyInputsPreprocessor.image_preprocessing(
            image_json
        )
        police_parts, _ = ConsistencyInputsPreprocessor.police_report_preprocessing(
            police_report_json,
            plate_number
        )

        input_data = ConsistencyInputsPreprocessor.get_preprocessed_model_input(
            pipeline_name_a=name_a, pipeline_name_b=name_b,
            parts_a=image_parts, parts_b=police_parts,
        )

        prompt = self.get_prompt_messages(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            comparison_description=comparison_description,
            input_schema=input_schema,
            input_data=input_data,
            additional_instructions=additional_instructions,
        )

        consistency_schema = ConsistencySchemas.make_schema(
            name_a=name_a,
            name_b=name_b,
            is_car_info_applied=False
        )

        response = self.run_with_retry(
            messages=prompt,
            schema=consistency_schema,
            name_a=name_a,
            name_b=name_b,
            parts_a=image_parts,
            parts_b=police_parts,
        )

        missings = self.compute_missing(
            parts_a=image_parts,
            parts_b=police_parts,
            matches=response.matches,
            name_a=name_a,
            name_b=name_b,
        )

        final_response = response.model_dump()
        final_response['missings'] = missings

        consistency_score = self.calculate_pw_consistency_score(final_response)
        final_response['consistency_score'] = round(consistency_score, 3)

        return final_response

    # ================================================================================
    # User Description and Invoice Comparison Function
    # ================================================================================
    def compare_description_invoice(self, description_json, invoice_json) -> dict:
        name_a, name_b = 'user_description', 'invoice'

        comparison_description = (
            "Compare the damaged parts that the claimant reported in the written description of the accident "
            "with the parts repaired or replaced according to the repair invoice."
        )

        additional_instructions = [
            "Compare the side word (left or right) in a user description name with the side in parentheses in an invoice name.",
        ]

        input_schema = ConsistencySchemas.get_input_schema(
            pipeline_a_name=name_a,
            pipeline_b_name=name_b,
            pipeline_a_description=(
                "Names of the vehicle parts that the claimant reported as damaged in the written description of the accident. "
                "Names are lowercase. The side (left or right) can be a word inside the name."
            ),
            pipeline_b_description=(
                "Names of the vehicle parts repaired or replaced according to the repair invoice. "
                "Names are lowercase standard part names. "
                "When the invoice gives a side, it is added at the end in parentheses: (left) or (right). "
                "A name with no parentheses has no side on the invoice."
            ),
        )
        input_schema = json.dumps(input_schema.model_json_schema(), ensure_ascii=False)

        description_parts, description_car_info = ConsistencyInputsPreprocessor.user_description_preprocessing(
            description_json
        )
        invoice_parts, invoice_car_info = ConsistencyInputsPreprocessor.invoice_preprocessing(
            invoice_json
        )

        input_data = ConsistencyInputsPreprocessor.get_preprocessed_model_input(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            parts_a=description_parts,
            parts_b=invoice_parts,
        )

        prompt = self.get_prompt_messages(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            comparison_description=comparison_description,
            input_schema=input_schema,
            input_data=input_data,
            additional_instructions=additional_instructions,
        )

        consistency_schema = ConsistencySchemas.make_schema(
            name_a=name_a,
            name_b=name_b,
            is_car_info_applied=False
        )

        response = self.run_with_retry(
            messages=prompt,
            schema=consistency_schema,
            name_a=name_a,
            name_b=name_b,
            parts_a=description_parts,
            parts_b=invoice_parts,
        )

        missings = self.compute_missing(
            parts_a=description_parts,
            parts_b=invoice_parts,
            matches=response.matches,
            name_a=name_a,
            name_b=name_b,
        )

        is_car_info_consistent = self.compare_car_info(description_car_info, invoice_car_info)

        final_response = response.model_dump()
        final_response['missings'] = missings
        final_response['is_car_info_consistent'] = is_car_info_consistent

        consistency_score = self.calculate_pw_consistency_score(final_response, is_car_info_consistent)
        final_response['consistency_score'] = round(consistency_score, 3)

        return final_response

    # ================================================================================
    # User Description and Police Report Comparison Function
    # ================================================================================
    def compare_description_police_report(self, description_json, police_report_json, plate_number) -> dict:
        name_a, name_b = 'user_description', 'police_report'

        comparison_description = (
            "Compare the damaged parts that the claimant reported in the written description of the accident "
            "with the damaged parts that the police report records for the claimant's vehicle."
        )

        additional_instructions = [
            "Compare the side word (left or right) in a user description name with the side word in a police report name.",
        ]

        input_schema = ConsistencySchemas.get_input_schema(
            pipeline_a_name=name_a,
            pipeline_b_name=name_b,
            pipeline_a_description=(
                "Names of the vehicle parts that the claimant reported as damaged in the written description of the accident. "
                "Names are lowercase. The side (left or right) can be a word inside the name."
            ),
            pipeline_b_description=(
                "Names of the vehicle parts that the police report records as damaged for the claimant's vehicle. "
                "Names are lowercase free text. The side (left or right) can be a word inside the name."
            ),
        )
        input_schema = json.dumps(input_schema.model_json_schema(), ensure_ascii=False)

        try:
            police_parts, police_report_car_info = ConsistencyInputsPreprocessor.police_report_preprocessing(
                police_report_json,
                plate_number
            )
        except PlateNotFoundError as e:
            return self.not_comparable_result(str(e))

        description_parts, user_description_car_info = ConsistencyInputsPreprocessor.user_description_preprocessing(
            description_json
        )

        input_data = ConsistencyInputsPreprocessor.get_preprocessed_model_input(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            parts_a=description_parts,
            parts_b=police_parts,
        )

        prompt = self.get_prompt_messages(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            comparison_description=comparison_description,
            input_schema=input_schema,
            input_data=input_data,
            additional_instructions=additional_instructions,
        )

        consistency_schema = ConsistencySchemas.make_schema(
            name_a=name_a,
            name_b=name_b,
            is_car_info_applied=False
        )

        response = self.run_with_retry(
            messages=prompt,
            schema=consistency_schema,
            name_a=name_a,
            name_b=name_b,
            parts_a=description_parts,
            parts_b=police_parts,
        )

        missings = self.compute_missing(
            parts_a=description_parts,
            parts_b=police_parts,
            matches=response.matches,
            name_a=name_a,
            name_b=name_b,
        )

        final_response = response.model_dump()
        final_response['missings'] = missings
        is_car_info_consistent = self.compare_car_info(police_report_car_info, user_description_car_info)
        final_response['is_car_info_consistent'] = is_car_info_consistent
        consistency_score = self.calculate_pw_consistency_score(final_response, is_car_info_consistent)
        final_response['consistency_score'] = round(consistency_score, 3)

        return final_response

    # ================================================================================
    # User Description and Police Report Comparison Function
    # ================================================================================
    def compare_invoice_police_report(self, invoice_json, police_report_json, plate_number) -> dict:
        name_a, name_b = 'police_report', 'invoice'

        comparison_description = (
            "police_report_module_parts lists the damaged parts that the police report records for the claimant's vehicle. "
            "invoice_module_parts lists the vehicle parts repaired or replaced according to the repair invoice. "
            "Police report part names are free text. The side, when written, is a word in the name. "
            "Invoice part names are standard part names. The side, when the invoice gives one, is in parentheses."
        )

        additional_instructions = [
            "Compare the side word in a police report name with the side in parentheses in an invoice name.",
        ]

        input_schema = ConsistencySchemas.get_input_schema(
            pipeline_a_name=name_a,
            pipeline_b_name=name_b,
            pipeline_a_description=(
                "Names of the vehicle parts that the police report records as damaged. "
                "Names are lowercase free text. The side (left or right) can be a word inside the name."
            ),
            pipeline_b_description=(
                "Names of the vehicle parts repaired or replaced according to the repair invoice. "
                "Names are lowercase standard part names. "
                "When the invoice gives a side, it is added at the end in parentheses: (left) or (right). "
                "A name with no parentheses has no side on the invoice."
            ),
        )
        input_schema = json.dumps(input_schema.model_json_schema(), ensure_ascii=False)

        police_parts, police_car_info = ConsistencyInputsPreprocessor.police_report_preprocessing(
            police_report_json,
            plate_number
        )
        invoice_parts, invoice_car_info = ConsistencyInputsPreprocessor.invoice_preprocessing(
            invoice_json
        )

        input_data = ConsistencyInputsPreprocessor.get_preprocessed_model_input(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            parts_a=police_parts,
            parts_b=invoice_parts,
        )

        prompt = self.get_prompt_messages(
            pipeline_name_a=name_a,
            pipeline_name_b=name_b,
            comparison_description=comparison_description,
            input_schema=input_schema,
            input_data=input_data,
            additional_instructions=additional_instructions,
        )

        consistency_schema = ConsistencySchemas.make_schema(
            name_a=name_a,
            name_b=name_b,
            is_car_info_applied=False
        )

        response = self.run_with_retry(
            messages=prompt,
            schema=consistency_schema,
            name_a=name_a,
            name_b=name_b,
            parts_a=police_parts,
            parts_b=invoice_parts,
        )

        missings = self.compute_missing(
            parts_a=police_parts,
            parts_b=invoice_parts,
            matches=response.matches,
            name_a=name_a,
            name_b=name_b,
        )

        final_response = response.model_dump()
        is_car_info_consistent = self.compare_car_info(police_car_info, invoice_car_info)

        final_response['missings'] = missings
        final_response['is_car_info_consistent'] = is_car_info_consistent
        consistency_score = self.calculate_pw_consistency_score(final_response, is_car_info_consistent)
        final_response['consistency_score'] = round(consistency_score, 3)

        return final_response

    # ================================================================================
    # Check Consistency Function (MAIN FUNCTION)
    # ================================================================================

    def check_consistency(
            self,
            image_json,
            invoice_json,
            description_json,
            police_report_json,
            plate_number
    ) -> dict:

        consistency_result = {}
        consistency_scores = []

        # Comparison 1 image and invoice
        consistency_result['image_invoice'] = self.compare_image_invoice(
            image_json=image_json,
            invoice_json=invoice_json
        )
        consistency_scores.append(consistency_result['image_invoice']['consistency_score'])

        # Comparison 2 image and description
        consistency_result['image_description'] = self.compare_image_description(
            image_json=image_json,
            description_json=description_json
        )
        consistency_scores.append(consistency_result['image_description']['consistency_score'])

        # Comparison 3 image and police report
        consistency_result['image_police_report'] = self.compare_image_police_report(
            image_json=image_json,
            police_report_json=police_report_json,
            plate_number=plate_number
        )
        consistency_scores.append(consistency_result['image_police_report']['consistency_score'])

        # Comparison 4 description and invoice
        consistency_result['description_invoice'] = self.compare_description_invoice(
            description_json=description_json,
            invoice_json=invoice_json
        )
        consistency_scores.append(consistency_result['description_invoice']['consistency_score'])

        # Comparison 5 Description and police report
        consistency_result['description_police_report'] = self.compare_description_police_report(
            description_json=description_json,
            police_report_json=police_report_json,
            plate_number=plate_number
        )
        consistency_scores.append(consistency_result['description_police_report']['consistency_score'])

        # Comparison 6 invoice and police report
        consistency_result['invoice_police_report'] = self.compare_invoice_police_report(
            invoice_json=invoice_json,
            police_report_json=police_report_json,
            plate_number=plate_number
        )
        consistency_scores.append(consistency_result['invoice_police_report']['consistency_score'])

        # Calculate Overall Consistency Score
        overall_score = self.calculate_overall_consistency_score(consistency_scores)
        consistency_result['overall_consistency_score'] = overall_score

        return consistency_result


