from pydantic import Field, create_model
from typing import List

class ConsistencySchemas:
    @staticmethod
    def make_schema(name_a: str, name_b: str, is_car_info_applied=True):
        PartMatch = create_model(
            "PartMatch",
            **{
                name_a: (str, Field(
                    description=f"Part name exactly as written in the {name_a} data and it must has an EQUIVELENT MEANING at least one part in {name_b}.")),
                name_b: (str, Field(
                    description=f"Part name exactly as written in the {name_b} data and it must has an EQUIVELENT MEANING at least one part in {name_a}.")),
            },
        )
        if is_car_info_applied:
            Result = create_model(
                "ConsistencyResult",
                matches=(
                    List[PartMatch],
                    Field(
                        default_factory=list,
                        description=(
                            f"Pairs of parts that are the same vehicle part in {name_a} and {name_b}. "
                            "Leave this list empty if no pair is the same vehicle part."
                        ),
                    ),
                ),
                consistent_parts=(
                    List[str],
                    Field(
                        default_factory=list,
                        description=(
                            f"Matched parts with no contradiction. Use the {name_b} name. "
                            "A part must appear in only one list."
                        ),
                    ),
                ),
                inconsistent_parts=(
                    List[str],
                    Field(
                        default_factory=list,
                        description=(
                            "Matched parts that explicitly contradict, for example left side vs right side. "
                            f"Use the {name_b} name. Usually empty. A part must appear in only one list."
                        ),
                    ),
                ),
                is_car_info_consistent=(
                    bool,
                    Field(
                        default=True,
                        description=(
                            "Compare make, model, model year, vehicle type and body class. "
                            "Set to false if any of them differs."
                        ),
                    ),
                ),
            )

        else:
            Result = create_model(
                "ConsistencyResult",
                matches=(
                    List[PartMatch],
                    Field(
                        default_factory=list,
                        description=f"Pairs of parts with the same meaning in {name_a} and {name_b}.",
                    ),
                ),
                consistent_parts=(
                    List[str],
                    Field(
                        default_factory=list,
                        description=(
                            f"Matched parts with no contradiction. Use the {name_b} name. "
                            "A part must appear in only one list."
                        ),
                    ),
                ),
                inconsistent_parts=(
                    List[str],
                    Field(
                        default_factory=list,
                        description=(
                            "Matched parts that explicitly contradict, for example left side vs right side. "
                            f"Use the {name_b} name. Usually empty. A part must appear in only one list."
                        ),
                    ),
                ),
            )

        return Result

    @staticmethod
    def get_input_schema(
            pipeline_a_name,
            pipeline_b_name,
            pipeline_a_description='',
            pipeline_b_description=''
    ):
        """ Create the dynamic schema for Consistency Layer."""

        if not pipeline_a_description:
            pipeline_a_description = f"List of vehicle damaged parts detected in {pipeline_a_name} analysis module.\n{pipeline_a_description}"

        if not pipeline_b_description:
            pipeline_b_description = f"List of vehicle damaged parts detected in {pipeline_b_name} analysis module.\n{pipeline_b_description}"

        schema_fields = {
            f"{pipeline_a_name}_module_parts": (
                List[str],
                Field(default_factory=list, description=pipeline_a_description)
            ),
            f"{pipeline_b_name}_module_parts": (
                List[str],
                Field(default_factory=list, description=pipeline_b_description)
            ),
        }

        schema = create_model("ConsistencyInputSchema", **schema_fields)

        return schema