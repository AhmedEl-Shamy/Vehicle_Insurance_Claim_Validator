from pydantic import  BaseModel, Field
from typing import List, Literal
from datetime import time

VehicleType = Literal[
    "PASSENGER CAR",
    "TRUCK",
    "MULTIPURPOSE PASSENGER VEHICLE (MPV)",
    "BUS",
    "INCOMPLETE VEHICLE",
    "OTHER"
]

BodyClass = Literal[
    "Cargo Van",
    "Convertible/Cabriolet",
    "Coupe",
    "Crossover Utility Vehicle (CUV)",
    "Hatchback/Liftback/Notchback",
    "Incomplete",
    "Incomplete - Chassis Cab (Number of Cab Unknown)",
    "Incomplete - Chassis Cab (Single Cab)",
    "Incomplete - Cutaway",
    "Incomplete - Motor Home Chassis",
    "Incomplete - Stripped Chassis",
    "Minivan",
    "Pickup",
    "Roadster",
    "Sedan/Saloon",
    "Sport Utility Truck (SUT)",
    "Sport Utility Vehicle (SUV)/Multi-Purpose Vehicle (MPV)",
    "Truck",
    "Van",
    "Wagon",
    "Other"
]

CarPart = Literal[
    "Bumper",
    "Front Bumper",
    "Rear Bumper",
    "Bumper Reinforcement",
    "Bumper Support",
    "Bumper Fascia",
    "Lower Valance",

    "Hood",
    "Grille",
    "Headlight",
    "Taillight",

    "Fender",
    "Quarter Panel",
    "Door",
    "Door Handle",
    "Door Jamb",
    "Side Mirror",

    "Windshield",
    "Side Window",
    "Rear Window",
    "Windshield Header",
    "Rear Window Header",
    "Rear Window Ledge",

    "Roof",
    "Roof Rail",
    "Roof Header",
    "Roof Rack",
    "Headliner",

    "Rocker Panel",

    "A-Pillar",
    "B-Pillar",
    "C-Pillar",
    "D-Pillar",
    "Other Pillar",

    "Trunk Lid",
    "Rear Hatch",
    "Tailgate",
    "Spoiler",

    "Body Trim",
    "Running Board",

    "Truck Cab",
    "Pickup Bed",
    "Bedside Panel",
    "Bed Rail",
    "Vehicle Canopy",

    "Wheel",
    "Tire",
    "Rim",
    "Wheel Cover",
    "Wheel Hub",
    "Wheel Well",
    "Axle",

    "Suspension",
    "Control Arm",
    "Differential",
    "Drivetrain",

    "Frame",
    "Frame Rail",
    "Crossmember",

    "Radiator",
    "Radiator Support",
    "Radiator Mounting Bracket",

    "Skid Plate",
    "Underbody",

    "Seat",
    "License Plate",
    "Trailer Hitch",

    "Other"
]

WeekDay =  Literal[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
]

Month = Literal[
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
]




class CarInfo(BaseModel):
    car_make: str = Field(
        ...,
        min_length=2,
        description= (
            "The manufacturer or brand of the car described in the accident statement, "
            "such as Ford, Toyota, Hyundai, or Chevrolet."
        )
    )

    car_model: str = Field(
        ...,
        min_length=2,
        description= (
            "The model name of the car described in the accident statement, "
            "such as Fusion, Camry, Accent, or Equinox."
        )
    )

    car_model_year: int = Field(
        ...,
        le=2022,
        description= (
            "The model year of the car as stated in the description. "
            "Do not confuse it with the year of the accident."
        )
    )

    vehicle_type: VehicleType = Field(
        ...,
        min_length=2,
        description= (
            "The general type of the car, such as Passenger Car, Truck, "
            "Multipurpose Passenger Vehicle (MPV), Bus, or another allowed vehicle type."
        )
    )

    body_class: BodyClass = Field(
        ...,
        min_length=2,
        description= (
            "The specific body style or class of the car, such as Sedan/Saloon, "
            "Pickup, SUV, CUV, Coupe, Minivan, Wagon, or another allowed body class."
        )
    )


class CarDamage(BaseModel):
    car_damage_description: str = Field(
        ...,
        description=(
            "A detailed summary of the physical damage described in the accident statement. "
            "Include the affected areas or sides of the car, impact or crush locations, "
            "pillar damage, and separate damage areas when mentioned. Preserve useful approximate "
            "location details, but do not invent damage that is not supported by the description."
        )
    )

    damage_parts: List[CarPart] = Field(
        ...,
        min_length=1,
        description=(
            "A normalized list of the car parts that are stated or clearly supported as damaged. "
            "Select only from the allowed CarPart values, include all supported damaged parts, "
            "and do not add duplicates. Do not mark a part as damaged when it is mentioned only "
            "as a reference for describing or measuring the location of other damage."
        )
    )

class AccidentDateTime (BaseModel):
    day_of_week: WeekDay = Field(..., description="The day of the week when the accident occurred.")
    month: Month= Field(..., description="The month when the accident occurred.")
    year: int= Field(..., le=2022, ge=2017, description="The year when the accident occurred.")
    acc_time: time= Field(..., description="The approximate time when the accident occurred.")

class UserDescriptionInfoSchema(BaseModel):
    car_info: CarInfo = Field(
        ...,
        description=(
            "Structured information about the car extracted from the accident description, "
            "including its make, model, model year, vehicle type, and body class."
        )
    )

    car_damage: CarDamage = Field(
        ...,
        description=(
            "Structured information about the physical damage extracted from the accident description, "
            "including a detailed damage summary and the normalized damaged car parts."
        )
    )

    accident_date_time: AccidentDateTime= Field(
        ...,
        description=(
            "Structured time information extracted from the accident description, "
            "including the day of the week, month, year, and approximate accident time when available."
        )
    )
