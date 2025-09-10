
from pydantic import BaseModel, field_validator
from typing import ClassVar, List


class RawInputModel(BaseModel):
    """
    Pydantic model to validate raw user input for income prediction.
    """

    age: int
    workclass: str
    education: str
    years_of_education: int
    marital_status: str
    occupation: str
    ethinicity: str
    gender: str
    capital_gain: int
    capital_loss: int
    working_hours_per_week: int
    origin_country: str

    _allowed_workclass: ClassVar[List[str]] = [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ]
    _allowed_gender: ClassVar[List[str]] = ["Male", "Female"]

    # ---- Validators ----

    @field_validator("age", "years_of_education", "capital_gain", "capital_loss", "working_hours_per_week", mode="before")
    @classmethod
    def cast_to_int(cls, v, info):
        """Ensure numeric fields are integers."""
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"{info.field_name} should be an int, got {v} ({type(v).__name__})")

    @field_validator("workclass", "education", "marital_status", "occupation", "ethinicity", "gender", "origin_country", mode="before")
    @classmethod
    def strip_str(cls, v, info):
        """Ensure string fields are stripped of leading/trailing whitespace."""
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} should be a string, got {v} ({type(v).__name__})")
        return v.strip()

    @field_validator("workclass")
    @classmethod
    def validate_workclass(cls, v):
        if v not in cls._allowed_workclass:
            raise ValueError(f"Invalid workclass '{v}'. Allowed: {cls._allowed_workclass}")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in cls._allowed_gender:
            raise ValueError(f"Invalid gender '{v}'. Allowed: {cls._allowed_gender}")
        return v
