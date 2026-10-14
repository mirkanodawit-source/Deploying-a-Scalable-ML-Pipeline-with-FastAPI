"""Validated Census input with aliases for the original hyphenated names."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "age": 39, "workclass": "State-gov", "fnlgt": 77516,
    "education": "Bachelors", "education-num": 13,
    "marital-status": "Never-married", "occupation": "Adm-clerical",
    "relationship": "Not-in-family", "race": "White", "sex": "Male",
    "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
    "native-country": "United-States",
}


class CensusRecord(BaseModel):
    """One person; categorical values unseen during training are supported."""

    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, populate_by_name=True,
        json_schema_extra={"examples": [EXAMPLE]},
    )
    age: int = Field(ge=17, le=120, strict=True)
    workclass: str = Field(min_length=1, max_length=100)
    fnlgt: int = Field(ge=0, strict=True)
    education: str = Field(min_length=1, max_length=100)
    education_num: int = Field(alias="education-num", ge=1, le=16, strict=True)
    marital_status: str = Field(alias="marital-status", min_length=1, max_length=100)
    occupation: str = Field(min_length=1, max_length=100)
    relationship: str = Field(min_length=1, max_length=100)
    race: str = Field(min_length=1, max_length=100)
    sex: str = Field(min_length=1, max_length=100)
    capital_gain: int = Field(alias="capital-gain", ge=0, strict=True)
    capital_loss: int = Field(alias="capital-loss", ge=0, strict=True)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=168, strict=True)
    native_country: str = Field(alias="native-country", min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: Literal["<=50K", ">50K"]
