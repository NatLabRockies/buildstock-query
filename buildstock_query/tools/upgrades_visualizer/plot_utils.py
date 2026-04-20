from pydantic import ConfigDict, BaseModel, Field
from enum import Enum
from typing import Optional
import re


class SavingsTypes(str, Enum):
    absolute = "Absolute"
    savings = "Savings"
    percent_savings = "Percent Savings"


class ValueTypes(str, Enum):
    total = "total"
    count = "count"
    mean = "mean"
    distribution = "distribution"
    scatter = "scatter"
    sorted = "Sorted"


class PlotParams(BaseModel):
    enduses: list[str]
    savings_type: SavingsTypes
    change_type: Optional[str] = None
    upgrade: Optional[str] = None
    group_by: list[str] = Field(default_factory=list)
    filter_bldgs: list[int] = Field(default_factory=list)
    sync_upgrade: Optional[int] = None
    resolution: str = "annual"
    value_type: ValueTypes

    model_config = ConfigDict(coerce_numbers_to_str=True)


num2month = {1: "January", 2: "February", 3: "March", 4: "April",
             5: "May", 6: "June", 7: "July", 8: "August",
             9: "September", 10: "October", 11: "November", 12: "December"}
month2numstr = {v.lower(): f"{k:02}" for k, v in num2month.items()}


def human_sort_key(input_str):
    """
    Transforms the element of a list into a list of strings and numbers to allow for human sorting
    eg. regular string sort: EF 19.3, EF 21.9, EF 6.7
        human sort:  EF 6.7, EF 19.3, EF 21.9
    Useful to sort alphanumeric strings
    """
    input_str = str(input_str).lower()
    input_str = [
        int(x) if x and x[0] in "0123456789" else month2numstr.get(x.lower(), x) if x else None
        for x in re.split(r"([\<\-])|([0-9]+)", input_str)
    ]
    return ["" if x is None else x for x in input_str]


def flatten_list(list_obj):
    return [item for sublist in list_obj for item in sublist]
