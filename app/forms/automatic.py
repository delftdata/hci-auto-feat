from enum import Enum
from typing import Set
from pydantic import BaseModel, Field, validator

from app.forms.common_fields import BaseTables, MatcherValues, Repositories, TargetVariable


class FeatureDiscovery(BaseModel):
    repositories: Set[Repositories] = Field(
        ...,
        description="Select one or multiple repositories."
        )
    matcher: MatcherValues = Field(
        default=MatcherValues.JACCARD, 
        description="Select one matcher."
        )
    base_table: BaseTables = Field(
        ...,
        description="Select the base table for augmentation."
    )
    target_variable: TargetVariable = Field(
        ...,
        description="Select the target variable.",
        
    )
    non_null_ratio: float = Field(
        default=0.65,
        ge=0,
        le=1,
        description="A number between 0 and 1. 0 means that null values are accepted in any proporion. 1 means that no null value is accepted."
    )
    top_k_features: int = Field(
        default=15,
        description="Maximum number of features to select."
    )
    top_k_join_trees: int = Field(
        default=4,
        description="Maximum number of join trees to return."
    )

    @validator('repositories', allow_reuse=True)
    def check_non_empty_set(cls, value):
        if not value:
            raise ValueError('You must select at least one item.')
        return value