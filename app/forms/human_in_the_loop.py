from enum import Enum
from typing import Set
from pydantic import BaseModel, Field, validator

from app.forms.common_fields import MatcherValues, Repositories


class HILProcess(BaseModel):
    repositories: Set[Repositories] = Field(
        ...,
        description="Select one or multiple repositories."
        )
    matcher: MatcherValues = Field(
        default=MatcherValues.JACCARD, 
        description="Select one matcher."
        )
    
    @validator('repositories', allow_reuse=True)
    def check_non_empty_set(cls, value):
        if not value:
            raise ValueError('You must select at least one item.')
        return value