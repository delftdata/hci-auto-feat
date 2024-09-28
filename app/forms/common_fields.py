from enum import Enum
from src.autotda.data_models.dataset_model import ALL_DATASETS


def get_valid_input_repositories() -> tuple[str, ...]:
    dataset_labels = [dataset.base_table_label for dataset in ALL_DATASETS]
    return dataset_labels

def get_valid_input_base_table() -> tuple[str, ...]:
    return "foo", "bar"

def get_valid_input_target_variable() -> tuple[str, ...]:
    return "foo", "bar"

Repositories = Enum(
    "Repositories",
    ((value, value) for value in get_valid_input_repositories()),
    type=str,
)

BaseTables = Enum(
    "Base Tables",
    ((value, value) for value in get_valid_input_base_table()),
    type=str,
)

TargetVariable = Enum(
    "Target Variable",
    ((value, value) for value in get_valid_input_target_variable()),
    type=str,
)

class MatcherValues(str, Enum):
    JACCARD = "Jaccard"
    COMA = "Coma" 