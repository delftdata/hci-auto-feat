from pathlib import Path
from typing import List, Optional

import pandas as pd

from autotda.config import DATA_FOLDER


CLASSIFICATION = "binary"
REGRESSION = "regression"


class Dataset:
    def __init__(self, base_table_path: Path, base_table_name: str, base_table_label: str, target_column: str,
                 dataset_type: bool, base_table_features: Optional[List] = None):
        self.base_table_path = base_table_path
        self.target_column = target_column
        self.base_table_name = base_table_name
        self.base_table_id = base_table_path / base_table_name
        self.base_table_label = base_table_label
        self.base_table_features = base_table_features
        self.base_table_df = None

        if dataset_type == "regression":
            self.dataset_type = REGRESSION
        else:
            self.dataset_type = CLASSIFICATION

    def set_features(self):
        if self.base_table_df is not None:
            self.base_table_features = list(self.base_table_df.drop(columns=[self.target_column]).columns)
        else:
            self.base_table_features = list(
                pd.read_csv(self.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                            escapechar='\\', nrows=1).drop(columns=[self.target_column]).columns)

    def set_base_table_df(self):
        self.base_table_df = pd.read_csv(self.base_table_id, header=0, engine="python", encoding="utf8", quotechar='"',
                                         escapechar='\\')


CLASSIFICATION_DATASETS = []
REGRESSION_DATASETS = []

def init_datasets():
    print("Initialising datasets ...")
    datasets_df = pd.read_csv(DATA_FOLDER / "datasets.csv")

    for index, row in datasets_df.iterrows():
        dataset = Dataset(base_table_label=row["base_table_label"],
                          target_column=row["target_column"],
                          base_table_path=Path(row["base_table_path"]),
                          base_table_name=row["base_table_name"],
                          dataset_type=row["dataset_type"])
        if row["dataset_type"] == REGRESSION:
            REGRESSION_DATASETS.append(dataset)
        else:
            CLASSIFICATION_DATASETS.append(dataset)

    return CLASSIFICATION_DATASETS + REGRESSION_DATASETS

ALL_DATASETS = init_datasets()

def filter_datasets(dataset_labels: Optional[List[str]] = None, problem_type: Optional[str] = None) -> List[Dataset]:
    # `is None` is missing on purpose, because typer cannot return None default values for lists, only []
    if problem_type == CLASSIFICATION:
        return CLASSIFICATION_DATASETS if not dataset_labels else [dataset for dataset in CLASSIFICATION_DATASETS if
                                                                   dataset.base_table_label in dataset_labels]
    if problem_type == REGRESSION:
        return REGRESSION_DATASETS if not dataset_labels else [dataset for dataset in REGRESSION_DATASETS if
                                                               dataset.base_table_label in dataset_labels]
    if not problem_type and dataset_labels:
        return [dataset for dataset in ALL_DATASETS if dataset.base_table_label in dataset_labels]

    return ALL_DATASETS


    