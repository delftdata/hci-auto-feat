from enum import Enum
import streamlit as st
from polars import List
# from main_page import BASE_TABLE_VALUES, TARGET_VARS
from src.autotda.data_models.dataset_model import ALL_DATASETS    


Repositories = Enum(
    "Repositories",
    ((value, value) for value in [dataset.base_table_label for dataset in ALL_DATASETS]),
    type=str,
)

class MatcherValues(str, Enum):
    JACCARD = "Jaccard"
    COMA = "Coma" 