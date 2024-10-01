from enum import Enum
from typing import Set
from polars import List
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel, Field
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

from app.forms.automatic import FeatureDiscovery
from app.forms.human_in_the_loop import HILProcess
from app.graph.graph_model import edge_styles, from_relations_to_graph, node_styles
from autotda.data_models.dataset_model import ALL_DATASETS, Dataset
from autotda.functions.tree_functions import HCIAutoFeat
from src.autotda.functions.relationship_functions import DatasetDiscovery, Relation


# st.set_page_config(layout="wide")

hil_process = "Human-in-the-loop"
auto_process = "Automatic"
process_types = [hil_process, auto_process]

BASE_TABLE_VALUES = []
TARGET_VARS = []

if 'stage' not in st.session_state:
    st.session_state.stage = 0

if "tree" not in st.session_state:
    st.session_state.tree = False

if "dataset_discovery" not in st.session_state:
    st.session_state.dataset_discovery = None
    
if "join_trees" not in st.session_state:
    st.session_state.join_trees = None

def set_state(i):
    st.session_state.stage = i

def set_tree_state(state: bool):
    st.session_state.tree = state

def find_relations(state: int, dataset_discovery: DatasetDiscovery):
    set_state(state)
    dataset_discovery.find_relationships()
    st.session_state.dataset_discovery = dataset_discovery

def compute_join_trees(autofeat: HCIAutoFeat):
    autofeat.streaming_feature_selection(queue={autofeat.base_table})

    trees = dict(sorted(autofeat.join_tree_maping.items(), key=lambda item: item[1][0].rank, reverse=True))

    st.session_state.join_trees = trees[:autofeat.top_k_join_trees]
    set_tree_state(True)
    
    
if st.session_state.stage == 0:
    st.header("Saved Process")

    st.write("Welcome to your saved feature discovery process")
    st.write("Empty process collections")

    btn_start = st.button("Start process", on_click=set_state, args=[1])

if st.session_state.stage == 1:
    st.header("1. Input Data")
    # Radio boxes with options for choosing the process: default = HIL 
    process_type = st.radio("Type of process", process_types, index=0)
    
    if process_type == hil_process:
        data = sp.pydantic_form(
            key="hil_process", 
            model=HILProcess, 
            submit_label="Submit data",
            # clear_on_submit=True,
            )
        
        if data:
            selected_repos = [repo.value for repo in list(data.repositories)]
            selected_matcher = data.matcher.value
            dataset_discovery = DatasetDiscovery(matcher=selected_matcher, data_repositories=selected_repos)
            BASE_TABLE_VALUES = dataset_discovery.table_repository
            TARGET_VARS = [dataset.target_column for dataset in dataset_discovery.data_repository]
           
            next_button = st.button("Find relations", on_click=find_relations, args=[2, dataset_discovery])

    else:
        data = sp.pydantic_form(
            key="auto_process", 
            model=FeatureDiscovery, 
            submit_label="Submit data",
            clear_on_submit=True,
            )
        
        if data:
            json_data = st.json(data) 
            next_button = st.button("Find relations", on_click=set_state, args=[2])


if st.session_state.stage == 2:
    st.header("2. Find relations")
    form = st.form(key='similarity_th_form')
    similarity_score = form.number_input(
        value=0.65,
        label='Similarity threshold',
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="The similarity threshold will filter the nodes.")
    submit_button = form.form_submit_button(label='Update graph')

    dataset_discovery = st.session_state.dataset_discovery
    if submit_button:
        dataset_discovery.set_similarity_threshold(similarity_score)
        dataset_discovery.find_relationships()
        st.session_state.dataset_discovery = dataset_discovery

    nodes, edges = from_relations_to_graph(dataset_discovery.relations)
    elements = {
        "nodes": nodes,
        "edges": edges
    }
        
    st_link_analysis(elements, "cose", node_styles, edge_styles)    

    create_tree_button = st.button("Create join trees", on_click=set_state, args=[3])


# def get_valid_input_base_table() -> tuple[str, ...]:
#     dataset_discovery = st.session_state.dataset_discovery
#     return dataset_discovery.table_repository

# def get_valid_input_target_variable() -> tuple[str, ...]:
#     dataset_discovery = st.session_state.dataset_discovery
#     return [dataset.target_column for dataset in dataset_discovery.data_repository]





if st.session_state.stage == 3:
    BaseTables = Enum(
        "Base Tables",
        ((value, value) for value in BASE_TABLE_VALUES),
        type=str,
    )

    TargetVariable = Enum(
        "Target Variable",
        ((value, value) for value in TARGET_VARS),
        type=str,
    )
    class JoinTreeData(BaseModel):
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



    st.header("3. Create join trees")

    if st.session_state.tree == False:
        data = sp.pydantic_form(
                key="join_tree_process", 
                model=JoinTreeData, 
                submit_label="Submit data",
                )
            
        if data:
            dataset_discovery = st.session_state.dataset_discovery
            autofeat = HCIAutoFeat(
                base_table=data.base_table.value,
                target_variable=data.target_variable.value,
                non_null_ratio=data.non_null_ratio.value,
                top_k_features=data.top_k_features.value,
                top_k_join_trees=data.top_k_join_trees.value,
                relations=dataset_discovery.relations,
            )
            next_button = st.button("Start process", on_click=compute_join_trees, args=[autofeat])
    else:
        placeholder = st.empty()
        join_trees = st.session_state.join_trees

        json_data = st.json(join_trees) 

        # for tr in join_trees.keys():
        #     st.write(tr)
        #     tree_node, filename = join_trees[tr]
        #     print(f"\t\t{tree_node.rank}")
        #     print(filename)
        # placeholder.write("Here we will have a list")
        click_tree_button = st.button("Click on a join tree", on_click=set_state, args=[4])


if st.session_state.stage == 4:
    st.header("4. Update and Evaluate")