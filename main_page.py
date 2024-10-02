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

    st.session_state.join_trees = trees
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

def print_value(value):
    st.write(value)

if st.session_state.stage == 3:
    st.header("3. Create join trees")

    if st.session_state.tree == False:
        dataset_discovery = st.session_state.dataset_discovery
        target_variable_choices = [dataset.target_column for dataset in dataset_discovery.data_repository]
        form_jt = st.form(key="join_tree_form")
        base_table = form_jt.selectbox(label="Base Table", options=tuple(dataset_discovery.table_repository), index=None)
        target_var = form_jt.selectbox(label="Target Variable", options=tuple(target_variable_choices), index=None)
        non_null_ratio = form_jt.number_input(
            value=0.65,
            label='Non-Null Ratio',
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="A number between 0 and 1. 0 means that null values are accepted in any proporion. 1 means that no null value is accepted."
        )
        top_k_features = form_jt.number_input(
            value=15,
            label='Top-k Features',
            min_value=1,
            step=1,
            help="Maximum number of features to select"
        )
        top_k_join_trees = form_jt.number_input(
            value=5,
            label='Top-k Join Trees',
            min_value=1,
            step=1,
            help="Maximum number of features to select"
        )
            
        submit_button = form_jt.form_submit_button(label='Submit data')
        if submit_button:
            autofeat = HCIAutoFeat(
                base_table=base_table,
                target_variable=target_var,
                non_null_ratio=non_null_ratio,
                top_k_features=top_k_features,
                top_k_join_trees=top_k_join_trees,
                relations=dataset_discovery.relations,
            )

            next_button = st.button("Start process", on_click=compute_join_trees, args=[autofeat])
    else:
        placeholder = st.empty()
        join_trees = st.session_state.join_trees

        # json_data = st.json(join_trees) 
        index = 0
        for tr in join_trees.keys():
            if index > 5:
                break
            index += 1
            tree_node, filename = join_trees[tr] 
            container = st.container(key=f"cont_{index}", border=True)
            a, b, c = container.columns(3)
            a.write(tree_node.rank)                
            b.write(filename)
            clicked = c.button("Click", key=f"btn_{index}", on_click=print_value, args=[index])
            if clicked:
                st.write(index)
        # placeholder.write("Here we will have a list")
        click_tree_button = st.button("Click on a join tree", on_click=set_state, args=[4])


if st.session_state.stage == 4:
    st.header("4. Update and Evaluate")