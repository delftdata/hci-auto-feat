from enum import Enum
from typing import Set
from polars import List
import streamlit as st
import streamlit_pydantic as sp
import polars as pl
from pydantic import BaseModel, Field
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

from app.forms.automatic import FeatureDiscovery
from app.forms.human_in_the_loop import HILProcess
from app.forms.join_trees import DisplayJoinTree, print_join_trees
from app.graph.graph_model import edge_styles, from_relations_to_graph, node_styles
from autotda.data_models.dataset_model import ALL_DATASETS, Dataset
from autotda.functions.evaluation_functions import evaluate_join_tree
from autotda.functions.tree_functions import HCIAutoFeat
from src.autotda.functions.relationship_functions import DatasetDiscovery, Relation


# st.set_page_config(layout="wide")

hil_process = "Human-in-the-loop"
auto_process = "Automatic"
process_types = [hil_process, auto_process]

node_styles_list = [
    NodeStyle("TABLE", "#FF7F3E", "alias"),
]

if 'stage' not in st.session_state:
    st.session_state.stage = 0

if "tree" not in st.session_state:
    st.session_state.tree = False

if "dataset_discovery" not in st.session_state:
    st.session_state.dataset_discovery = None
    
if "join_trees" not in st.session_state:
    st.session_state.join_trees = None

if "top_k_join_trees" not in st.session_state:
    st.session_state.top_k_join_trees = None

if "selected_tree" not in st.session_state:
    st.session_state.selected_tree = None

if "target_variable" not in st.session_state:
    st.session_state.target_variable = None

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

    # for k, v in trees.items():
    #     print(k)
    #     print(v[0].rank)
    st.session_state.join_trees = trees
    set_tree_state(True)
    
    
if st.session_state.stage == 0:
    st.title("Saved Process")

    st.write("Welcome to your saved feature discovery process")
    st.write("Empty process collections")

    btn_start = st.button("Start process", on_click=set_state, args=[1])

if st.session_state.stage == 1:
    st.title("1. Input Data")
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
    st.title("2. Find relations")
    
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


def prepare_next_state(display_trees, value, state_num):
    st.session_state.selected_tree = display_trees[value]
    set_state(state_num)

if st.session_state.stage == 3:
    st.title("3. Create join trees")

    if st.session_state.tree == False:
        dataset_discovery = st.session_state.dataset_discovery
        target_variable_choices = [dataset.target_column for dataset in dataset_discovery.data_repository]
        form_jt = st.form(key="join_tree_form")
        base_table = form_jt.selectbox(label="Base Table", options=tuple(dataset_discovery.table_repository), index=None)
        target_var = form_jt.selectbox(label="Target Variable", options=tuple(target_variable_choices), index=None)
        st.session_state.target_variable = target_var
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
        
        st.session_state.top_k_join_trees = top_k_join_trees
            
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

        top_k_join_trees = st.session_state.top_k_join_trees
        display_trees = print_join_trees(join_trees=join_trees, top_k_trees=top_k_join_trees)

        for i, tr in enumerate(display_trees):
            container = st.container(key=f"cont_{i}", border=True)
            rank_col, graph_col, btn_col = container.columns([.4, 2, .3], vertical_alignment="center")

            rank_col.write("Score:")
            rank_col.write("{:.3f}".format(tr.rank)) 
            with graph_col:
                st_link_analysis(tr.elements, "breadthfirst", node_styles_list, edge_styles, height=150) 

            # table_col.dataframe(tr.table_data, width=250, height=150)

            clicked = btn_col.button(" ", key=f"btn_{i}", icon=":material/arrow_forward_ios:", on_click=prepare_next_state, args=[display_trees, i, 4])


class MLModels(Enum):
    GBM = "LightGBM" 
    XGB = "XGBoost"
    RF = "Random Forest" 
    XT = "Extremely Randomized Trees"
    KNN = "k-Nearest Neighbors"
    LR = "Linear Regression"

    
if st.session_state.stage == 4:
    st.title("4. View and Evaluate")

    display_tree: DisplayJoinTree = st.session_state.selected_tree

    container = st.container(key=f"cont_selected_tree", border=True)
    rank_col, graph_col = container.columns([.4, 2], vertical_alignment="center")

    rank_col.write("Score:")
    rank_col.write("{:.3f}".format(display_tree.rank)) 
    with graph_col:
        st_link_analysis(display_tree.elements, "breadthfirst", node_styles_list, edge_styles, height=150) 

    bottom_cont = container.container(key="cont_bottom")

    tab1, tab2, tab3, tab4 = bottom_cont.tabs(["Selected Features", "Discarded Features", "Augmented Table", "Evaluation"])

    with tab1:
        st.subheader("Selected Features")
        event = st.dataframe(
            display_tree.selected_features,
            key="selected_features",
            on_select="rerun",
            column_config={
                "column_name": "Feature Name",
                "redundancy_score": "Redundancy Score",
                "relevance_score": "Relevance Score"
            }
        )

    with tab2:
        st.subheader("Discarded Features")
        event = st.dataframe(
            display_tree.discarded_features,
            key="discarded_features",
            on_select="rerun",
            column_config={
                "column_name": "Feature Name",
                "redundancy_score": "Redundancy Score",
                "relevance_score": "Relevance Score"
            }
        )
    with tab3:
        st.subheader("Augmented Table")

        join_trees = st.session_state.join_trees
        selected_tree, _ = join_trees[display_tree.join_tree_id]

        columns_to_drop = set(selected_tree.features).intersection(set(selected_tree.join_keys))
        materialised_join = pl.read_parquet(display_tree.join_tree_filename).select(selected_tree.features).drop(columns_to_drop)
        
        st.dataframe(materialised_join)

    with tab4:
        st.subheader("Evaluation")

        target_variable = st.session_state.target_variable

        model = st.selectbox("Select ML Model:", options=[model.value for model in MLModels])

        result = evaluate_join_tree(join_trees=join_trees, join_tree_id=display_tree.join_tree_id, target_variable=target_variable, ml_model=MLModels(model).name)
        st.write(result)

