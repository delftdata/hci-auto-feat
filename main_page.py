from enum import Enum
from typing import List, Set
import streamlit as st
import streamlit_pydantic as sp
import polars as pl
from pydantic import BaseModel, Field
from st_link_analysis import Event, st_link_analysis, NodeStyle, EdgeStyle

from app.forms.automatic import FeatureDiscovery
from app.forms.human_in_the_loop import HILProcess
from app.forms.join_trees import DisplayJoinTree, print_join_trees
from app.graph.graph_model import edge_styles, from_relations_to_graph, node_styles
from autotda.config import DATA_FOLDER
from autotda.data_models.dataset_model import ALL_DATASETS, Dataset
from autotda.functions.evaluation_functions import evaluate_join_tree
from autotda.functions.tree_functions import HCIAutoFeat, JoinTree
from src.autotda.functions.relationship_functions import DatasetDiscovery, Relation


# st.set_page_config(layout="wide")

hil_process = "Human-in-the-loop"
auto_process = "Automatic"
process_types = [hil_process, auto_process]

# node_styles_list = [
#     NodeStyle("TABLE", "#FF7F3E", "alias"),
# ]

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

if "elements" not in st.session_state:
    st.session_state.elements = None

if "relations_updated" not in st.session_state:
    st.session_state.relations_updated = False  

if "selected_edge" not in st.session_state:
    st.session_state.selected_edge = None

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

def set_state(i):
    st.session_state.stage = i

def set_tree_state(state: bool):
    st.session_state.tree = state

def find_relations(state: int, dataset_discovery: DatasetDiscovery):
    set_state(state)
    dataset_discovery.find_relationships()
    st.session_state.dataset_discovery = dataset_discovery

def compute_join_trees(autofeat: HCIAutoFeat):
    autofeat.compute_join_trees(queue={autofeat.base_table})

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


def discover_relations(similarity_score=None):
    dataset_discovery = st.session_state.dataset_discovery
    if similarity_score:
        dataset_discovery.set_similarity_threshold(similarity_score)
    dataset_discovery.find_relationships()
    nodes, edges = from_relations_to_graph(dataset_discovery.relations)
    elements = {
        "nodes": nodes,
        "edges": edges
    }
    
    st.session_state.dataset_discovery = dataset_discovery
    st.session_state.elements = elements

    set_state(2)

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
        _, col2 = st.columns([3, 1], gap="large")
        if data:
            selected_repos = [repo.value for repo in list(data.repositories)]
            selected_matcher = data.matcher.value
            dataset_discovery = DatasetDiscovery(matcher=selected_matcher, data_repositories=selected_repos)
            st.session_state.dataset_discovery = dataset_discovery
           
            next_button = col2.button("Find relations", on_click=discover_relations)
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


def update_weight(weight_value, selected_edge, elements):
    dataset_discovery: DatasetDiscovery = st.session_state.dataset_discovery

    nodes = elements['nodes']
    edges = elements['edges']

    edges.remove(selected_edge[0])

    my_edge = selected_edge[0]['data']
    relations = [rel for rel in dataset_discovery.relations if (rel.from_table == my_edge['source_name'] and
                                                                     rel.to_table == my_edge['target_name'] and
                                                                     rel.from_col == my_edge['source_column'] and 
                                                                     rel.to_col == my_edge['target_column'] and 
                                                                     rel.similarity == my_edge['weight'])]
    dataset_discovery.relations.remove(relations[0])
    my_relation = relations[0]
    my_relation.similarity = weight_value
    dataset_discovery.relations.append(my_relation)

    selected_edge[0]['data']['weight'] = weight_value
    edges.append(selected_edge[0])
   
    st.session_state.elements = {
        "nodes": nodes,
        "edges": edges
    }
    st.session_state.selected_edge = None
    st.session_state.dataset_discovery = dataset_discovery


def delete_edge(selected_edge, elements):
    dataset_discovery: DatasetDiscovery = st.session_state.dataset_discovery
    nodes = elements['nodes']
    edges = elements['edges']
    edges.remove(selected_edge[0])

    my_edge = selected_edge[0]['data']
    relations = [rel for rel in dataset_discovery.relations if (rel.from_table == my_edge['source_name'] and
                                                                     rel.to_table == my_edge['target_name'] and
                                                                     rel.from_col == my_edge['source_column'] and 
                                                                     rel.to_col == my_edge['target_column'] and 
                                                                     rel.similarity == my_edge['weight'])]
    dataset_discovery.relations.remove(relations[0])

    st.session_state.elements = {
        "nodes": nodes,
        "edges": edges
    }
    st.session_state.selected_edge = None
    st.session_state.dataset_discovery = dataset_discovery


def graph_actions() -> None:
    dataset_discovery: DatasetDiscovery = st.session_state.dataset_discovery
    payload = st.session_state['drg']

    if payload["action"] == "remove":
        node_ids = payload["data"]["node_ids"]

        elements = st.session_state.elements
        nodes: List = elements['nodes']
        edges = elements['edges']

        nodes_to_remove = [node for node in nodes if node['data']['id'] in node_ids]
        for node in nodes_to_remove:
            nodes.remove(node)

            relations = [rel for rel in dataset_discovery.relations if not (rel.from_table == node['data']['name'] or rel.to_table == node['data']['name'])]
            dataset_discovery.relations = relations

        elements = {
            "nodes": nodes,
            "edges": edges
        }

        st.session_state.elements = elements
        st.session_state.dataset_discovery = dataset_discovery
        st.session_state.relations_updated = True

    elif payload['action'] == 'clicked_edge':
        elements = st.session_state.elements
        nodes: List = elements['nodes']
        edges: List = elements['edges']

        edge_id = payload['data']['target_id']
        selected_edge = [edge for edge in edges if edge['data']['id'] == edge_id]
        st.session_state.selected_edge = selected_edge

    elif payload['action'] == 'clicked_node':
        elements = st.session_state.elements
        nodes: List = elements['nodes']
        node_id = payload['data']['target_id']
        selected_node = [node for node in nodes if node['data']['id'] in node_id]
        if selected_node and len(selected_node) > 0:
            selected_node = selected_node[0]
            st.session_state.selected_node = selected_node

    else:
        st.session_state.selected_edge = None
        st.session_state.selected_node = None

    # st.session_state['drg'] = None


if st.session_state.stage == 2:
    _, b = st.columns([3, .4])
    b.button("Back", on_click=set_state, args=[1])
    st.title("2. Find relations")

    sim_col, buffer, create_col = st.columns([.6, .5, .4], vertical_alignment="bottom", gap="large")
    
    form = sim_col.form(key='similarity_th_form', border=False)
    similarity_score = form.number_input(
        value=0.65,
        label='Similarity threshold',
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="The similarity threshold will filter the nodes.")
    submit_button = form.form_submit_button(label='Update graph', on_click=discover_relations, args=[similarity_score])
   
    events = [
        Event("clicked_node", "click tap", "node"),
        Event("clicked_edge", "click tap", "edge"),
        Event("deselect", "unselect", "edge"),
        Event("deselect_node", "unselect", "node")
    ]

    elements = st.session_state.elements 

    st_link_analysis(elements, "cose", node_styles, edge_styles, 
                    key="drg", 
                    node_actions=['remove'], 
                    events=events,
                    on_change=graph_actions) 
    
    selected_edge = st.session_state.selected_edge
    
    if selected_edge and len(selected_edge) > 0:
        slt_col, upt_col, del_col = st.columns([.7, .5, .4], vertical_alignment="bottom", gap="large")

        weight_value = slt_col.number_input(
            value=selected_edge[0]['data']['weight'],
            label='Edge Weight',
            min_value=0.0,
            step=0.01,
            max_value=1.0,
            help="Adjust the weight of the relation."
        )

        if weight_value != float(selected_edge[0]['data']['weight']):
            upt_col.button("Update edge", on_click=update_weight, args=[weight_value, selected_edge, elements])

        del_col.button("Delete edge", on_click=delete_edge, args=[selected_edge, elements])    

    create_tree_button = create_col.button("Create join trees", on_click=set_state, args=[3])

    selected_node = st.session_state.selected_node
    if selected_node:
        st.subheader(f"Sample from table: {selected_node['data']['name']}")
        st.write(pl.read_csv(DATA_FOLDER / selected_node['data']['name'], n_rows=20))


def prepare_next_state(display_trees, value, state_num):
    st.session_state.selected_tree = display_trees[value]
    set_state(state_num)

if st.session_state.stage == 3:
    _, b = st.columns([3, .4])
    b.button("Back", on_click=set_state, args=[2])
    st.title("3. Create join trees")

    if st.session_state.tree == False:
        dataset_discovery = st.session_state.dataset_discovery
        target_variable_choices = [dataset.target_column for dataset in dataset_discovery.data_repository]
        base_table_options = [dataset.base_table_id for dataset in dataset_discovery.data_repository]

        form_jt = st.form(key="join_tree_form")

        base_table = form_jt.selectbox(label="Base Table", options=tuple(base_table_options), index=None, help="The Base Table is the table to be augmented.")
        target_var = form_jt.selectbox(label="Target Variable", options=tuple(target_variable_choices), index=None, help="The Target Variable is the column containing the class labels for ML modelling.")
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
        _, col2 = st.columns([3, 1], gap="large")
        if submit_button:
            autofeat = HCIAutoFeat(
                base_table=str(base_table),
                target_variable=target_var,
                non_null_ratio=non_null_ratio,
                top_k_features=top_k_features,
                top_k_join_trees=top_k_join_trees,
                relations=dataset_discovery.relations,
                updated=st.session_state.relations_updated
            )

            next_button = col2.button("Start process", on_click=compute_join_trees, args=[autofeat])
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
                st_link_analysis(tr.elements, "breadthfirst", node_styles, edge_styles, height=150) 

            # table_col.dataframe(tr.table_data, width=250, height=150)

            clicked = btn_col.button(" ", key=f"btn_{i}", icon=":material/arrow_forward_ios:", on_click=prepare_next_state, args=[display_trees, i, 4])


class MLModels(Enum):
    XGB = "XGBoost"
    GBM = "LightGBM" 
    RF = "Random Forest" 
    XT = "Extremely Randomized Trees"
    KNN = "k-Nearest Neighbors"
    LR = "Linear Regression"

def move_to_discarded(rows_to_discard: List[int], display_tree: DisplayJoinTree):
    join_trees = st.session_state.join_trees

    features_to_discard = list(display_tree.selected_features[rows_to_discard]['column_name'])
    jt, fn = join_trees[display_tree.join_tree_id]
    join_tree_features = jt.features

    for el in features_to_discard:
        if el in join_tree_features:
            join_tree_features.remove(el)

    jt.features = join_tree_features
    
    join_trees[display_tree.join_tree_id] = (jt, fn) 

    display_tree.discarded_features.extend(display_tree.selected_features[rows_to_discard])
    display_tree.selected_features = display_tree.selected_features.with_row_index().filter(~pl.col("index").is_in(rows_to_discard)).drop(['index'])
    
    st.session_state.join_trees = join_trees
    st.session_state.selected_tree = display_tree


def move_to_selected(rows_to_select: List[int], display_tree: DisplayJoinTree):
    join_trees = st.session_state.join_trees

    features_to_add = list(display_tree.discarded_features[rows_to_select]['column_name'])

    jt, fn = join_trees[display_tree.join_tree_id]
    jt.features.extend(features_to_add)
    join_trees[display_tree.join_tree_id] = (jt, fn) 

    display_tree.selected_features.extend(display_tree.discarded_features[rows_to_select])
    display_tree.discarded_features = display_tree.discarded_features.with_row_index().filter(~pl.col("index").is_in(rows_to_select)).drop(['index'])
    
    st.session_state.join_trees = join_trees
    st.session_state.selected_tree = display_tree


def update_tree():
    payload = st.session_state['small_join_tree']
    
    if payload['action'] == 'remove':
        node_ids = payload["data"]["node_ids"]
        display_tree: DisplayJoinTree = st.session_state.selected_tree


if st.session_state.stage == 4:
    _, b = st.columns([3, .4])
    b.button("Back", on_click=set_state, args=[3])
    st.title("4. View and Evaluate")

    display_tree: DisplayJoinTree = st.session_state.selected_tree

    container = st.container(key=f"cont_selected_tree", border=True)
    rank_col, graph_col = container.columns([.4, 2], vertical_alignment="center")

    rank_col.write("Score:")
    rank_col.write("{:.3f}".format(display_tree.rank)) 
    with graph_col:
        st_link_analysis(display_tree.elements, "breadthfirst", node_styles, edge_styles, 
                         height=150,
                         key="small_join_tree") 

    bottom_cont = container.container(key="cont_bottom")

    tab1, tab2, tab3, tab4 = bottom_cont.tabs(["Selected Features", "Discarded Features", "Augmented Table", "Evaluation"])

    with tab1:
        display_tree: DisplayJoinTree = st.session_state.selected_tree
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
        if "selection" in event and 'rows' in event['selection']:
            rows_to_discard = event['selection']['rows']
            
        _, col2 = st.columns([2.5, 1], gap="large")
        col2.button("Move to discarded", key='discarded-button', on_click=move_to_discarded, args=[rows_to_discard, display_tree])
        

    with tab2:
        display_tree: DisplayJoinTree = st.session_state.selected_tree
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

        if "selection" in event and 'rows' in event['selection']:
            rows_to_select = event['selection']['rows']
            
        _, col2 = st.columns([2.5, 1], gap="large")
        col2.button("Move to selected", key='selected-button', on_click=move_to_selected, args=[rows_to_select, display_tree])

    with tab3:
        st.subheader("Augmented Table")

        join_trees = st.session_state.join_trees
        display_tree: DisplayJoinTree = st.session_state.selected_tree
        selected_tree, _ = join_trees[display_tree.join_tree_id]

        columns_to_drop = set(selected_tree.features).intersection(set(selected_tree.join_keys))
        materialised_join = pl.read_parquet(display_tree.join_tree_filename).select(selected_tree.features).drop(columns_to_drop)
        
        st.dataframe(materialised_join)

    with tab4:
        st.subheader("Evaluation")

        join_trees = st.session_state.join_trees
        target_variable = st.session_state.target_variable
        display_tree: DisplayJoinTree = st.session_state.selected_tree

        model = st.selectbox("Select ML Model:", options=[model.value for model in MLModels])

        result = evaluate_join_tree(join_trees=join_trees, 
                                    join_tree_id=display_tree.join_tree_id, 
                                    target_variable=target_variable, 
                                    ml_model=MLModels(model).name
                                    )
        if len(result) > 0:
            st.write("We evaluate the dataset using the AutoML framework AutoGluon.")
            st.write(f"The **accuracy** of the model {result[0].model} is **{result[0].accuracy}**")
            st.write("The model used the following features:")
            st.write(pl.DataFrame({"Feature": result[0].feature_importance.keys(),
                                   "Feature Importance": result[0].feature_importance.values()
                                   }))

