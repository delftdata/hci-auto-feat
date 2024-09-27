from enum import Enum
from typing import Set
import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel, Field

from app.forms.automatic import FeatureDiscovery
from app.forms.human_in_the_loop import HILProcess
from app.forms.join_trees import JoinTreeData

hil_process = "Human-in-the-loop"
auto_process = "Automatic"
process_types = [hil_process, auto_process]


if 'stage' not in st.session_state:
    st.session_state.stage = 0

if "tree" not in st.session_state:
    st.session_state.tree = False

def set_state(i):
    st.session_state.stage = i

def set_tree_state(state: bool):
    st.session_state.tree = state
    
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
            clear_on_submit=True,
            )
        
        if data:
            json_data = st.json(data)
            next_button = st.button("Find relations", on_click=set_state, args=[2])

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
    form.number_input(
        value=0.0,
        label='Similarity threshold',
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="The similarity threshold will filter the nodes.")
    submit_button = form.form_submit_button(label='Update graph')

    placeholder = st.empty()
    placeholder.write("This is where the graph is going to be")

    create_tree_button = st.button("Create join trees", on_click=set_state, args=[3])

if st.session_state.stage == 3:
    st.header("3. Create join trees")

    if st.session_state.tree == False:
        data = sp.pydantic_form(
                key="join_tree_process", 
                model=JoinTreeData, 
                submit_label="Submit data",
                clear_on_submit=True,
                )
            
        if data:
            json_data = st.json(data)
            next_button = st.button("Start process", on_click=set_tree_state, args=[True])
    else:
        placeholder = st.empty()
        placeholder.write("Here we will have a list")
        click_tree_button = st.button("Click on a join tree", on_click=set_state, args=[4])


if st.session_state.stage == 4:
    st.header("4. Update and Evaluate")