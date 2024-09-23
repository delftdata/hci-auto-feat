import streamlit as st
import streamlit_pydantic as sp
from pydantic import BaseModel

class FeatureDiscovery(BaseModel):
    type: bool
    repository: str
    matcher: str
    base_table: str
    target_var: str
    data_quality: float 
    top_features: int
    top_join_tree: int 


st.header("Saved Process")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

btn_start = st.button("Start process", on_click=set_state, args=[1])

placeholder = st.empty()

with placeholder.container():
    st.write("Welcome to your saved feature discovery process")
    st.write("Empty process collections")

if st.session_state.stage >= 1:
    placeholder.empty()
    data = sp.pydantic_form(key="my_sample_form", model=FeatureDiscovery)
    if data:
        st.json(data.model_dump())


