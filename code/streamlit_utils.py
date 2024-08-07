import streamlit as st

_suffix = "_qp"

def process_query_param(k, v):
    if "list" not in k and len(v) == 1:
        v = v[0]
    if v in ["True", "False"]:
        v = (v == "True")
    return v

def ss_from_qp():
    state = {key: st.query_params.get_all(key) for key in st.query_params}
    state = {k: process_query_param(k, v) for k, v in state.items()}
    st.session_state.update(state)

def ss_to_qp():
    qp_state = {k: v for k, v in st.session_state.items() if _suffix in k}
    qp_state = {k: v for k, v in qp_state.items() if v is not None}
    st.query_params.from_dict(qp_state)
