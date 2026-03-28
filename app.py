import streamlit as st

pg = st.navigation(
    [
        st.Page("analysis_and_model.py", title="Анализ и модель"),
        st.Page("presentation.py", title="Презентация"),
    ],
    position="sidebar",
    expanded=True
)

pg.run()