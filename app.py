import streamlit as st
from presentation import presentation_page

PAGES = {
    "Презентация": presentation_page,
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Страницы:", list(PAGES.keys()))
page = PAGES[selection]
page()
