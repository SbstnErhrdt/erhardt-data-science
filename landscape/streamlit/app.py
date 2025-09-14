import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# streamlit config

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Password protection


# Get postgres connection using streamlit
# use the environment variables
DB_DATABASE = os.getenv("DB_DATABASE")
DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
# check if empty
if not DB_DATABASE or not DB_HOST or not DB_PASSWORD or not DB_PORT or not DB_USER:
    raise ValueError("Please provide the necessary environment variables for the database connection.")

# connect to the database
engine = st.connection(
    "postgresql",
    type="sql",
    url="postgresql://" + DB_USER + ":" + DB_PASSWORD + "@" + DB_HOST + ":" + DB_PORT + "/" + DB_DATABASE,
).engine

# add a title
st.title("Landscaping")

st.write("This is a demonstration of the landscaping process using PaECTER.")


st.divider()

# go to page patent landscaping
st.subheader("Patent to Patent Landscaping")
st.markdown("""
If you have a list of patent numbers, this tool helps you generate a patent landscape.
""")
if st.button("Start", key="patent_patent"):
    st.switch_page("pages/patent_patent.py")

st.divider()

# go to page technology landscaping
st.subheader("Document to Patent Landscaping")
st.markdown("""
If you have document titles and abstracts, this tool can generate a patent landscape.
""")
if st.button("Start", key="document_patent"):
    st.switch_page("pages/document_patent.py")
