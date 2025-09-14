import datetime
import os

import pandas as pd
import plotly.express as px
import streamlit as st
# load environment variables
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# local imports
from dimensionality_reduction import reduce_dimensionality_of_project_data
from clustering import calculate_cluster_flat
from cluster_naming import generate_cluster_names

# streamlit config
st.set_page_config(layout="wide")

# Password protection
PW = "awesome"


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == PW:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.write("CDTM is mostly ...")
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


# load the model
@st.cache_resource
def load_model():
    model = SentenceTransformer('mpi-inno-comp/paecter')
    return model


model = load_model()

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

st.header("Automated Patent Landscaping")

################################################################################
# Create an empty dataframe using id, title, abstract

demo_data = [
    {
        "id": "1",
        "title": "Safe model-based reinforcement learning with stability guarantees",
        "abstract": "Reinforcement learning is a powerful paradigm for learning optimal policies from experimental data. However, to find optimal policies, most reinforcement learning algorithms explore all possible actions, which may be harmful for real-world systems. As a consequence, learning algorithms are rarely applied on safety-critical systems in the real world. In this paper, we present a learning algorithm that explicitly considers safety, defined in terms of stability guarantees. Specifically, we extend control-theoretic results on Lyapunov stability verification and show how to use statistical models of the dynamics to obtain high-performance control policies with provable stability certificates. Moreover, under additional regularity assumptions in terms of a Gaussian process prior, we prove that one can effectively and safely collect data in order to learn about the dynamics and thus both improve control performance and expand the safe region of the state space. In our experiments, we show how the resulting algorithm can safely optimize a neural network policy on a simulated inverted pendulum, without the pendulum ever falling down."
    },
    {
        "id": "2",
        "title": "Safe learning in robotics: From learning-based control to safe reinforcement learning",
        "abstract": "The last half decade has seen a steep rise in the number of contributions on safe learning methods for real-world robotic deployments from both the control and reinforcement learning communities. This article provides a concise but holistic review of the recent advances made in using machine learning to achieve safe decision-making under uncertainties, with a focus on unifying the language and frameworks used in control theory and reinforcement learning research. It includes learning-based control approaches that safely improve performance by learning the uncertain dynamics, reinforcement learning approaches that encourage safety or robustness, and methods that can formally certify the safety of a learned control policy. As data- and learning-based robot control methods continue to gain traction, researchers must understand when and how to best leverage them in real-world scenarios where safety is"
    },
]

df_demo_data = pd.DataFrame(demo_data)

# add a dataframe editor
st.subheader("Data")
st.write("Please provide the data you want to use for the landscaping process.")
st.write("The data should contain at least an 'id', 'title', and 'abstract' column.")
st.write("The 'id' column should contain a unique identifier for each row.")
st.write("The 'title' column should contain the title of the document.")
st.write("The 'abstract' column should contain the abstract of the document.")

# show the data
df = st.data_editor(
    df_demo_data,
    num_rows="dynamic",
    hide_index=True,
    column_config={
    }
)

if 'submitted' not in st.session_state:
    st.session_state.submitted = False


def click_button():
    st.session_state.submitted = not st.session_state.submitted


if st.session_state.submitted:
    st.button('Stop', on_click=click_button)
else:
    st.button('Start', on_click=click_button)

# stop process if not already submitted
if not st.session_state.submitted:
    st.stop()

################################################################################
# Encode the data using the PaECTER model
st.divider()
st.subheader("Encoding")
st.write("The data will be encoded using the PaECTER model.")
st.write("This might take a while depending on the size of the data ...")


def encode_sentences(sentences):
    """
    Encode the sentences
    :param sentences:
    :return:
    """
    embeddings = model.encode(sentences)
    return embeddings


with st.spinner("Encoding Data..."):
    # transform the data
    df["text"] = df["title"] + "[SEP]" + df["abstract"]
    sentences = df["text"].tolist()
    embeddings = encode_sentences(sentences)
    # embeddings are a list of lists of floats
    df["embedding"] = list(embeddings)
    # drop the text column
    df = df.drop(columns=["text"])

################################################################################
# Add neighbors
st.divider()
st.subheader("Expanding")

if 'expanding' not in st.session_state:
    st.session_state.expanding = False


nearest_neighbors = st.slider("Number of Nearest Neighbors", 0, 200, 150)


def click_expanding_button():
    st.session_state.expanding = not st.session_state.expanding


if st.session_state.expanding:
    st.button('Stop', key="expanding_stop", on_click=click_expanding_button)

else:
    st.button('Start', key="expanding_start",  on_click=click_expanding_button)

# stop process if not already submitted
if not st.session_state.expanding:
    st.stop()


@st.cache_data
def find_nearest_neighbors(df, n_neighbors):
    """
    Find the nearest neighbors
    :param df:
    :param n_neighbors:
    :return:
    """
    if n_neighbors == 0:
        df["is_neighbor"] = False
        return df

    neighbor_progress_bar_text = "Finding Nearest Neighbors"
    neighbor_progress_bar = st.progress(0, text=neighbor_progress_bar_text)
    results = []

    total = len(df)
    for index, row in df.iterrows():
        print(f"Finding neighbors for {index} of {total}")
        progress_value = min(1, index / total)
        neighbor_progress_bar.progress(progress_value, text=neighbor_progress_bar_text)

        embedding_str = "[" + ",".join([str(x) for x in row["embedding"]]) + "]"
        query = f"""
            SET hnsw.ef_search = {n_neighbors * 2};
            with ann as (
                SELECT *, 1 - (embedding <=> '{embedding_str}') as cosine_similarity FROM patent_embeddings_hsnw ORDER BY embedding <=> '{embedding_str}' LIMIT {n_neighbors}
            ),
            title as (
                select * from tls202_appln_title
                where appln_title_lg = 'en'
            ),
            abstract as (
                select * from tls203_appln_abstr
                where appln_abstract_lg = 'en'
            ),
            organizations as (
                SELECT orb.bvdid_number, orb.name_internat, 
                CASE
                   WHEN orb.website_address IS NULL THEN NULL
                   ELSE CONCAT('https://', orb.website_address)
                END AS website_address,
                orb.country
                FROM orb_contact_info as orb
            )
            SELECT DISTINCT ON (p.appln_id) t.appln_title as title, 
            a.appln_abstract as abstract, 
            p.*, 
            orb.name_internat as organization,
            orb.website_address as website,
            orb.country as country,
            e.cosine_similarity,
            e.embedding FROM ann as e
            LEFT JOIN mv_bvdid_patpublnr_appln_id as p on e.appln_id = p.appln_id
            LEFT JOIN title as t on p.appln_id = t.appln_id
            LEFT JOIN abstract as a on p.appln_id = a.appln_id
            LEFT JOIN mv_bvdid_patpublnr_appln_id as p2 on e.appln_id = p2.appln_id
            LEFT JOIN organizations as orb on p2.bvdid = orb.bvdid_number
            ORDER BY p.appln_id, cosine_similarity
        """
        # print(query)

        df_neighbors = pd.read_sql(query, engine)
        df_neighbors = df_neighbors.dropna(subset=["embedding"])
        # drop the ones that have no title or abstract
        df_neighbors = df_neighbors.dropna(subset=["title", "abstract"])

        # convert the string embedding to a list of floats
        df_neighbors["embedding"] = df_neighbors["embedding"].apply(lambda x: [float(y) for y in x[1:-1].split(",")])
        df_neighbors = df_neighbors.drop_duplicates(subset=["appln_id"])
        # add the id of the original row
        df_neighbors["is_neighbor_to"] = row["id"]
        print("len df_neighbors of ", row["id"], len(df_neighbors))
        results.append(df_neighbors)

    # merge the results
    df_neighbors = pd.concat(results)
    # mark the original rows
    df_neighbors["is_neighbor"] = True
    df["is_neighbor"] = False
    # combine the data
    df = pd.concat([df, df_neighbors])

    # reset the index
    df = df.reset_index(drop=True)

    neighbor_progress_bar.empty()

    return df


df_with_neighbors = find_nearest_neighbors(df, nearest_neighbors)

print("len df_with_neighbors", len(df_with_neighbors))

df_with_neighbors = st.data_editor(
    df_with_neighbors,
    num_rows="dynamic",
    hide_index=True,
    column_config={"bvdid": None},
)

################################################################################
# Dimensionality Reduction
st.divider()
st.subheader("Dimensionality Reduction")
toggle_3d = st.checkbox("3D")
n_components = 2  # either 2 or 3
if toggle_3d:
    n_components = 3

n_neighbors = st.slider("Number of Neighbors", 1, 100, 3)
min_dist = st.slider("Minimum Distance", 0.0, 1.0, 0.1)
metric = st.selectbox("Metric", ["euclidean", "cosine", "manhattan"], index=0)

with st.spinner("Reducing Dimensionality..."):
    df = reduce_dimensionality_of_project_data(
        df_with_neighbors,
        vector_column_name="embedding",
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    )

################################################################################
# Clustering
st.divider()
st.subheader("Clustering")

cols = ["x", "y"]
if n_components == 3:
    cols.append("z")

min_cluster_size = st.slider("Minimum Cluster Size", 1, 100, 5)

with st.spinner("Clustering..."):
    df_clusters = calculate_cluster_flat(df, cols=cols, min_cluster_size=min_cluster_size)

with st.spinner("Generating Cluster Names..."):
    df_cluster_names = generate_cluster_names(df_clusters, content_columns=["title", "abstract"])

with st.spinner("Generate Cluster Centers..."):
    grouped = df_cluster_names.groupby('cluster_id')
    cluster_centers = grouped[['x', 'y']].mean()
    # add cluster names
    cluster_centers = cluster_centers.join(df_cluster_names.set_index('cluster_id')[['cluster_name']], on='cluster_id')

################################################################################
# Organizations
st.divider()
st.subheader("Organizations")


# get the organizations
@st.cache_data
def get_organizations(df):
    # Ensure 'bvdid' column exists in the input DataFrame
    if 'bvdid' not in df.columns:
        raise KeyError("The input DataFrame does not contain the 'bvdid' column.")

    # Count the occurrences of the 'bvdid'
    counts = df["bvdid"].value_counts().reset_index()
    counts.columns = ["bvdid", "count"]

    # Get the unique bvdids
    unique_bvdids = df["bvdid"].unique()
    str_bvdids = ", ".join(["'" + str(x) + "'" for x in unique_bvdids])

    # Get the organizations from the database
    query = f"""
    with organizations as (
        SELECT orb.bvdid_number, 
        orb.name_internat, 
        CASE
           WHEN orb.website_address IS NULL THEN NULL
           ELSE CONCAT('https://', orb.website_address)
        END AS website_address,
        orb.country
        FROM orb_contact_info as orb
        WHERE bvdid_number IN ({str_bvdids})
    ),
    embeddings as (
        select distinct ON (emb.bvdid_number) *
        FROM firm_embeddings_hsnw as emb order by emb.bvdid_number, emb.t desc 
    )
    SELECT orb.*, e.embedding as embedding
    FROM organizations as orb
    LEFT JOIN embeddings as e ON e.bvdid_number = orb.bvdid_number
    """

    df_organizations = pd.read_sql(query, engine)

    # Ensure 'bvdid_number' column exists in the df_organizations DataFrame
    if 'bvdid_number' not in df_organizations.columns:
        raise KeyError("The DataFrame returned by the SQL query does not contain the 'bvdid_number' column.")

    # Rename 'bvdid_number' to 'bvdid' to match the counts DataFrame
    df_organizations.rename(columns={'bvdid_number': 'bvdid'}, inplace=True)

    # Parse the embedding if it exists
    df_organizations["embedding"] = df_organizations["embedding"].apply(
        lambda x: [float(y) for y in x[1:-1].split(",")] if x else None)

    # Add the count to the DataFrame
    df_organizations = df_organizations.join(counts.set_index('bvdid'), on="bvdid", how="left")

    # order by count
    df_organizations = df_organizations.sort_values(by="count", ascending=False)

    return df_organizations


with st.spinner("Getting Organizations..."):
    df_organizations = get_organizations(df_cluster_names)

# add link column
st.dataframe(
    df_organizations,
    use_container_width=True,
    hide_index=True,
    column_config={
        "website_address": st.column_config.LinkColumn(),
        "embedding": None,
        "bvdid": None,
    },

)

# add a barchart

fig = px.bar(
    df_organizations,
    x='name_internat',
    y='count',
    color='country',
    hover_data=["website_address"],
    height=1000,
    title="Organizations"
)
st.plotly_chart(fig)

################################################################################
# Visualization
# use plotly to visualize the clusters


fig = None
st.divider()
st.subheader("Visualization")

show_cluster_names = st.checkbox("Show Cluster Names", value=True)

with st.spinner("Visualizing..."):
    if n_components == 2:
        fig = px.scatter(
            df_cluster_names,
            x='x', y='y',
            color='cluster_name',
            symbol='is_neighbor',
            symbol_map={
                True: "circle-open",
                False: "circle"
            },
            color_discrete_map={
                "no cluster": "grey"
            },
            hover_data=["title"],
            height=800,
        )
        if show_cluster_names:
            # add cluster centers as annotations
            for index, row in cluster_centers.iterrows():
                if row["cluster_name"] == "no cluster":
                    continue
                fig.add_annotation(
                    x=row['x'],
                    y=row['y'],
                    yshift=10,
                    text=str(row["cluster_name"]),
                    showarrow=False,
                    opacity=0.1
                )
    else:
        fig = px.scatter_3d(df_cluster_names, x='x', y='y', z='z', color='cluster_name')

    # remove axis and grid
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if n_components == 3:
        fig.update_zaxes(visible=False)

    st.plotly_chart(fig)

st.divider()
st.subheader("Dataset")

st.write(df_cluster_names)

################################################################################
# Download

st.divider()
st.subheader("Download")
st.write("You can download the data and the visualization as a CSV file.")


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


csv = convert_df(df_cluster_names)

# current time

file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%-M%-S") + "-landscaping.csv"

st.download_button(
    "Press to Download",
    csv,
    file_name,
    "text/csv",
    key='download-csv'
)
