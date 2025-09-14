import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
# load environment variables
from dotenv import load_dotenv
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


@st.cache_data
def load_data(pat_publ_nrs: list):
    str_pat_publ_nrs = ", ".join(["'" + str(x) + "'" for x in pat_publ_nrs])
    query = f"""
        with patents as (
        select * from mv_bvdid_patpublnr_appln_id
        where patpublnr in ({str_pat_publ_nrs})
        ),
        title as (
        select * from tls202_appln_title
        where appln_title_lg = 'en'
        ),
        abstract as (
        select * from tls203_appln_abstr
        where appln_abstract_lg = 'en'
        )

        select t.appln_title as title, a.appln_abstract as abstract, p.*, pe.embedding from patents as p
        left join patent_embeddings_hsnw as pe on p.appln_id = pe.appln_id
        left join title as t on p.appln_id = t.appln_id
        left join abstract as a on p.appln_id = a.appln_id
        """

    df = pd.read_sql(query, engine)

    # remove rows with missing embeddings
    df = df.dropna(subset=["embedding"])

    # convert the string embedding to a list of floats
    df["embedding"] = df["embedding"].apply(lambda x: [float(y) for y in x[1:-1].split(",")])
    return df


st.header("Automated Patent Landscaping")

# demo data
df_initial = pd.read_csv("mrna.csv")

# only take the 100
df_initial = df_initial.head(100)

st.info('The initial data', icon="ℹ️")
initial_data_df = st.data_editor(df_initial, num_rows="dynamic", column_config={"bvdid": None})

# Shot the number of rows
st.write(f"Number of rows: {initial_data_df.shape[0]}")

# take the top 10 rows
# edited_df = edited_df.head(300)
# append A1 to pat_publ_nr
initial_data_df["pat_publ_nr"] = initial_data_df["pat_publ_nr"].apply(lambda x: x + "A1")

################################################################################
# Get data and embeddings from server
pat_publ_nrs = initial_data_df["pat_publ_nr"].tolist()
# get the data from the database
with st.spinner("Loading data..."):
    df = load_data(pat_publ_nrs)

# show the data
initial_data_df = st.data_editor(
    df,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "bvdid": None,
        "embedding": None,
    }
)

################################################################################
# Add neighbors
st.divider()
st.subheader("Expanding")

amount_of_neighbors = 5
if len(df) > 500:
    amount_of_neighbors = 3
elif len(df) > 1000:
    amount_of_neighbors = 1

nearest_neighbors = st.slider("Number of Nearest Neighbors", 0, 100, amount_of_neighbors)


@st.cache_data
def find_nearest_neighbors(df, n_neighbors=25):
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
            )
            SELECT DISTINCT ON (p.appln_id) t.appln_title as title, a.appln_abstract as abstract, p.*, e.embedding FROM ann as e
            LEFT JOIN mv_bvdid_patpublnr_appln_id as p on e.appln_id = p.appln_id
            LEFT JOIN title as t on p.appln_id = t.appln_id
            LEFT JOIN abstract as a on p.appln_id = a.appln_id
            ORDER BY p.appln_id, cosine_similarity
        """

        df_neighbors = pd.read_sql(query, engine)
        df_neighbors = df_neighbors.dropna(subset=["embedding"])
        # drop also the ones that are the same as the original
        df_neighbors = df_neighbors[df_neighbors["appln_id"] != row["appln_id"]]
        # drop the ones that have no title or abstract
        df_neighbors = df_neighbors.dropna(subset=["title", "abstract"])

        # convert the string embedding to a list of floats
        df_neighbors["embedding"] = df_neighbors["embedding"].apply(lambda x: [float(y) for y in x[1:-1].split(",")])
        df_neighbors = df_neighbors.drop_duplicates(subset=["appln_id"])
        results.append(df_neighbors)

    # merge the results
    df_neighbors = pd.concat(results)
    # mark the original rows
    df_neighbors["is_neighbor"] = True
    df["is_neighbor"] = False
    # combine the data
    df = pd.concat([df, df_neighbors])

    neighbor_progress_bar.empty()

    return df


df_with_neighbors = find_nearest_neighbors(df, n_neighbors=nearest_neighbors)

df_with_neighbors = st.data_editor(
    df_with_neighbors,
    num_rows="dynamic",
    hide_index=True,
    column_config={"bvdid": None}
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
        SELECT orb.bvdid_number, orb.name_internat, concat('https://', orb.website_address) as website_address, orb.country
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

st.write(df_cluster_names)
