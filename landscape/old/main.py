import math
from typing import List

import nltk
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import cluster_naming
import competitors
# import cluster_naming
import dimensionality_reduction
import han_id_embeddings
import sql
from clustering import calculate_cluster_flat

# download nltk punkt
nltk.download('punkt')

# connect to database
sql.connect()

# init api
app = FastAPI()


# connect to database after starting the app
@app.on_event("startup")
async def startup_event():
    sql.connect()


# close database connection after stopping the app
@app.on_event("shutdown")
async def shutdown_event():
    sql.close()


class ReportRequestDTO(BaseModel):
    hanIds: List[str]

    class Config:
        schema_extra = {
            "hanIds": ["2697604", "2697605"]
        }


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.put("/analyze")
def analyze_endpoint(payload: ReportRequestDTO):
    han_ids = payload.hanIds
    print("han_ids: ", han_ids)
    # generate data from request
    # get embeddings
    print("get embeddings")
    df = han_id_embeddings.get_han_id_data_as_dataframe(han_ids)
    if df is None:
        return {"points": [], "clusters": []}
    # dim reduction
    print("dim reduction")
    df_dim_2d = dimensionality_reduction.reduce_dimensionality_of_project_data(
        df,
        n_components=2,
        n_neighbors=8,
        min_dist=0.03,
        metric="euclidean"
    )

    # clustering parameters
    """
    If you want many highly specific clusters, use a small min_samples and a small min_cluster_size.
    If you want more generalized clusters but still want to keep most detail, use a small min_samples and a large min_cluster_size
    If you want very very general clusters and to discard a lot of noise in the clusters, use a large min_samples and a large min_cluster_size.
    """

    amount_of_data_points = len(df_dim_2d)
    if amount_of_data_points < 100:
        min_cluster_size = 5
    else:
        min_cluster_size = math.ceil(amount_of_data_points / 20)

    # clustering
    print("clustering")
    df_clusters = calculate_cluster_flat(df_dim_2d, cols=["x", "y"], min_cluster_size=min_cluster_size)
    df_clusters.rename(columns={'doc_id': 'docID', 'cluster_id': 'clusterID', 'han_id': 'hanID'}, inplace=True)
    # cluster naming
    print("cluster naming")
    try:
        df_cluster_names = cluster_naming.generate_cluster_names(df_clusters,
                                                                 content_columns=["title"],
                                                                 cluster_id_column="clusterID",
                                                                 cluster_name_column="clusterName")
    except Exception as e:
        print(e)
        df_cluster_names = df_clusters
        df_cluster_names["clusterName"] = "no cluster"
    # calculate cluster centers
    print("calculate cluster centers")
    grouped = df_cluster_names.groupby('clusterID')
    cluster_centers = grouped[['x', 'y']].mean()
    df_custer_centers = pd.merge(df_cluster_names, cluster_centers, on='clusterID', suffixes=['', 'Center'])

    clusters = df_custer_centers[["clusterID", "xCenter", "yCenter", "clusterName"]].drop_duplicates(
        'clusterID').to_dict(orient="records")
    # generate response
    points = df_clusters[["docID", "x", "y", "clusterID", "hanID", "title"]].to_dict(orient="records")
    response = {"points": points, "clusters": clusters}
    print(response)
    return response


@app.put("/competitors")
def competitors_endpoint(payload: ReportRequestDTO):
    han_ids = payload.hanIds
    print("han_ids: ", han_ids)
    # generate data from request
    # get embeddings
    print("get embeddings")
    df = han_id_embeddings.get_han_id_data_as_dataframe(han_ids)
    if df is None:
        return {"competitors": []}
    # get competitors
    print("get competitors")

    df_result = competitors.get_competitor_by_han_ids_df(df, vector_column_name="vector", amount_of_neighbours=10)

    df_result.rename(
        columns={
            'han_id': 'hanID',
            'han_name': 'hanName',
        },
        inplace=True)
    # generate response
    response = {"competitors": df_result.to_dict(orient="records")}
    print(response)
    return response
