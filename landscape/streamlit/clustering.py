import pandas as pd
from hdbscan.flat import (HDBSCAN,
                          HDBSCAN_flat)


def calculate_cluster(df, cols=["x", "y", "z"], *args, **kwargs) -> pd.DataFrame:
    """
    Cluster data using hdbscan
    :param df:
    :param cols:
    :param args:
    :param kwargs:
    :return:
    """
    print("cluster: ", cols, args, kwargs)
    clusterer = HDBSCAN(*args, **kwargs)
    clusterer.fit(df[cols].to_numpy())
    df["cluster_id"] = clusterer.labels_
    df["cluster_id"] = df["cluster_id"].astype(str)

    # print amount of clusters
    print("amount clusters: ", len(df["cluster_id"].unique()))
    return df


def calculate_cluster_flat(df, cols=["x", "y", "z"], n_clusters=None, min_cluster_size=10) -> pd.DataFrame:
    """
    Cluster data using hdbscan
    :param df:
    :param cols:
    :param n_clusters:
    :param min_cluster_size:
    :return:
    """
    print("cluster: ", cols, n_clusters, min_cluster_size)
    clusterer = HDBSCAN_flat(
        df[cols].to_numpy(),
        cluster_selection_method='eom',
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size
    )
    df["cluster_id"] = clusterer.labels_
    df["cluster_id"] = df["cluster_id"].astype(str)

    # print amount of clusters
    print("amount clusters: ", len(df["cluster_id"].unique()))
    return df
