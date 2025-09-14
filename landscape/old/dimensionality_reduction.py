import pandas as pd
import umap

import vector_utils


def reduce_dimensionality_of_project_data(
        df,
        vector_column_name="vector",
        n_components=3,
        n_neighbors=3,
        min_dist=0.1,
        metric="euclidean"
) -> pd.DataFrame:
    """
    Reduce dimensionality of project data
    :param df: dataframe with project data
    :param vector_column_name: name of column with vector
    :param n_components: number of components / dimensions
    :param n_neighbors: parameter controls how UMAP balances local versus global structure in the data
    :param min_dist: minimum distance
    :param metric: metric to use in UMAP
    :return:
    """
    # get vector column
    vectors = df[vector_column_name].values
    # convert to numpy array
    missing_data_rows, x = vector_utils.get_vector_as_x(vectors)
    # remove rows with missing data
    df = df.drop(missing_data_rows, axis=0)
    print(x.shape)
    # reduce dimensionality
    print('start reducing dimensionality')
    dim_reduction_result = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric
    ).fit_transform(x)
    print('done reduced dimensionality')
    # add dimension reduction result to dataframe
    dims_names = ["x", "y", "z"]
    for i in range(n_components):
        df[dims_names[i]] = dim_reduction_result[:, i]

    return df
