import multiprocessing

import pandas as pd
import psycopg2.extras
import requests

import config
import embeddings
import sql
import vector_utils


def get_competitor_ids_by_embedding(vector: [float], amount: 20, store_embedding=False) -> [str]:
    doc_ids = []
    doc_ids_2_embedding = {}

    # build graphql query
    query = """
    query VectorSimilaritySearch($index: String!, $vector: [Float]!, $amount: Int!) {
      VectorSimilaritySearch(vector: $vector, index: $index, amount: $amount) {
        id
        index        
        document {
          vector
          metadata {
            kind
            docNumber
          }
        }
      }
    }
    """

    # build variables
    variables = {"vector": vector, "index": "epo_cos", "amount": amount}

    # send request
    print("send VectorSimilaritySearch request")

    r = requests.post(
        config.LOGIC_MILL_API_ENDPOINT,
        json={'query': query, 'variables': variables},
        headers=config.HEADERS)

    # handle response
    if r.status_code != 200:
        print(f"Error executing\n{query}\n")
    else:
        response = r.json()
        # print(response)

        if "errors" in response:
            print("errors", response["errors"])
            return doc_ids

        if "data" not in response:
            print("no data in response")
            return doc_ids

        if len(response["data"]["VectorSimilaritySearch"]) == 0:
            print("no documents found")
        elif len(response["data"]["VectorSimilaritySearch"]) > 1:
            print("multiple documents found", len(response["data"]["VectorSimilaritySearch"]))

            for d in response["data"]["VectorSimilaritySearch"]:
                doc_id = "EP" + d["document"]["metadata"]["docNumber"] + d["document"]["metadata"]["kind"]
                print("similar_doc", doc_id)
                doc_ids.append(doc_id)
                doc_ids_2_embedding[doc_id] = d["document"]["vector"]
                if store_embedding:
                    embeddings.store_embeddings_of_doc(doc_id, d["document"]["vector"])
                    print(doc_id, "embedding stored")

    return doc_ids


def count_competitors(doc_ids: [str]) -> [dict]:
    print(doc_ids)
    print(len(doc_ids))
    #  sql query
    query = """
       SELECT han_name, han_id, count(*) as count from mv_publn_pers_owner
       WHERE doc_id IN %(doc_ids)s
       GROUP BY han_name, han_id
       ORDER BY count DESC
       """
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, {"doc_ids": tuple(doc_ids)})
    results = cursor.fetchall()
    print("results", results)
    return results


def competitors_worker(queue, results_queue):
    while True:
        vector = queue.get()
        print("process vector", vector[:5])
        res_doc_ids = get_competitor_ids_by_embedding(vector, 10)
        # add result to results queue
        for doc_id in res_doc_ids:
            results_queue.append(doc_id)

        queue.task_done()


def get_competitor_by_han_ids_df(
        df,
        vector_column_name="vector",
        amount_of_neighbours=25,
) -> pd.DataFrame:
    """
    Reduce dimensionality of project data
    :param df: dataframe with project data
    :param vector_column_name: name of column with vector
    :param amount_of_neighbours: amount of competitors to get
    :return:
    """
    # get vector column
    vectors = df[vector_column_name].values
    # convert to numpy array
    _, vectors = vector_utils.get_vector_as_x(vectors)

    # Create a queue to communicate with the worker processes
    queue = multiprocessing.JoinableQueue()
    results = multiprocessing.Manager().list()

    # Create worker processes
    num_workers = 20
    workers = []
    for i in range(num_workers):
        print("Create worker process: " + str(i))
        worker_process = multiprocessing.Process(target=competitors_worker, args=(queue, results))
        worker_process.start()
        workers.append(worker_process)

    # Enqueue items in the queue
    print("Start enqueue items in the queue")
    for vector in vectors:
        queue.put(vector.tolist())
    print("Done enqueue items in the queue")

    # Wait for all tasks to complete
    queue.join()

    print("Done wait for all tasks to complete")
    # Terminate worker processes
    print("Terminate worker processes")
    for worker_process in workers:
        worker_process.terminate()
        worker_process.join()

    # count competitors
    print("Count competitors")
    competitors_dict = count_competitors(results)

    # build dataframe with competitor data
    print("Build dataframe with competitor data")
    df_result = pd.DataFrame(competitors_dict, columns=["han_name", "han_id", "count"])

    return df_result


def get_competitors_stats_by_embedding(vector: [float], amount=10) -> [dict]:
    """
    Get competitors stats by embedding
    :param vector:
    :param amount:
    :return:
    """
    doc_ids = get_competitor_ids_by_embedding(vector, amount=amount, store_embedding=True)
    competitors = count_competitors(doc_ids)
    return competitors
