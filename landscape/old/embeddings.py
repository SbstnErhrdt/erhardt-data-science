import requests
from retry import retry

import config
import sql


def get_embedding_and_metadata_of_doc_from_logic_mill(doc_id: str):
    """
    Get the embedding of a document from logic mill.
    :param doc_id:
    :return:
    """
    result = {}

    # build graphql query
    query = """
    query searchDocuments($index: String!, $keyword: String!) {
      searchDocuments(index: $index, keyword: $keyword) {
        id
        documentParts {
          abstract
          title
        }
        metadata {
          createdAt
        }
        vector
      }
    }
    """

    # build variables
    variables = {"keyword": doc_id, "index": "epo_cos"}

    # send request
    r = requests.post(
        config.LOGIC_MILL_API_ENDPOINT,
        json={'query': query, 'variables': variables},
        headers=config.HEADERS)

    # handle response
    if r.status_code != 200:
        print(f"Error executing\n{query}\n")
    else:
        response = r.json()
        print(response)

        if len(response["data"]["searchDocuments"]) == 0:
            print("no embedding for", doc_id)
        elif len(response["data"]["searchDocuments"]) > 1:
            print("multiple embeddings found for", doc_id, len(response["data"]["searchDocuments"]))

            result["vector"] = response["data"]["searchDocuments"][0]["vector"]
            result["title"] = response["data"]["searchDocuments"][0].get('documentParts').get('title') or "No title"
            result["abstract"] = response["data"]["searchDocuments"][0].get('documentParts').get(
                'abstract') or "No abstract"

        elif len(response["data"]["searchDocuments"][0]["vector"]) == 0:
            print("embedding with len 0", doc_id)
        else:
            print("one embedding found for", doc_id)

            result["vector"] = response["data"]["searchDocuments"][0]["vector"]
            result["title"] = response["data"]["searchDocuments"][0].get('documentParts').get('title') or "No title"
            result["abstract"] = response["data"]["searchDocuments"][0].get('documentParts').get(
                'abstract') or "No abstract"

        return result


def create_sql_embedding_table():
    """
    Creates the sql table for the embeddings.
    :return:
    """
    query = """
    CREATE TABLE IF NOT EXISTS embeddings (
        doc_id VARCHAR(20) PRIMARY KEY,
        vector FLOAT[]
    );
    """
    cursor = sql.con.cursor()
    cursor.execute(query)
    sql.con.commit()


@retry()
def store_embeddings_of_doc(doc_id: str, embedding: [float]):
    """
    Stores the embedding of a document in the sql database.
    :param doc_id:
    :param embedding:
    :return:
    """
    query = """
    INSERT INTO embeddings (doc_id, vector)
    VALUES (%(doc_id)s, %(vector)s)
    ON CONFLICT (doc_id) DO UPDATE SET vector = %(vector)s;
    """
    cursor = sql.con.cursor()
    cursor.execute(query, {"doc_id": doc_id, "vector": embedding})
    sql.con.commit()


def check_if_already_exists(doc_id: str) -> bool:
    """
    Checks if the embedding of a document already exists in the database.
    :param doc_id:
    :return:
    """
    query = """
    SELECT doc_id FROM embeddings WHERE doc_id = %(doc_id)s;
    """
    cursor = sql.con.cursor()
    cursor.execute(query, {"doc_id": doc_id})
    result = cursor.fetchone()
    return bool(result)


def retrieve_embedding_of_doc(doc_id: str):
    """
    Retrieves the embedding of a document from the sql database.
    :param doc_id:
    :return:
    """
    query = """
    SELECT vector FROM embeddings WHERE doc_id = %(doc_id)s;
    """
    cursor = sql.con.cursor()
    cursor.execute(query, {"doc_id": doc_id})
    result = cursor.fetchone()
    return result


def get_embedding_of_doc_and_store(doc_id: str):
    """
    Get the embedding of a document from logic mill and stores it in the database.
    :param doc_id:
    :return:
    """
    if check_if_already_exists(doc_id):
        print("already exists", doc_id)
        # get embedding from database
        embedding = retrieve_embedding_of_doc(doc_id)
        return embedding
    try:
        embedding = get_embedding_and_metadata_of_doc_from_logic_mill(doc_id)
        # check if embedding is empty
        if embedding:
            store_embeddings_of_doc(doc_id, embedding["vector"])
            return embedding
        else:
            return None
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    sql.connect()
    create_sql_embedding_table()

    data = get_embedding_and_metadata_of_doc_from_logic_mill('EP1818409A1')
    print(data)

    exists = check_if_already_exists('HELLOWORLD')
    assert exists is False
    print(exists, "should be false")

    store_embeddings_of_doc('EP1818409A1', data['vector'])

    exists = check_if_already_exists('EP1818409A1')
    assert exists is True

    # get embedding from sql
    res = retrieve_embedding_of_doc('EP1818409A1')
    # compare with embedding from logicmill
    assert res[0] == data['vector']
    print(res, data['vector'])
    sql.close()
