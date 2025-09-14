from typing import List, Optional

import pandas as pd
import psycopg2.extras

import sql


def get_han_id_data_as_dataframe(han_ids: List[str]) -> Optional[pd.DataFrame]:
    """
    :param han_ids:
    :return:
    """
    sql.get_db()
    query = """
    SELECT DISTINCT ON (publ.doc_id) publ.doc_id as doc_id, publ.title as title, e.vector as vector, publ.han_id as han_id  
    FROM mv_publn_pers_owner AS publ
    INNER JOIN embeddings AS e ON publ.doc_id = e.doc_id
    WHERE publ.han_id IN %s;
    """
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(query, (tuple(han_ids),))
    result = cursor.fetchall()
    print("results", len(result))
    if len(result) == 0:
        return None
    cursor.close()
    df = pd.DataFrame(result)
    df['han_id'] = df['han_id'].astype("string")
    return df


if __name__ == '__main__':
    sql.connect()
    print("get embeddings")
    df_test = get_han_id_data_as_dataframe(["2697604"])
    print(df_test)
    sql.con.close()
    print('get embeddings done')
