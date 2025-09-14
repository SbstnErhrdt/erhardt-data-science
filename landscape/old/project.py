import pandas as pd
import psycopg2.extras

import competitors
import project_organization
import sql
from dimensionality_reduction import get_vector_as_x
import project_organization_competitor_document


def create_project_sql_table():
    # create project in database
    sql_string = """
    CREATE SEQUENCE projects_id_seq;

    CREATE TABLE IF NOT EXISTS projects (
        id integer NOT NULL DEFAULT nextval('projects_id_seq'),
        name text NOT NULL
    );
    
    ALTER SEQUENCE projects_id_seq
    OWNED BY projects.id;
    """
    sql.cur.execute(sql_string)
    sql.con.commit()
    return


def create_project(project_name: str) -> int:
    # create project in database
    query = "INSERT INTO projects (name) VALUES (%s) RETURNING id;"
    sql.cur.execute(query, (project_name,))
    sql.con.commit()
    return_id = sql.cur.fetchone()[0]
    return return_id


def delete_project(project_id: int):
    # delete project from database
    sql.cur.execute("DELETE FROM projects WHERE id = %s;", (project_id,))
    sql.con.commit()
    return


def get_and_store_all_embeddings(project_id: int):
    """
    Get all organizations from project and store embeddings in database
    :param project_id:
    :return:
    """
    organizations = project_organization.read_project_organizations(project_id)
    for organization in organizations:
        project_organization.get_and_store_embeddings(project_id, organization['id'])
    return


def get_project_data_as_dataframe(project_id: int):
    """
    Get all organizations from project and store embeddings in database
    :param project_id:
    :return:
    """
    query = """
    SELECT DISTINCT ON (publ.doc_id) * FROM projects AS p
    INNER JOIN project_organizations AS po ON p.id = po.project_id
    INNER JOIN project_organization_han_ids AS poi ON po.id = poi.project_organization_id
    INNER JOIN mv_publn_pers_owner AS publ ON publ.han_id = poi.han_id
    INNER JOIN embeddings AS e ON publ.doc_id = e.doc_id
    LEFT JOIN tls202_appln_title AS t ON t.appln_id = publ.appln_id
    LEFT JOIN tls203_appln_abstr AS a ON a.appln_id = publ.appln_id
    WHERE p.id = %s;
    """
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(query, (project_id,))
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    return df

def get_competitor_data_as_dataframe(project_id: int, amount_of_competitors=10):
    df = get_project_data_as_dataframe(project_id)
    # get vector column
    vectors = df["vector"].values
    # convert to numpy array
    missing_data_rows, x = get_vector_as_x(vectors)
    # remove rows with missing data
    df = df.drop(missing_data_rows, axis=0)

    # iterate over all rows and get competitor data
    competitor_ids = []
    for index, row in df.iterrows():
        c_ids = competitors.get_competitor_ids_by_embedding(vectors[index], amount=amount_of_competitors, store_embedding=True)
        competitor_ids.append(c_ids)
        # add ids
        for c_id in c_ids:
            project_organization_competitor_document.add_project_organization_competitor_document(project_id, row['project_organization_id'],
                                                                                                  c_id)

    # remove duplicates from list
    competitor_ids = list(set([item for sublist in competitor_ids for item in sublist]))

    # get competitor data
    query = """
       SELECT DISTINCT ON (publ.doc_id) * FROM mv_publn_pers_owner AS publ
       INNER JOIN embeddings AS e ON publ.doc_id = e.doc_id
       LEFT JOIN tls202_appln_title AS t ON t.appln_id = publ.appln_id
       LEFT JOIN tls203_appln_abstr AS a ON a.appln_id = publ.appln_id
       WHERE publ.doc_id IN %(doc_ids)s;
       """
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(query, {"doc_ids": tuple(competitor_ids)})
    result = cursor.fetchall()
    df_result = pd.DataFrame(result)

    return df_result


if __name__ == '__main__':
    sql.connect()
    # create_project_sql_table()
    create_project('sartorius')
    # get_and_store_all_embeddings(2)
    # df_competitors = get_competitor_data_as_dataframe(2)
    sql.close()
    print('created project')
