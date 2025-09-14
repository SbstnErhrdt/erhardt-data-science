import embeddings
import sql
import psycopg2.extras

from publications import get_publications_from_organization


def create_project_organization_sql_table():
    # create project in database
    sql_string = """
    CREATE SEQUENCE project_organizations_id_seq;

    CREATE TABLE IF NOT EXISTS project_organizations (
        id integer NOT NULL DEFAULT nextval('project_organizations_id_seq'),
        project_id integer NOT NULL,
        name text NOT NULL
    );

    ALTER SEQUENCE project_organizations_id_seq
    OWNED BY project_organizations.id;
    """
    sql.cur.execute(sql_string)
    sql.con.commit()
    return


def create_project_organization(project_id: int, project_organization_name: str) -> int:
    # create project in database
    query = "INSERT INTO project_organizations (project_id, name) VALUES (%s, %s) RETURNING id;"
    sql.cur.execute(query, (project_id, project_organization_name,))
    sql.con.commit()
    return_id = sql.cur.fetchone()[0]
    return return_id


def delete_project_organization(project_id: int, project_organization_id: int):
    # delete project from database
    sql.cur.execute("DELETE FROM project_organizations WHERE project_id = %s AND id = %s;",
                    (project_id, project_organization_id))
    sql.con.commit()
    return


def read_project_organizations(project_id: int):
    # read project from database
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT * FROM project_organizations WHERE project_id = %s;", (project_id,))
    sql.con.commit()
    result = cursor.fetchall()
    return result


def get_and_store_embeddings(project_id, project_organization_id):
    """
    Get publications from organization and store embeddings in database
    :param project_id:
    :param project_organization_id:
    :return:
    """

    # Get the publications for the company
    pubs = get_publications_from_organization(project_id, project_organization_id)

    # iterate over the publications and store the embeddings
    for pub in pubs:
        # skip if its A4
        if pub["publn_kind"] == "A4":
            continue
        # build id
        pub_id = pub["publn_auth"] + pub["publn_nr"] + pub["publn_kind"]
        # get the embeddings and store them
        embeddings.get_embedding_of_doc_and_store(pub_id)


if __name__ == '1__main__':
    sql.connect()
    # create_project_organization_sql_table()
    create_project_organization(5, 'sartorius')

    orgs = read_project_organizations(1)
    for org in orgs:
        print(org)

    sql.con.close()
    print('created project organization')

if __name__ == '__main__':
    sql.connect()
    print("get and store embeddings")
    get_and_store_embeddings(5, 9)  # 3,7 siemens
    sql.con.close()
    print('get embeddings done')