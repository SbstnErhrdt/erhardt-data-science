import sql


def create_project_organization_competitor_documents_sql_table():
    # create project in database
    sql_string = """
    CREATE TABLE IF NOT EXISTS project_organization_competitor_document_ids (
        project_id integer NOT NULL,
        project_organization_id integer NOT NULL,
        project_organization_competitor_document_id varchar(100) NOT NULL,
        PRIMARY KEY (project_id, project_organization_id, project_organization_competitor_document_id)
    );
    """
    sql.cur.execute(sql_string)
    sql.con.commit()
    return


def add_project_organization_competitor_document(project_id: int, project_organization_id: int,
                                                 project_organization_competitor_document_id: str) -> int:
    # create project in database
    query = """
    INSERT INTO project_organization_competitor_document_ids 
    (project_id, project_organization_id, project_organization_competitor_document_id)
    VALUES (%s, %s, %s) 
    ON CONFLICT DO NOTHING
    RETURNING *;
    """
    sql.cur.execute(query, (project_id, project_organization_id, project_organization_competitor_document_id,))
    sql.con.commit()
    return


def remove_project_organization_competitor_document(project_id: int, project_organization_id: int,
                                                    project_organization_competitor_document_id: str):
    # delete project from database
    query = """
        DELETE FROM project_organization_competitor_document_ids
        WHERE project_id = %s 
        AND project_organization_id = %s 
        AND project_organization_competitor_document_id = %s;
        """
    sql.cur.execute(query,
                    (project_id, project_organization_id, project_organization_competitor_document_id))
    sql.con.commit()
    return


if __name__ == '__main__':
    sql.connect()
    create_project_organization_competitor_documents_sql_table()
    add_project_organization_competitor_document(1, 1, "hello world")
    sql.con.close()
    print('added project organization competitor document id')
