import sql


def create_project_organization_han_ids_sql_table():
    # create project in database
    sql_string = """
    CREATE TABLE IF NOT EXISTS project_organization_han_ids (
        project_id integer NOT NULL,
        project_organization_id integer NOT NULL,
        han_id integer NOT NULL,
        PRIMARY KEY (project_id, project_organization_id, han_id)
    );
    """
    sql.cur.execute(sql_string)
    sql.con.commit()
    return


def add_project_organization_han_id(project_id: int, project_organization_id: int, han_id: int) -> int:
    # create project in database
    query = """
    INSERT INTO project_organization_han_ids (project_id, project_organization_id, han_id)
    VALUES (%s, %s, %s) 
    ON CONFLICT DO NOTHING
    RETURNING *;
"""
    sql.cur.execute(query, (project_id, project_organization_id, han_id,))
    sql.con.commit()
    return


def remove_project_organization_han_id(project_id: int, project_organization_id: int, han_id: int):
    # delete project from database
    sql.cur.execute(
        "DELETE FROM project_organization_han_ids WHERE project_id = %s AND project_organization_id = %s AND han_id;",
        (project_id, project_organization_id, han_id))
    sql.con.commit()
    return


if __name__ == '__main__':
    sql.connect()
    create_project_organization_han_ids_sql_table()
    add_project_organization_han_id(1, 1, 324246)
    sql.con.close()
    print('added project organization han id')
