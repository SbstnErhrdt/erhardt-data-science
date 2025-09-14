import sql
import psycopg2.extras


def create_sql_table():
    """
    Create table 3d table
    """
    query = """
    CREATE TABLE IF NOT EXISTS project_doc_3d (
    project_id integer NOT NULL,
    doc_id VARCHAR(255) NOT NULL,
    x float NOT NULL,
    y float NOT NULL,
    z float NOT NULL,
    PRIMARY KEY (project_id, doc_id)
    );
    """
    sql.cur.execute(query)
    sql.con.commit()
    return


def store_3d_of_doc(project_id: int, doc_id: str, x: float, y: float, z: float):
    """
    Store 3d coordinates of document in database
    """
    query = """
    INSERT INTO project_doc_3d (project_id, doc_id, x, y, z)
    VALUES (%(project_id)s, %(doc_id)s, %(x)s, %(y)s, %(z)s)
    ON CONFLICT (project_id, doc_id) DO UPDATE SET x = %(x)s, y = %(y)s, z = %(z)s;
    """
    sql.cur.execute(query, {"project_id": project_id, "doc_id": doc_id, "x": x, "y": y, "z": z})
    sql.con.commit()
    return


def get_3d_of_project(project_id: int):
    """
    Get 3d coordinates of documents in project
    """
    query = """
    SELECT doc_id, x, y, z  FROM project_doc_3d WHERE project_id = %(project_id)s;
    """
    # use dict cursor to get column names
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(query, {"project_id": project_id})
    result = cursor.fetchall()
    return result


if __name__ == '__main__':
    sql.connect()
    create_sql_table()
    store_3d_of_doc(1, "123", 1.0, 2.0, 3.0)
    print(get_3d_of_project(1))
    # close
    sql.con.close()

