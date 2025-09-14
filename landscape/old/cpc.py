from typing import List

import sql
import psycopg2.extras
import pandas as pd

def get_cpc(project_id: int, organization_id: int) -> List[dict]:
    #  sql query
    query = """
    SELECT appln_cpc.cpc_class_symbol as cpc_class_symbol, count(appln_cpc.cpc_class_symbol) as count FROM
    (SELECT * FROM project_organizations WHERE project_id = %s AND id = %s) AS po
    INNER JOIN project_organization_han_ids AS poh ON po.id = poh.project_organization_id
    INNER JOIN mv_publn_pers_owner as publ ON publ.han_id = poh.han_id
    LEFT JOIN tls224_appln_cpc as appln_cpc ON publ.appln_id = appln_cpc.appln_id
    group by appln_cpc.cpc_class_symbol
    order by count desc
    """

    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute(query, (project_id, organization_id,))
    result = cursor.fetchall()
    return result


if __name__ == '__main__':
    sql.connect()
    res = get_cpc(2, 2)
    sql.con.close()
    for r in res:
        print(r["cpc_class_symbol"], r["count"])

    df = pd.DataFrame(res, columns=['cpc', 'count'])
