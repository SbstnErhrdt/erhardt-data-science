from typing import List

import pandas as pd
import sql
import psycopg2.extras


def export_pupln(name):
    query = """
SELECT * FROM (SELECT * FROM tls206_person where lower(han_name) like '%{}%') AS pers
    LEFT JOIN tls227_pers_publn as pp ON pers.person_id = pp.person_id
INNER JOIN tls211_pat_publn as publ on pp.pat_publn_id = publ.pat_publn_id
LEFT JOIN tls201_appln as appln ON publ.appln_id = appln.appln_id
LEFT JOIN tls202_appln_title as appln_title ON publ.appln_id = appln_title.appln_id
LEFT JOIN tls203_appln_abstr as appln_abstract ON publ.appln_id = appln_abstract.appln_id
WHERE publ.publn_auth = 'EP'
;
    """.format(name)
    df = pd.read_sql(query, con=sql.con)
    df.to_csv(name + "_publn.csv")

    return


def get_publications_from_organization(project_id: int, organization_id: int) -> List[dict]:
    #  sql query
    query = """
    SELECT * FROM
    (SELECT * FROM project_organizations WHERE project_id = %s AND id = %s) AS po
    INNER JOIN project_organization_han_ids AS poh ON po.id = poh.project_organization_id
    INNER JOIN mv_publn_pers_owner as publ ON publ.han_id = poh.han_id
    ORDER BY RANDOM ()
    """
    print("get publications from organization")
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, (project_id, organization_id,))
    print("get publications from organization")
    result = cursor.fetchall()
    return result


def get_publications_from_han_ids(hanIds: List[str]) -> List[dict]:
    #  sql query
    query = """
        SELECT * FROM        
        mv_publn_pers_owner as publ 
        WHERE publ.han_id IN %s
        """
    print("get publications from organization")
    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, (hanIds,))
    print("get publications from organization")
    result = cursor.fetchall()
    return result



if __name__ == '__main__':
    sql.connect()
    res = get_publications_from_organization(5, 9)
    sql.con.close()
    print(res)
    for r in res:
        print(r["han_id"], r["han_name"], r["publn_auth"], r["publn_kind"], r["publn_nr"])