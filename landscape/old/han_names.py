from typing import List

import sql
import psycopg2
import psycopg2.extras


def get_hand_ids_from_name(name) -> List[dict]:
    #  lowercase name
    name = name.lower()

    #  sql query
    query = """
    SELECT han_id, han_name 
    FROM tls206_person 
    WHERE 
    psn_sector = 'COMPANY'
    AND 
    lower(han_name) like lower('%{}%') 
    GROUP BY han_id, han_name
    ORDER BY han_id ASC
    """.format(name)

    cursor = sql.con.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute(query)
    result = cursor.fetchall()
    return result


if __name__ == '__main__':
    sql.connect()
    res = get_hand_ids_from_name('curevac')
    sql.con.close()
    print(res)
    for r in res:
        print(r["han_id"], r["han_name"])
