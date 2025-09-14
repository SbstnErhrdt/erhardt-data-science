import sql


def create_materialized_view_publ_owner():
    """
    Create materialized view publ_owner
    """
    query = """
    CREATE MATERIALIZED VIEW mv_ep_publn_pers_owner AS
    SELECT CONCAT(publn.publn_auth, publn.publn_nr, publn.publn_kind) as doc_id, publn.*, pers.* FROM (SELECT * FROM tls211_pat_publn as a WHERE a.publn_auth = 'EP') as publn
    INNER JOIN tls207_pers_appln pers_appln on publn.appln_id = pers_appln.appln_id
    INNER JOIN tls206_person pers on pers_appln.person_id = pers.person_id
    WHERE pers_appln.invt_seq_nr = 0;
    """

    sql.cur.execute(query)
    sql.con.commit()
    return


if __name__ == '__main__':
    sql.connect()
    create_materialized_view_publ_owner()
    sql.con.close()