import sql

def get_publn(han_id):
    query = """
SELECT * FROM (SELECT * FROM tls206_person where han_id = '{}') AS pers
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

