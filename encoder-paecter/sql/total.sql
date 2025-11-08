  WITH filtered AS (
      SELECT family_id::bigint AS family_id
      FROM epo_doc_db.mv_patent_family
      WHERE family_id ~ '^[0-9]+$'
        AND title IS NOT NULL
        AND btrim(title) <> ''
        AND abstract IS NOT NULL
        AND btrim(abstract) <> ''
  )
  SELECT now() as ts , COUNT(*) AS families_pending
  FROM filtered f
  WHERE NOT EXISTS (
      SELECT 1
      FROM export_embeddings e
      WHERE e.docdb_family_id = f.family_id
  );


-- 65310714
-- 65308218
-- 65220282