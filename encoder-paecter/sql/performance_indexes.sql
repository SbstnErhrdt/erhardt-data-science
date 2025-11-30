-- Recommended supporting indexes for the PAECTER ingestion query.
-- These statements are safe to rerun thanks to IF NOT EXISTS, but consider
-- using CONCURRENTLY in production/non-locking contexts only when appropriate.

-- 1) Speed up the NOT EXISTS lookup against export_embeddings by ensuring a
--    narrow btree index exists on docdb_family_id.
CREATE INDEX IF NOT EXISTS export_embeddings_docdb_family_id_idx
    ON public.export_embeddings (docdb_family_id);

-- 2) Help PostgreSQL evaluate the EP/US authority prioritisation without
--    scanning every array by adding a GIN index over family_authorities. When
--    combined with a query that uses array operators (e.g. @>), this keeps the
--    plan index-friendly after rewriting the CASE predicate.
CREATE INDEX IF NOT EXISTS mv_patent_family_family_authorities_gin_idx
    ON epo_doc_db.mv_patent_family
    USING gin (family_authorities);

-- 3) Provide a partial btree index that only tracks families already meeting
--    the title/abstract requirements. This lets the planner skip rows that
--    would be filtered out early and pairs well with ORDER BY family_id LIMIT.
CREATE INDEX IF NOT EXISTS mv_patent_family_ready_family_id_idx
    ON epo_doc_db.mv_patent_family (family_id)
    WHERE family_id ~ '^[0-9]+$'
      AND title IS NOT NULL
      AND btrim(title) <> ''
      AND abstract IS NOT NULL
      AND btrim(abstract) <> '';
