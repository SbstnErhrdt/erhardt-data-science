-- Update script to improve mv_patent_family and pending-family query performance.
-- Apply in a maintenance window or use CONCURRENTLY where noted to avoid long locks.

BEGIN;

-- 1) Support the pending-family NOT EXISTS probe.
CREATE INDEX IF NOT EXISTS export_embeddings_docdb_family_id_idx
    ON public.export_embeddings (docdb_family_id);

-- 2) Speed up authority prioritization (EP/US) via array operators.
CREATE INDEX IF NOT EXISTS mv_patent_family_family_authorities_gin_idx
    ON epo_doc_db.mv_patent_family
    USING gin (family_authorities);

-- 3) Narrow candidates to rows with usable title/abstract and numeric family_id.
CREATE INDEX IF NOT EXISTS mv_patent_family_ready_family_id_idx
    ON epo_doc_db.mv_patent_family (family_id)
    WHERE family_id ~ '^[0-9]+$'
      AND title IS NOT NULL
      AND btrim(title) <> ''
      AND abstract IS NOT NULL
      AND btrim(abstract) <> '';

-- 4) Base-table helpers to speed up mv_patent_family refresh. Use CONCURRENTLY in prod.
CREATE INDEX IF NOT EXISTS patents_family_id_idx
    ON epo_doc_db.patents (family_id);

CREATE INDEX IF NOT EXISTS patents_authority_kind_pub_idx
    ON epo_doc_db.patents (authority, kind, publication_date DESC);

CREATE INDEX IF NOT EXISTS patents_nonempty_title_abstract_idx
    ON epo_doc_db.patents (family_id, publication_date DESC, id)
    WHERE title IS NOT NULL
      AND btrim(title) <> ''
      AND abstract IS NOT NULL
      AND btrim(abstract) <> '';

-- Optional: helps unnest(cpc_classes) patterns in the view build.
CREATE INDEX IF NOT EXISTS patents_cpc_classes_gin_idx
    ON epo_doc_db.patents
    USING gin (cpc_classes);

COMMIT;

-- Reminder: after deploying, ANALYZE the affected tables/views.
-- For production, consider CREATE INDEX CONCURRENTLY and run outside peak hours.
