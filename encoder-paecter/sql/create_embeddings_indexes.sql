-- Index to speed up detection of families missing embeddings.
-- Run with: psql -f encoder-paecter/sql/create_embeddings_indexes.sql

CREATE INDEX IF NOT EXISTS export_embeddings_docdb_family_id_idx
    ON export_embeddings (docdb_family_id);

CREATE INDEX IF NOT EXISTS mv_patent_family_ready_idx
    ON epo_doc_db.mv_patent_family ((family_id::bigint))
    WHERE family_id ~ '^[0-9]+$'
      AND title IS NOT NULL
      AND btrim(title) <> ''
      AND abstract IS NOT NULL
      AND btrim(abstract) <> '';
