GRANT USAGE ON SCHEMA epo_doc_db TO encoder;
GRANT SELECT ON TABLE epo_doc_db.mv_patent_family TO encoder;

GRANT USAGE ON SCHEMA public TO encoder;
GRANT INSERT, UPDATE, SELECT ON TABLE public.export_embeddings TO encoder;
  -- add DELETE if the workflow needs it:
  -- GRANT DELETE ON TABLE public.export_embeddings TO encoder;

  -- if export_embeddings has owned sequences (e.g. serial/identity columns), also grant:
  -- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO encoder;
ALTER ROLE encoder WITH PASSWORD 'xxx';
