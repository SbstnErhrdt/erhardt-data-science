We want to generate embeddings using the https://huggingface.co/mpi-inno-comp/paecter PAECTER.

```python
from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('mpi-inno-comp/paecter')
model = AutoModel.from_pretrained('mpi-inno-comp/paecter')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```

The embeddings should be stored in the export_embeddings.


```
-- auto-generated definition
create table export_embeddings
(
    docdb_family_id integer,
    embedding       vector(1024)
);

alter table export_embeddings
    owner to postgres;

create index export_embeddings_hnsw
    on export_embeddings using hnsw (embedding vector_cosine_ops);

create index export_embeddings_docdb_family_id_idx
    on export_embeddings (docdb_family_id);
```


The source of the text is the patent table.
However, there are many patents so we need to create a patent family view to reduce the number of records.
We only store english titles and abstracts.

We need both, title and abstract to create the embedding.


Schema: epo_doc_db
```
create table patents
(
    id               text not null
        primary key,
    doc_db_id        text,
    family_id        text,
    authority        varchar(2),
    doc_number       varchar(20),
    kind             varchar(10),
    publication_date date,
    title            text,
    abstract         text,
    cpc_classes      varchar(200)[],
    applicants       text[],
    inventors        text[]
);

alter table patents
    owner to postgres;

create index idx_epo_doc_db_patents_kind
    on patents (kind);

create index idx_epo_doc_db_patents_doc_number
    on patents (doc_number);

create index idx_epo_doc_db_patents_authority
    on patents (authority);

create index idx_epo_doc_db_patents_family_id
    on patents (family_id);

create index idx_epo_doc_db_patents_doc_db_id
    on patents (doc_db_id);

create index patents_title_idx
    on patents (family_id, epo_doc_db.authority_prio(authority::text), publication_date, doc_number,
                id) include (title, authority)
    where ((title IS NOT NULL) AND (length(btrim(title)) > 0));

create table state
(
    file_path  varchar(2048) not null
        primary key,
    done_at    timestamp with time zone,
    created_at timestamp with time zone,
    updated_at timestamp with time zone,
    deleted_at timestamp with time zone
);

alter table state
    owner to postgres;

create index idx_state_deleted_at
    on state (deleted_at);

create index idx_state_done_at
    on state (done_at);

create materialized view mv_patent_family as
WITH scored AS (SELECT p.id,
                       p.doc_db_id,
                       p.family_id,
                       p.authority,
                       p.doc_number,
                       p.kind,
                       p.publication_date,
                       p.title,
                       p.abstract,
                       p.cpc_classes,
                       p.applicants,
                       p.inventors,
                       CASE upper(COALESCE(p.authority, ''::character varying)::text)
                           WHEN 'EP'::text THEN 1
                           WHEN 'WO'::text THEN 2
                           WHEN 'US'::text THEN 3
                           WHEN 'CN'::text THEN 4
                           WHEN 'JP'::text THEN 5
                           WHEN 'KR'::text THEN 6
                           WHEN 'GB'::text THEN 7
                           WHEN 'CA'::text THEN 8
                           WHEN 'TW'::text THEN 9
                           WHEN 'DE'::text THEN 10
                           WHEN 'ES'::text THEN 11
                           WHEN 'RU'::text THEN 12
                           WHEN 'AU'::text THEN 13
                           WHEN 'FR'::text THEN 14
                           WHEN 'MX'::text THEN 15
                           WHEN 'UA'::text THEN 16
                           WHEN 'NZ'::text THEN 17
                           ELSE 99
                           END AS authority_priority,
                       CASE
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                p.kind::text = 'B2'::text THEN 1
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                p.kind::text = 'B3'::text THEN 2
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                p.kind::text = 'B1'::text THEN 3
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                p.kind::text = 'A1'::text THEN 4
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                p.kind::text = 'A3'::text THEN 5
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                p.kind::text = 'A2'::text THEN 6
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'EP'::text AND
                                (p.kind::text = ANY
                                 (ARRAY ['B9'::character varying, 'B8'::character varying, 'A9'::character varying, 'A8'::character varying]::text[]))
                               THEN 9
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'WO'::text AND
                                p.kind::text = 'A1'::text THEN 1
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'WO'::text AND
                                p.kind::text = 'A3'::text THEN 2
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'WO'::text AND
                                p.kind::text = 'A2'::text THEN 3
                           WHEN upper(COALESCE(p.authority, ''::character varying)::text) = 'WO'::text AND
                                p.kind::text = 'A4'::text THEN 4
                           WHEN p.kind::text ~* '^[B]'::text THEN 1
                           WHEN p.kind::text = 'A1'::text THEN 2
                           WHEN p.kind::text ~* '^[A]'::text THEN 3
                           ELSE 99
                           END AS kind_priority,
                       CASE
                           WHEN NULLIF(btrim(p.title), ''::text) IS NOT NULL AND
                                NULLIF(btrim(p.abstract), ''::text) IS NOT NULL THEN 0
                           WHEN NULLIF(btrim(p.title), ''::text) IS NOT NULL OR
                                NULLIF(btrim(p.abstract), ''::text) IS NOT NULL THEN 1
                           ELSE 2
                           END AS content_priority
                FROM epo_doc_db.patents p),
     rep AS (SELECT DISTINCT ON (scored.family_id) scored.family_id,
                                                   scored.id               AS rep_id,
                                                   scored.authority        AS rep_authority,
                                                   scored.kind             AS rep_kind,
                                                   scored.publication_date AS rep_pub_date,
                                                   scored.title            AS rep_title,
                                                   scored.abstract         AS rep_abstract
             FROM scored
             ORDER BY scored.family_id, scored.authority_priority, scored.kind_priority, scored.content_priority,
                      scored.publication_date DESC NULLS LAST, scored.doc_number, scored.id),
     members AS (SELECT scored.family_id,
                        array_agg(scored.id ORDER BY scored.publication_date, scored.id) AS family_members
                 FROM scored
                 GROUP BY scored.family_id),
     auths AS (SELECT x.family_id,
                      array_agg(x.auth ORDER BY x.rank, x.auth) AS family_authorities
               FROM (SELECT DISTINCT s.family_id,
                                     s.authority AS auth,
                                     CASE upper(COALESCE(s.authority, ''::character varying)::text)
                                         WHEN 'EP'::text THEN 1
                                         WHEN 'WO'::text THEN 2
                                         WHEN 'US'::text THEN 3
                                         WHEN 'CN'::text THEN 4
                                         WHEN 'JP'::text THEN 5
                                         WHEN 'KR'::text THEN 6
                                         WHEN 'GB'::text THEN 7
                                         WHEN 'CA'::text THEN 8
                                         WHEN 'TW'::text THEN 9
                                         WHEN 'DE'::text THEN 10
                                         WHEN 'ES'::text THEN 11
                                         WHEN 'RU'::text THEN 12
                                         WHEN 'AU'::text THEN 13
                                         WHEN 'FR'::text THEN 14
                                         WHEN 'MX'::text THEN 15
                                         WHEN 'UA'::text THEN 16
                                         WHEN 'NZ'::text THEN 17
                                         ELSE 99
                                         END     AS rank
                     FROM scored s
                     WHERE s.authority IS NOT NULL) x
               GROUP BY x.family_id),
     dates AS (SELECT scored.family_id,
                      min(scored.publication_date)
                      FILTER (WHERE scored.kind::text ~ '^[A]'::text)                              AS first_application_pub_date,
                      min(scored.publication_date)
                      FILTER (WHERE scored.kind::text ~ '^[B]'::text)                              AS first_grant_pub_date
               FROM scored
               GROUP BY scored.family_id),
     cpcs AS (SELECT u.family_id,
                     COALESCE(array_agg(DISTINCT u.c ORDER BY u.c),
                              ARRAY []::character varying[]::character varying(200)[]) AS cpc
              FROM (SELECT scored.family_id,
                           unnest(scored.cpc_classes)::character varying(200) AS c
                    FROM scored
                    WHERE scored.cpc_classes IS NOT NULL) u
              GROUP BY u.family_id),
     apps AS (SELECT u.family_id,
                     COALESCE(array_agg(DISTINCT u.a ORDER BY u.a), ARRAY []::text[]) AS applicants
              FROM (SELECT scored.family_id,
                           unnest(scored.applicants) AS a
                    FROM scored
                    WHERE scored.applicants IS NOT NULL) u
              GROUP BY u.family_id),
     invs AS (SELECT u.family_id,
                     COALESCE(array_agg(DISTINCT u.i ORDER BY u.i), ARRAY []::text[]) AS inventors
              FROM (SELECT scored.family_id,
                           unnest(scored.inventors) AS i
                    FROM scored
                    WHERE scored.inventors IS NOT NULL) u
              GROUP BY u.family_id),
     best_title AS (SELECT scored.family_id,
                           (array_agg(scored.title
                                      ORDER BY scored.authority_priority, scored.kind_priority, scored.publication_date DESC NULLS LAST, scored.doc_number, scored.id))[1] AS best_title
                    FROM scored
                    WHERE NULLIF(btrim(scored.title), ''::text) IS NOT NULL
                    GROUP BY scored.family_id),
     best_abstract AS (SELECT scored.family_id,
                              (array_agg(scored.abstract
                                         ORDER BY scored.authority_priority, scored.kind_priority, scored.publication_date DESC NULLS LAST, scored.doc_number, scored.id))[1] AS best_abstract
                       FROM scored
                       WHERE NULLIF(btrim(scored.abstract), ''::text) IS NOT NULL
                       GROUP BY scored.family_id)
SELECT r.family_id,
       m.family_members,
       a.family_authorities,
       COALESCE(NULLIF(btrim(r.rep_title), ''::text), bt.best_title)            AS title,
       COALESCE(NULLIF(btrim(r.rep_abstract), ''::text), ba.best_abstract)      AS abstract,
       COALESCE(c.cpc, ARRAY []::character varying[]::character varying(200)[]) AS cpc,
       COALESCE(ap.applicants, ARRAY []::text[])                                AS applicants,
       COALESCE(iv.inventors, ARRAY []::text[])                                 AS inventors,
       d.first_application_pub_date,
       d.first_grant_pub_date,
       r.rep_id,
       r.rep_authority,
       r.rep_kind,
       r.rep_pub_date
FROM rep r
         LEFT JOIN members m USING (family_id)
         LEFT JOIN auths a USING (family_id)
         LEFT JOIN cpcs c USING (family_id)
         LEFT JOIN apps ap USING (family_id)
         LEFT JOIN invs iv USING (family_id)
         LEFT JOIN dates d USING (family_id)
         LEFT JOIN best_title bt USING (family_id)
         LEFT JOIN best_abstract ba USING (family_id);

alter materialized view mv_patent_family owner to postgres;

create unique index mv_patent_family_family_id_idx
    on mv_patent_family (family_id);

create index mv_patent_family_search_idx
    on mv_patent_family using gin (to_tsvector('english'::regconfig, (COALESCE(title, ''::text) || ' '::text) ||
                                                                     COALESCE(abstract, ''::text)));

create materialized view mv_patent_applicants_family as
WITH exploded AS (SELECT f.family_id,
                         a.a                                      AS applicant_raw,
                         epo_doc_db.normalize_applicant_name(a.a) AS applicant_norm
                  FROM epo_doc_db.mv_patent_family f
                           CROSS JOIN LATERAL unnest(f.applicants) a(a)),
     dedup AS (SELECT exploded.family_id,
                      exploded.applicant_norm,
                      array_agg(DISTINCT exploded.applicant_raw ORDER BY exploded.applicant_raw) AS raw_variants
               FROM exploded
               WHERE exploded.applicant_norm IS NOT NULL
                 AND btrim(exploded.applicant_norm) <> ''::text
               GROUP BY exploded.family_id, exploded.applicant_norm)
SELECT dedup.applicant_norm AS applicant,
       dedup.family_id,
       dedup.raw_variants
FROM dedup;

alter materialized view mv_patent_applicants_family owner to postgres;

create unique index mv_patent_applicants_family_applicant_family_idx
    on mv_patent_applicants_family (applicant, family_id);

create index mv_patent_applicants_family_applicant_idx
    on mv_patent_applicants_family (applicant);

create index mv_patent_applicants_family_family_id_idx
    on mv_patent_applicants_family (family_id);

create materialized view mv_applicant_candidates as
WITH base AS (SELECT mv_patent_applicants_family.applicant,
                     mv_patent_applicants_family.family_id,
                     mv_patent_applicants_family.raw_variants
              FROM epo_doc_db.mv_patent_applicants_family),
     counts AS (SELECT base.applicant,
                       count(*)::integer AS family_count
                FROM base
                GROUP BY base.applicant),
     vars AS (SELECT u.applicant,
                     array_agg(DISTINCT u.v ORDER BY u.v) AS variants
              FROM (SELECT base.applicant,
                           unnest(base.raw_variants) AS v
                    FROM base) u
              GROUP BY u.applicant)
SELECT c.applicant,
       c.family_count,
       v.variants
FROM counts c
         JOIN vars v USING (applicant)
ORDER BY c.family_count DESC;

alter materialized view mv_applicant_candidates owner to postgres;

create index mv_applicant_candidates_applicant_idx
    on mv_applicant_candidates (applicant);

create index mv_applicant_candidates_applicant_trgm_idx
    on mv_applicant_candidates using gin (applicant public.gin_trgm_ops);

create index mv_applicant_candidates_family_count_idx
    on mv_applicant_candidates (family_count desc);

create function authority_prio(auth text) returns smallint
    immutable
    strict
    parallel safe
    language sql
as
$$
SELECT CASE auth
           WHEN 'EP' THEN 1 WHEN 'WO' THEN 2 WHEN 'US' THEN 3
           WHEN 'CN' THEN 4 WHEN 'JP' THEN 5 WHEN 'KR' THEN 6
           WHEN 'GB' THEN 7 WHEN 'CA' THEN 8 WHEN 'TW' THEN 9
           WHEN 'DE' THEN 10 WHEN 'ES' THEN 11 WHEN 'RU' THEN 12
           WHEN 'AU' THEN 13 WHEN 'FR' THEN 14 WHEN 'MX' THEN 15
           WHEN 'UA' THEN 16 WHEN 'NZ' THEN 17 ELSE 999 END;
$$;

alter function authority_prio(text) owner to postgres;

create function normalize_applicant_name(name_in text) returns text
    immutable
    strict
    language plpgsql
as
$$
DECLARE
    s text;
BEGIN
    s := btrim(name_in);
    IF s IS NULL OR s = '' THEN
        RETURN NULL;
    END IF;

    -- case-fold + remove accents
    s := unaccent(lower(s));

    -- normalize symbols
    s := regexp_replace(s, '&', ' and ', 'g');

    -- remove common corporate/legal suffixes (word-boundary aware: \m ... \M)
    s := regexp_replace(
            s,
            '(\m(?:inc(?:orporated)?|corp(?:oration)?|company|co\.?[[:space:]]*ltd\.?|co|ltd|limited'
                || '|llc|llp|plc|gmbh|mbh|ag|kg|ohg'
                || '|sarl|sas|sa|bv|nv|oy|oyj|ab|as'
                || '|spa|s\.p\.a|bvba|aps|kft'
                || '|sro|s\.r\.o\.|sp\.?[[:space:]]*z\.?[[:space:]]*o\.?[[:space:]]*o\.?'
                || '|zao|oao|ooo'
                || '|pte|pte\.?[[:space:]]*ltd|ltda'
                || '|sl|s\.l\.|slu'
                || '|kk|k\.k\.|kabushiki[[:space:]]+kaisha'
                || '|holdings?|group'
                || '|sa[[:space:]]*de[[:space:]]*cv|s\.a\.[[:space:]]*de[[:space:]]*cv'
                || ')\M)',
            ' ',
            'gi'   -- g = global, i = case-insensitive
         );

    -- drop leading "the "
    s := regexp_replace(s, '^(?:the)[[:space:]]+', '', 'i');

    -- strip remaining punctuation (keep letters/digits/space)
    s := regexp_replace(s, '[^[:alnum:][:space:]]+', ' ', 'g');

    -- collapse whitespace
    s := regexp_replace(s, '[[:space:]]+', ' ', 'g');
    s := btrim(s);

    IF s = '' THEN
        RETURN NULL;
    END IF;

    RETURN s;
END;
$$;

alter function normalize_applicant_name(text) owner to postgres;
```

Write a python cron job to generate the embeddings for all missing embeddings of patent families and store them in the export_embeddings table.
Make use of an env file with the database connection parameters.