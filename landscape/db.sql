create table ip_landscapes
(
    uid            uuid      default gen_random_uuid() not null
        primary key,
    client_uid     uuid                                not null
        references clients
            on delete cascade,
    created_at     timestamp default now(),
    created_by     uuid      default auth.uid()        not null
        references auth.users
            on delete set null,
    name           varchar(255)                        not null,
    description    text,
    memo           jsonb,
    search_vectors jsonb,
    _status        varchar(255),
    _progress      smallint  default 0,
    _output        jsonb,
    _message       text
);

alter table ip_landscapes
    owner to postgres;

create policy "Users can create ip landscapes" on ip_landscapes
    as permissive
    for insert
    with check ((created_by = auth.uid()) AND f_has_client_access(client_uid));

create policy "Users can read ip landscapes" on ip_landscapes
    as permissive
    for select
    using (f_has_client_access(client_uid) OR f_is_super_admin());

create policy "Users can update ip landscapes" on ip_landscapes
    as permissive
    for update
    using (f_has_client_access(client_uid) OR f_is_super_admin());

create policy "Users can delete ip landscapes" on ip_landscapes
    as permissive
    for delete
    using ((f_has_client_access(client_uid) AND (created_by = auth.uid())) OR f_is_super_admin());

grant delete, insert, references, select, trigger, truncate, update on ip_landscapes to anon;

grant delete, insert, references, select, trigger, truncate, update on ip_landscapes to authenticated;

grant delete, insert, references, select, trigger, truncate, update on ip_landscapes to service_role;

--

create table ip_landscape_items
(
    landscape_uid uuid                         not null
        references ip_landscapes
            on delete cascade,
    patent_id     varchar(50)                  not null
        references epo_doc_db.patents
            on delete set null,
    client_uid    uuid                         not null
        references clients
            on delete cascade,
    created_at    timestamp default now(),
    created_by    uuid      default auth.uid() not null
        references auth.users
            on delete set null,
    primary key (landscape_uid, patent_id)
);

alter table ip_landscape_items
    owner to postgres;

create policy "Users can create ip landscape items" on ip_landscape_items
    as permissive
    for insert
    with check ((created_by = auth.uid()) AND f_has_client_access(client_uid));

create policy "Users can read ip landscape items" on ip_landscape_items
    as permissive
    for select
    using (f_has_client_access(client_uid) OR f_is_super_admin());

create policy "Users can update ip landscape items" on ip_landscape_items
    as permissive
    for update
    using (f_has_client_access(client_uid) OR f_is_super_admin());

create policy "Users can delete ip landscape items" on ip_landscape_items
    as permissive
    for delete
    using ((f_has_client_access(client_uid) AND (created_by = auth.uid())) OR f_is_super_admin());

grant delete, insert, references, select, trigger, truncate, update on ip_landscape_items to anon;

grant delete, insert, references, select, trigger, truncate, update on ip_landscape_items to authenticated;

grant delete, insert, references, select, trigger, truncate, update on ip_landscape_items to service_role;

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

grant delete, insert, references, select, trigger, truncate, update on export_embeddings to anon;

grant delete, insert, references, select, trigger, truncate, update on export_embeddings to authenticated;

grant delete, insert, references, select, trigger, truncate, update on export_embeddings to service_role;

-- RPC: vector similarity search over export_embeddings
-- Returns the nearest neighbors by cosine distance using HNSW index
create or replace function public.match_export_embeddings(
    query_embedding vector(1024),
    match_count int
)
returns table(
    docdb_family_id int,
    distance double precision
)
language sql
stable
as $$
    select e.docdb_family_id,
           (e.embedding <=> query_embedding) as distance
    from export_embeddings e
    order by e.embedding <=> query_embedding
    limit match_count;
$$;

grant execute on function public.match_export_embeddings(vector(1024), int) to anon;
grant execute on function public.match_export_embeddings(vector(1024), int) to authenticated;
grant execute on function public.match_export_embeddings(vector(1024), int) to service_role;

create view ip_patent_families
            (family_id, family_members, family_authorities, title, abstract, cpc, applicants, inventors,
             first_application_pub_date, first_grant_pub_date, rep_id, rep_authority, rep_kind, rep_pub_date,
             cpc_titles)
as
SELECT p.family_id,
       p.family_members,
       p.family_authorities,
       p.title,
       p.abstract,
       p.cpc,
       p.applicants,
       p.inventors,
       p.first_application_pub_date,
       p.first_grant_pub_date,
       p.rep_id,
       p.rep_authority,
       p.rep_kind,
       p.rep_pub_date,
       COALESCE((SELECT jsonb_agg(jsonb_build_object('code', s.code, 'title', s.title, 'level', s.level)
                                  ORDER BY s.code) AS jsonb_agg
                 FROM (SELECT DISTINCT ON (ct.cpc_code) ct.cpc_code AS code,
                                                        ct.title,
                                                        ct.level,
                                                        h.hier_rank
                       FROM unnest(p.cpc) c(raw_code)
                                CROSS JOIN LATERAL ( WITH norm
                                                              AS (SELECT upper(replace(c.raw_code::text, ' '::text, ''::text)) AS code),
                                                          parts
                                                              AS (SELECT "substring"(norm.code, '^([A-HY])'::text)           AS sec,
                                                                         "substring"(norm.code, '^([A-HY]\d{2})'::text)      AS cls,
                                                                         "substring"(norm.code, '^([A-HY]\d{2}[A-Z])'::text) AS subcls,
                                                                         "substring"(norm.code, '^([A-HY]\d{2}[A-Z]\d{1,3})'::text) ||
                                                                         '/00'::text                                         AS main_group,
                                                                         CASE
                                                                             WHEN POSITION(('/'::text) IN (norm.code)) > 0
                                                                                 THEN norm.code
                                                                             ELSE NULL::text
                                                                             END                                             AS subgroup
                                                                  FROM norm)
                                                     SELECT v.code_piece,
                                                            v.rnk AS hier_rank
                                                     FROM parts p2,
                                                          LATERAL ( VALUES (p2.sec, 1),
                                                                           (p2.cls, 2),
                                                                           (p2.subcls, 3),
                                                                           (p2.main_group, 4),
                                                                           (p2.subgroup, 5)) v(code_piece, rnk)
                                                     WHERE v.code_piece IS NOT NULL) h
                                JOIN cpc.cpc_titles ct ON ct.cpc_code::text = h.code_piece
                       ORDER BY ct.cpc_code) s), '[]'::jsonb) AS cpc_titles
FROM epo_doc_db.mv_patent_family p;

alter table ip_patent_families
    owner to postgres;

grant delete, insert, references, select, trigger, truncate, update on ip_patent_families to anon;

grant delete, insert, references, select, trigger, truncate, update on ip_patent_families to authenticated;

grant delete, insert, references, select, trigger, truncate, update on ip_patent_families to service_role;


create or replace function nn_ip_patent_family_search_by_id(c_uid uuid, p_id integer, k integer DEFAULT 10)
    returns TABLE(patent_family_id text, title text, abstract text, family_authorities character varying[], first_application_pub_date date, similarity double precision, sonar_is_active boolean, lists json[])
    language plpgsql
as
$$
DECLARE
    q vector(1024);
BEGIN
    SELECT e.embedding
    INTO q
    FROM export_embeddings AS e
    WHERE e.docdb_family_id = p_id;

    IF q IS NULL THEN
        RAISE EXCEPTION 'No embedding for family_id=%', p_id;
    END IF;

    PERFORM set_config('hnsw.ef_search', GREATEST(k, 24)::text, true);

    RETURN QUERY
        with q as (SELECT e.docdb_family_id,
                          1 - (e.embedding <=> q) AS cosine_similarity -- now types match
                   FROM export_embeddings AS e
                   WHERE e.docdb_family_id <> p_id
                   ORDER BY e.embedding <=> q
                   LIMIT k),
             client_sonar AS (SELECT s.patent_family_id,
                                     s.client_uid,
                                     s.uid as sonar_uid,
                                     s.radius,
                                     s.created_at,
                                     s.created_by,
                                     s.deactivated_at
                              FROM ip_patent_sonars AS s
                              WHERE s.client_uid = c_uid
                                AND s.deactivated_at IS NULL),
             patent_lists AS (SELECT *
                              from ip_patent_lists
                              where client_uid = c_uid),
             patent_list_items AS (SELECT i.patent_family_id,
                                          array_agg(json_build_object('name', l.name, 'list_uid', l.uid)) as lists
                                   from ip_patent_list_items as i
                                            left join patent_lists as l on l.uid = i.list_uid
                                   where i.client_uid = c_uid
                                   group by i.patent_family_id)
        SELECT f.family_id                                                as patent_family_id,
               f.title::text                                              as title,
               f.abstract                                                 as abstract,
               f.family_authorities                                       as family_authorities,
               f.first_application_pub_date                               as first_application_pub_date,
               round(q.cosine_similarity::numeric, 4)::double precision   as similarity,
               CASE WHEN c.sonar_uid IS NOT NULL THEN true ELSE false END as sonar_is_active,
               pli.lists                                                  as lists
        from ip_patent_families as f
                 JOIN q ON f.family_id = q.docdb_family_id::text
                 LEFT JOIN client_sonar as c ON c.patent_family_id = f.family_id
                 LEFT JOIN patent_list_items as pli ON pli.patent_family_id = f.family_id;
END
$$;

alter function nn_ip_patent_family_search_by_id(uuid, integer, integer) owner to postgres;

grant execute on function nn_ip_patent_family_search_by_id(uuid, integer, integer) to anon;

grant execute on function nn_ip_patent_family_search_by_id(uuid, integer, integer) to authenticated;

grant execute on function nn_ip_patent_family_search_by_id(uuid, integer, integer) to service_role;

SELECT nn_ip_patent_family_search_by_id('ea1a588c-378d-4d0a-817a-555439080997', 37420969, 200);


