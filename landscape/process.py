from __future__ import annotations

from typing import List, Optional, Union, Literal, Annotated, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict
import asyncio

# -----------------
# Filters & helpers
# -----------------

DateField = Literal["priority", "filing", "publication", "grant"]
Status = Literal["pending", "granted", "expired", "withdrawn"]
CPCDepth = Literal["exact", "children", "subtree"]


class DateFilter(BaseModel):
    field: Optional[DateField] = None
    from_: Optional[str] = Field(default=None, alias="from")
    to: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class RangeMinMax(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class AssigneeFilter(BaseModel):
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None
    fuzzy: Optional[bool] = None

    model_config = ConfigDict(extra="forbid")


class PartyFilter(BaseModel):
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


class Filters(BaseModel):
    date: Optional[DateFilter] = None
    authority: Optional[List[str]] = None
    kindCodes: Optional[List[str]] = None
    status: Optional[List[Status]] = None
    language: Optional[List[str]] = None

    assignee: Optional[AssigneeFilter] = None
    inventor: Optional[PartyFilter] = None
    applicant: Optional[PartyFilter] = None

    cpcInclude: Optional[List[str]] = None
    cpcExclude: Optional[List[str]] = None

    familySize: Optional[RangeMinMax] = None
    citationCount: Optional[RangeMinMax] = None
    collections: Optional[List[str]] = None

    textMustContain: Optional[List[str]] = None
    textMustNotContain: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


# -----------------
# Retrieval nodes
# -----------------

class RetrievalCommon(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    filters: Optional[Filters] = None

    model_config = ConfigDict(extra="forbid")


class SeedPatentsSeeds(BaseModel):
    patentFamilyIds: Optional[List[str]] = None
    negativePatentFamilyIds: Optional[List[str]] = None

    model_config = ConfigDict(extra="forbid")


class SeedPatentsNode(RetrievalCommon):
    type: Literal["seed_patents"]
    k: Optional[int] = None
    seeds: SeedPatentsSeeds


class CPCClass(BaseModel):
    code: str
    depth: Optional[CPCDepth] = None

    model_config = ConfigDict(extra="forbid")


class CPCNode(RetrievalCommon):
    type: Literal["cpc"]
    classes: List[CPCClass]


class TextRef(BaseModel):
    uri: str
    sha256: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class SeedDoc(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    textRef: Optional[TextRef] = None
    weight: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class SeedDocsNode(RetrievalCommon):
    type: Literal["seed_docs"]
    k: Optional[int] = None
    docs: Optional[List[SeedDoc]]


# Discriminated union on "type"
RetrievalNode = Annotated[
    Union[SeedPatentsNode, CPCNode, SeedDocsNode],
    Field(discriminator="type"),
]


class HybridChild(BaseModel):
    node: RetrievalNode
    order: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class LandscapeSearchVectors(BaseModel):
    v: Literal[1]
    type: Literal["hybrid"]
    children: List[HybridChild] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def parse(data: dict) -> LandscapeSearchVectors:
    return LandscapeSearchVectors.model_validate(data)


async def run(
    uid: str,
    client_uid: str,
    search_vectors: LandscapeSearchVectors,
    supabase_client,
    status_cb,
) -> Dict[str, Any]:
    """
    Execute the landscape generation (v1 scope):
    - Seed patents via family IDs
    - Seed CPC classes (exact match)
    - Vector search via export_embeddings HNSW index
    - Status updates throughout
    """

    await status_cb(uid, "calculating", 5, {"_message": "Parsing search vectors"})

    seed_family_ids, k_hint = _collect_seed_family_ids(search_vectors, supabase_client)
    if not seed_family_ids:
        raise ValueError("No seed family IDs found from search vectors")
    seed_family_ids = sorted(set(seed_family_ids))

    await status_cb(uid, "calculating", 15, {"_message": f"Found {len(seed_family_ids)} seed families"})

    k = k_hint or 200
    await status_cb(uid, "calculating", 30, {"_message": f"Running NN search (k={k}) by seed id"})
    neighbors = await _neighbors_by_seeds(client_uid, supabase_client, seed_family_ids, k)

    await status_cb(uid, "calculating", 80, {"_message": f"Aggregated {len(neighbors)} unique neighbors"})

    items = [
        {
            "family_id": str(fid),
            "similarity": data["similarity"],
            "title": data.get("title"),
            "abstract": data.get("abstract"),
            "family_authorities": data.get("family_authorities"),
            "first_application_pub_date": data.get("first_application_pub_date"),
            "sonar_is_active": data.get("sonar_is_active"),
            "lists": data.get("lists"),
            "source_seeds": sorted(list(data.get("source_seeds", []))),
        }
        for fid, data in neighbors.items()
    ]

    await status_cb(uid, "calculating", 90, {"_message": "Assembled result payload"})

    return {
        "v": 1,
        "mode": "neighbors_by_id",
        "seed_count": len(seed_family_ids),
        "neighbor_count": len(items),
        "neighbors": items,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

def _collect_seed_family_ids(
    search_vectors: LandscapeSearchVectors,
    supabase_client,
) -> Tuple[List[int], Optional[int]]:
    seeds: Set[int] = set()
    negatives: Set[int] = set()
    k_hint: Optional[int] = None

    for child in sorted(search_vectors.children, key=lambda c: (c.order if c.order is not None else 0)):
        node = child.node
        if isinstance(node, SeedPatentsNode):
            if node.k:
                k_hint = max(k_hint or 0, node.k)
            fams = node.seeds.patentFamilyIds or []
            negs = node.seeds.negativePatentFamilyIds or []
            for f in fams:
                try:
                    seeds.add(int(f))
                except Exception:
                    continue
            for f in negs:
                try:
                    negatives.add(int(f))
                except Exception:
                    continue

        elif isinstance(node, CPCNode):
            codes = [c.code for c in (node.classes or []) if c.code]
            fams = _query_family_ids_by_cpc_exact(supabase_client, codes)
            seeds.update(fams)

    final = list(seeds.difference(negatives))
    return final, k_hint


def _query_family_ids_by_cpc_exact(supabase_client, codes: List[str]) -> Set[int]:
    out: Set[int] = set()
    for code in codes:
        try:
            resp = (
                supabase_client.table("ip_patent_families")
                .select("family_id")
                .contains("cpc_titles", [{"code": code}])
                .execute()
            )
            for row in (resp.data or []):
                fid = row.get("family_id")
                if fid is not None:
                    out.add(int(fid))
        except Exception:
            continue
    return out


async def _fetch_embeddings_for_families(supabase_client, family_ids: List[int]) -> List[List[float]]:
    if not family_ids:
        return []
    embeddings: List[List[float]] = []
    chunk_size = 1000
    for i in range(0, len(family_ids), chunk_size):
        chunk = family_ids[i:i + chunk_size]
        def _do():
            return (
                supabase_client.table("export_embeddings")
                .select("docdb_family_id, embedding")
                .in_("docdb_family_id", chunk)
                .execute()
            )
        resp = await _run_in_threadpool(_do)
        for row in (resp.data or []):
            vec = row.get("embedding")
            if isinstance(vec, list) and vec:
                embeddings.append(vec)
    return embeddings


def _compute_centroid(vectors: List[List[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i in range(dim):
            acc[i] += float(v[i])
    n = len(vectors)
    if n == 0:
        return None
    return [x / n for x in acc]


async def _vector_search(supabase_client, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
    def _do():
        return supabase_client.rpc(
            "match_export_embeddings",
            {"query_embedding": query_embedding, "match_count": int(k)},
        ).execute()
    resp = await _run_in_threadpool(_do)
    return resp.data or []


async def _neighbors_by_seeds(
    client_uid: str,
    supabase_client,
    seed_family_ids: List[int],
    k: int,
) -> Dict[int, Dict[str, Any]]:
    """
    For each seed family id, call the RPC `nn_ip_patent_family_search_by_id` and merge results.
    Returns a dict keyed by neighbor family_id (int) with best similarity and merged metadata.
    """
    results: Dict[int, Dict[str, Any]] = {}

    async def _one(seed_id: int):
        def _do():
            return supabase_client.rpc(
                "nn_ip_patent_family_search_by_id",
                {"c_uid": client_uid, "p_id": int(seed_id), "k": int(k)},
            ).execute()
        return await _run_in_threadpool(_do)

    # Run sequentially to avoid overwhelming RPC; can batch later
    for seed in seed_family_ids:
        try:
            resp = await _one(seed)
            for row in (resp.data or []):
                fid_txt = row.get("patent_family_id")
                try:
                    fid = int(fid_txt) if fid_txt is not None else None
                except Exception:
                    fid = None
                if fid is None:
                    continue
                sim = row.get("similarity")
                # Keep the best (max) similarity for duplicates
                cur = results.get(fid)
                if (cur is None) or (sim is not None and cur.get("similarity", -1) < sim):
                    results[fid] = {
                        "similarity": sim,
                        "title": row.get("title"),
                        "abstract": row.get("abstract"),
                        "family_authorities": row.get("family_authorities"),
                        "first_application_pub_date": row.get("first_application_pub_date"),
                        "sonar_is_active": row.get("sonar_is_active"),
                        "lists": row.get("lists"),
                        "source_seeds": {int(seed)},
                    }
                else:
                    # augment source seeds
                    cur.setdefault("source_seeds", set()).add(int(seed))
        except Exception:
            # continue on RPC errors for single seed
            continue

    # remove any neighbor that is itself a seed (RPC already excludes seed, but double-safeguard)
    for s in list(seed_family_ids):
        results.pop(int(s), None)

    return results


async def _fetch_family_metadata(supabase_client, family_ids: List[int]) -> List[Dict[str, Any]]:
    if not family_ids:
        return []
    out: List[Dict[str, Any]] = []
    chunk_size = 1000
    for i in range(0, len(family_ids), chunk_size):
        chunk = family_ids[i:i + chunk_size]
        def _do():
            return (
                supabase_client.table("ip_patent_families")
                .select(
                    "family_id,title,abstract,cpc_titles,rep_id,rep_authority,rep_kind,rep_pub_date"
                )
                .in_("family_id", chunk)
                .execute()
            )
        resp = await _run_in_threadpool(_do)
        out.extend(resp.data or [])
    return out


async def _run_in_threadpool(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
