# models.py
from __future__ import annotations

from typing import List, Optional, Union, Literal, Annotated, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict
import asyncio
import math

# -----------------
# Filters & helpers
# -----------------

DateField = Literal["priority", "filing", "publication", "grant"]
Status = Literal["pending", "granted", "expired", "withdrawn"]
CPCDepth = Literal["exact", "children", "subtree"]


class DateFilter(BaseModel):
    # "from" is a reserved keyword — use aliasing
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
    """
    Parse a dictionary into a LandscapeSearchVectors object.
    """
    return LandscapeSearchVectors.model_validate(data)


async def run(
    uid: str,
    search_vectors: LandscapeSearchVectors,
    supabase_client,
    status_cb,
) -> Dict[str, Any]:
    """
    Execute the landscape generation using the current minimal, production-ish pipeline.

    Scope (v1):
    - Seed patents via family IDs
    - Seed CPC classes (exact code match)
    - Vector search backbone via export_embeddings HNSW index
    - Status updates at key milestones

    Returns a dict suitable for storage in `_output`.
    """

    await status_cb(uid, "calculating", 5, {"_message": "Parsing search vectors"})

    # 1) Collect seed family IDs from nodes
    seed_family_ids, k_hint = _collect_seed_family_ids(search_vectors, supabase_client)
    if not seed_family_ids:
        raise ValueError("No seed family IDs found from search vectors")

    # 2) Remove negatives if any (handled inside collector), but ensure it's a clean list
    seed_family_ids = sorted(set(seed_family_ids))

    await status_cb(uid, "calculating", 15, {"_message": f"Found {len(seed_family_ids)} seed families"})

    # 3) Retrieve seed embeddings and compute centroid
    seed_embeddings = await _fetch_embeddings_for_families(supabase_client, seed_family_ids)
    if not seed_embeddings:
        raise ValueError("No embeddings found for seed families")

    await status_cb(uid, "calculating", 30, {"_message": f"Fetched {len(seed_embeddings)} seed embeddings"})

    centroid = _compute_centroid(seed_embeddings)
    if centroid is None:
        raise ValueError("Failed to compute centroid from seed embeddings")

    # 4) Vector similarity search via RPC
    k = k_hint or 200
    matches = await _vector_search(supabase_client, centroid, k)
    await status_cb(uid, "calculating", 55, {"_message": f"Vector search complete: {len(matches)} neighbors"})

    # 5) Fetch family metadata for matched IDs
    neighbor_ids = [m["docdb_family_id"] for m in matches]
    families_meta = await _fetch_family_metadata(supabase_client, neighbor_ids)
    await status_cb(uid, "calculating", 75, {"_message": f"Fetched metadata for {len(families_meta)} families"})

    # 6) Assemble output (minimal v1)
    # Align metadata with distances
    dist_map = {m["docdb_family_id"]: m["distance"] for m in matches}
    items = []
    for fam in families_meta:
        fid = fam.get("family_id")
        items.append(
            {
                "family_id": fid,
                "distance": dist_map.get(fid),
                "rep_id": fam.get("rep_id"),
                "rep_authority": fam.get("rep_authority"),
                "rep_kind": fam.get("rep_kind"),
                "rep_pub_date": fam.get("rep_pub_date"),
                "title": fam.get("title"),
                "abstract": fam.get("abstract"),
                "cpc_titles": fam.get("cpc_titles", []),
            }
        )

    await status_cb(uid, "calculating", 90, {"_message": "Assembled result payload"})

    # Keep it simple for v1; later we can add UMAP/clustering outputs
    result = {
        "v": 1,
        "mode": "neighbors",
        "seed_count": len(seed_family_ids),
        "neighbor_count": len(items),
        "neighbors": items,
    }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────────

def _collect_seed_family_ids(
    search_vectors: LandscapeSearchVectors,
    supabase_client,
) -> Tuple[List[int], Optional[int]]:
    """
    Gather seed family IDs from seed_patents and CPC nodes.
    - CPC: exact code matches only (depth exact); children/subtree not yet implemented.
    Returns (seed_family_ids, k_hint) where k_hint is the max k seen on nodes.
    """
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
            # Only exact code matching for now
            codes = [c.code for c in (node.classes or []) if c.code]
            # Aggregate families for each code (exact)
            fams = _query_family_ids_by_cpc_exact(supabase_client, codes)
            seeds.update(fams)

    # Remove negatives
    final = list(seeds.difference(negatives))
    return final, k_hint


def _query_family_ids_by_cpc_exact(supabase_client, codes: List[str]) -> Set[int]:
    """
    Fetch family_ids where ip_patent_families.cpc_titles contains an object with code == given code.
    Runs multiple queries (one per code) and unions the results.
    """
    out: Set[int] = set()
    for code in codes:
        # JSON containment: array contains object with at least {"code": code}
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
    # Chunk to avoid URL length issues
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


if __name__ == "__main__":
    import json

    example = {
        "v": 1,
        "type": "hybrid",
        "children": [
            {
                "node": {
                    "type": "seed_patents",
                    "name": "My Seed Patents Node",
                    "description": "This is a description of my seed patents node.",
                    "k": 100,
                    "seeds": {
                        "patentFamilyIds": ["US1234567A", "US2345678B"],
                        "negativePatentFamilyIds": ["US3456789C"],
                    },
                    "filters": {
                        "date": {"field": "filing", "from": "2010-01-01", "to": "2020-12-31"},
                        "authority": ["US", "EP"],
                        "status": ["granted"],
                        "assignee": {"include": ["Google"], "fuzzy": True},
                    },
                },
                "order": 1,
            },
            {
                "node": {
                    "type": "cpc",
                    "name": "My CPC Node",
                    "description": "This is a description of my CPC node.",
                    "classes": [
                        {"code": "G06F", "depth": "subtree"},
                        {"code": "H04L", "depth": "children"},
                    ],
                    "filters": {
                        "date": {"field": "publication", "from": "2015-01-01"},
                        "language": ["EN"],
                        "familySize": {"min": 5},
                    },
                },
                "order": 2,
            },
            {
                "node": {
                    "type": "seed_docs",
                    "name": "My Seed Docs Node",
                    "description": "This is a description of my seed docs node.",
                    "k": 50,
                    "docs": [
                        {
                            "id": None,
                            "title": None,
                            "abstract": None,
                            "textRef": {
                                "uri": "https://example.com/mydoc.txt",
                                # SHA256 is optional but recommended
                                # to ensure content integrity
                                # and avoid re-fetching if content changes
                                # e.g. due to website updates.
                                # You can compute it with:
                                #   import hashlib
                                #   sha256 = hashlib.sha256(content_bytes).
