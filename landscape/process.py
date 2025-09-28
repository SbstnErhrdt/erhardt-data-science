from __future__ import annotations

from typing import Annotated, Any, Awaitable, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import asyncio
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

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


StatusCallback = Callable[[str, int, Optional[Dict[str, Any]]], Awaitable[None]]


async def run(
    _job_uid: str,
    client_uid: str,
    search_vectors: LandscapeSearchVectors,
    supabase_client,
    status_cb: StatusCallback,
) -> Dict[str, Any]:
    """
    Execute the landscape generation (v1 scope):
    - Seed patents via family IDs
    - Seed CPC classes (exact match)
    - Vector search via export_embeddings HNSW index
    - Status updates throughout
    `status_cb` should accept `(status: str, progress: int, extra: Optional[Dict[str, Any]])`.
    """

    await status_cb("calculating", 5, {"_message": "Parsing search vectors"})

    seed_family_ids, k_hint = _collect_seed_family_ids(search_vectors, supabase_client)
    if not seed_family_ids:
        raise ValueError("No seed family IDs found from search vectors")
    seed_family_ids = sorted(set(seed_family_ids))

    await status_cb("calculating", 15, {"_message": f"Found {len(seed_family_ids)} seed families"})

    k = k_hint or 200
    await status_cb("calculating", 30, {"_message": f"Running NN search (k={k}) by seed id"})
    neighbors = await _neighbors_by_seeds(client_uid, supabase_client, seed_family_ids, k)

    await status_cb("calculating", 55, {"_message": f"Aggregated {len(neighbors)} unique neighbors"})

    # Include seeds as items as well
    seed_set: Set[int] = set(int(s) for s in seed_family_ids)
    all_ids: List[int] = sorted(set(list(neighbors.keys())) | seed_set)

    # Fetch embeddings for dimensionality reduction (x,y) for all ids
    emb_map = await _fetch_embeddings_map(supabase_client, all_ids)

    # Compute 2D projection for those with embeddings; default (0,0) otherwise
    ordered_ids: List[int] = [fid for fid in all_ids if fid in emb_map]
    X = np.array([emb_map[fid] for fid in ordered_ids], dtype=float)

    xs: List[float] = []
    ys: List[float] = []
    if len(ordered_ids) >= 2 and X.ndim == 2 and X.shape[0] >= 2:
        coords = _umap_2d(X)
        xs = coords[:, 0].tolist()
        ys = coords[:, 1].tolist()
    else:
        xs = [0.0 for _ in ordered_ids]
        ys = [0.0 for _ in ordered_ids]

    id_to_xy = {fid: (xs[i], ys[i]) for i, fid in enumerate(ordered_ids)}

    # Fetch titles for all ids (uniformly); neighbors may already include titles
    meta = await _fetch_family_metadata(supabase_client, all_ids)
    title_map: Dict[int, str] = {}
    for row in meta:
        try:
            fid = int(row.get("family_id")) if row.get("family_id") is not None else None
        except Exception:
            fid = None
        if fid is not None:
            title_map[fid] = row.get("title") or ""

    # Build items for all ids, flagging seeds
    items: List[Dict[str, Any]] = []
    for fid in all_ids:
        x, y = id_to_xy.get(fid, (0.0, 0.0))
        title = title_map.get(fid) or (neighbors.get(fid, {}).get("title") if fid in neighbors else "") or ""
        items.append({
            "family_id": str(fid),
            "title": title,
            "x": float(x),
            "y": float(y),
            "is_seed": bool(fid in seed_set),
        })

    await status_cb("calculating", 90, {"_message": "Assembled result payload"})

    # Temporary compatibility: mirror items into `neighbors` for legacy UIs
    return {"v": 1, "items": items, "neighbors": items, "neighbor_count": len(items)}


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


async def _fetch_embeddings_map(supabase_client, family_ids: List[int]) -> Dict[int, List[float]]:
    if not family_ids:
        return {}
    out: Dict[int, List[float]] = {}
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
            fid = row.get("docdb_family_id")
            vec_raw = row.get("embedding")
            if fid is None:
                continue
            vec = _parse_embedding(vec_raw)
            if vec is not None and len(vec) > 0:
                out[int(fid)] = vec
    return out


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


def _umap_2d(X: np.ndarray) -> np.ndarray:
    """UMAP 2D projection; falls back to PCA if UMAP unavailable."""
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.1, metric="euclidean")
        return reducer.fit_transform(X)
    # Fallback to PCA if umap not installed
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:2, :]
    return Xc @ components.T


def _parse_embedding(v: Any) -> Optional[List[float]]:
    """Parse embedding value from PostgREST/Supabase into a list[float].
    Accepts list/tuple/ndarray or a string representation like "[1,2,3]" or "{1,2,3}" or "(1,2,3)".
    """
    import numpy as _np
    # Already a sequence of numbers
    if isinstance(v, (list, tuple, _np.ndarray)):
        try:
            return [float(x) for x in list(v)]
        except Exception:
            return None
    # String representation
    if isinstance(v, str):
        s = v.strip()
        # Remove common wrappers
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("{") and s.endswith("}")):
            s = s[1:-1]
        # Split and parse
        parts = [p for p in s.split(",") if p.strip()]
        try:
            return [float(p) for p in parts]
        except Exception:
            return None
    return None


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
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)  # type: ignore[attr-defined]
    except AttributeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
