import os
import uuid
import pytest
import asyncio

from dotenv import load_dotenv

load_dotenv()


pytestmark = pytest.mark.asyncio


def have_supabase_env():
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_KEY"))


@pytest.mark.skipif(not have_supabase_env(), reason="Missing SUPABASE_URL or SUPABASE_KEY in environment")
async def test_run_neighbors_by_id_pipeline():
    # Lazy import to avoid requiring deps when env is missing
    from supabase import create_client
    from landscape import process

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    supabase = create_client(supabase_url, supabase_key)

    # Build minimal search_vectors payload
    payload = {
        "v": 1,
        "type": "hybrid",
        "children": [
            {
                "node": {
                    "type": "seed_patents",
                    "k": 5,
                    "seeds": {
                        "patentFamilyIds": ["33308133", "52130238"],
                        "negativePatentFamilyIds": [],
                    },
                },
                "order": 1,
            },
            {
                "node": {
                    "type": "cpc",
                    "classes": [
                        {"code": "F16H25/2252", "depth": "exact"}
                    ],
                },
                "order": 2,
            },
        ],
    }

    sv = process.parse(payload)

    # Dummy job and client IDs (client_uid may not exist; RPC uses it as a filter only)
    job_uid = str(uuid.uuid4())
    client_uid = str(uuid.uuid4())

    async def status_cb(uid: str, status: str, progress: int, extra=None):
        # No-op for tests; could print for debugging
        return None

    result = await process.run(job_uid, client_uid, sv, supabase, status_cb)

    # Basic shape assertions
    assert result["v"] == 1
    assert result["mode"] == "neighbors_by_id"
    assert isinstance(result["seed_count"], int)
    assert isinstance(result["neighbor_count"], int)
    assert isinstance(result["neighbors"], list)

    # If no data available, neighbors could be empty; but typically expect some
    if result["neighbors"]:
        n0 = result["neighbors"][0]
        assert "family_id" in n0
        assert "similarity" in n0
        sim = n0["similarity"]
        if sim is not None:
            assert 0.0 <= sim <= 1.0
        # Ensure seed provenance structure exists
        assert "source_seeds" in n0

