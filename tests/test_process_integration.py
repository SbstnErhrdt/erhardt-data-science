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
    # New minimal format exposes `items` with only id and title
    items = result.get("items") or result.get("neighbors")
    assert isinstance(items, list)

    if items:
        it0 = items[0]
        assert "family_id" in it0
        assert "title" in it0
        # With dimensionality reduction we expose x,y
        assert "x" in it0 and "y" in it0
