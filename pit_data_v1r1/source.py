from __future__ import annotations

import math

import requests

from pit_data_v1 import core as parent


FLOW_PAGE_SIZE = 100


def fetch_flow_pages(session: requests.Session | None = None) -> list[tuple[bytes, dict]]:
    """Use the provider's observed hard maximum instead of V1's requested 500 rows."""
    session = session or requests.Session()
    base = {
        "fid": "f62",
        "po": "1",
        "pz": str(FLOW_PAGE_SIZE),
        "pn": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
        "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f124",
    }
    first = parent._get_json(session, parent.FLOW_URL, base)
    data = first[1].get("data")
    if not isinstance(data, dict) or not isinstance(data.get("diff"), list):
        raise ValueError("fund-flow response shape invalid")
    total = int(data.get("total") or 0)
    pages = int(math.ceil(total / FLOW_PAGE_SIZE))
    output = [first]
    for page in range(2, pages + 1):
        output.append(parent._get_json(session, parent.FLOW_URL, {**base, "pn": str(page)}))
    actual = sum(len(item[1]["data"].get("diff") or []) for item in output)
    if actual != total:
        raise ValueError(f"fund-flow pagination mismatch: expected={total}, actual={actual}")
    return output
