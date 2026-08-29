"""Download only four public quote series into a new, exclusive audit directory."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from research_v20.freeze import digest, write_new


ROOT = Path("data/corporate_actions_v20r2_sources")
QUOTE_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    for symbol in ("600837", "601211", "601989", "600150"):
        target = ROOT / f"{symbol}_2025_unadjusted.json"
        if target.exists():
            continue
        url = QUOTE_URL + "?" + urlencode({"param": f"sh{symbol},day,2025-01-01,2025-12-31,640,"})
        request = Request(url, headers={"User-Agent": "StockPilot-research-data-audit/1.0"})
        with urlopen(request, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw)
        rows = payload.get("data", {}).get("sh" + symbol, {}).get("day")
        if not rows:
            raise ValueError(f"No unadjusted daily quotes: {symbol}")
        with target.open("xb") as handle:
            handle.write(raw)
        write_new(target.with_suffix(".provenance.json"), {
            "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": digest(target), "adjustment": "unadjusted", "rows": len(rows),
        })
        print(f"quote source saved: {symbol}, rows={len(rows)}", flush=True)


if __name__ == "__main__":
    prepare()
