from __future__ import annotations

import json

from .validation import run_research_v7


if __name__ == "__main__":
    print(json.dumps(run_research_v7(), ensure_ascii=False, indent=2))
