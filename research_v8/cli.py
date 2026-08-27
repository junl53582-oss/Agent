from __future__ import annotations

import json

from .validation import run_research_v8


def main() -> None:
    print(json.dumps(run_research_v8(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
