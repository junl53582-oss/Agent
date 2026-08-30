import argparse
import json

from .core import admit_parent_expectations
from .freeze import create_lock, verify_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit frozen prospective PIT evidence offline")
    parser.add_argument("action", choices=("freeze", "verify", "admit"))
    args = parser.parse_args()
    result = {"freeze": create_lock, "verify": verify_lock, "admit": admit_parent_expectations}[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
