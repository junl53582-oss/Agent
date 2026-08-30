import argparse
import json

from .core import observe
from .freeze import create_lock, verify_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-isolated prospective PIT observations")
    parser.add_argument("action", choices=("freeze", "verify", "observe"))
    parser.add_argument("--date")
    args = parser.parse_args()
    result = {"freeze": create_lock, "verify": verify_lock, "observe": lambda: observe(args.date)}[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
