import argparse
import json

from .freeze import freeze, verify
from .runner import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify", "run"))
    args = parser.parse_args()
    result = {"freeze": freeze, "verify": verify, "run": run}[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

