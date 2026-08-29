import argparse
import json

from .freeze import freeze, verify
from .runner import observe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify", "observe"))
    parser.add_argument("--date")
    args = parser.parse_args()
    result = {"freeze": freeze, "verify": verify, "observe": lambda: observe(args.date)}[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

