import argparse
import json

from .freeze import freeze, verify


def main():
    parser = argparse.ArgumentParser(description="V20 independent implementation repair")
    parser.add_argument("command", choices=("freeze", "verify", "run"))
    args = parser.parse_args()
    if args.command == "run":
        from .validation import run
        result = run()
    else:
        result = freeze() if args.command == "freeze" else verify()
        result = {key: result[key] for key in ("lock_sha256", "frozen_inputs_intact", "execution_authorized")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
