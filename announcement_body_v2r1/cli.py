import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify", "run"))
    command = parser.parse_args().command
    if command == "run":
        from .runner import run
        result = run()
    else:
        from .freeze import freeze, verify
        result = (freeze if command == "freeze" else verify)()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
