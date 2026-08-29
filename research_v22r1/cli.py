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
        result = {key: result[key] for key in ("lock_sha256", "frozen_inputs_intact", "execution_authorized")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
