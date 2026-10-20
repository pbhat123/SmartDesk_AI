"""Run a command and kill it, with everything it started, if it runs longer than a limit. macOS has no `timeout`.

    python scripts/run_with_limit.py 600 -- .venv/bin/python -m pytest -q

Exit code: the command's own, or TIMED_OUT (124, the code GNU timeout uses) when the limit was hit. The pre-commit hook uses it so a
hung test run fails the commit instead of waiting for ever."""
import os
import signal
import subprocess
import sys

TIMED_OUT = 124


def run_with_limit(seconds: float, command: list[str]) -> int:
    process = subprocess.Popen(command, start_new_session=True)  # its own process group, so the whole tree can be killed
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print(f"run_with_limit: killed after {seconds:g} s: {' '.join(command[:4])} ...", file=sys.stderr)
        return TIMED_OUT


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print("usage: run_with_limit.py SECONDS -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    return run_with_limit(float(argv[0]), argv[2:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
