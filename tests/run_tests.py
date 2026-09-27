"""Minimal test runner (pytest is not installable in this environment)."""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(modules):
    failed = 0
    for modname in modules:
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"PASS {modname}.{name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {modname}.{name}: {exc}")
                traceback.print_exc()
    print(f"\n{'ALL PASS' if not failed else str(failed) + ' FAILURES'}")
    return 1 if failed else 0


if __name__ == "__main__":
    mods = sys.argv[1:] or ["tests.test_core"]
    sys.exit(main(mods))
