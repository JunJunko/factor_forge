from __future__ import annotations

import argparse
import json

from .breakout_qlib_walkforward import BreakoutQlibWalkForwardRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qlib breakout walk-forward training")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(BreakoutQlibWalkForwardRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
