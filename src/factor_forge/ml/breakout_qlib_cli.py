from __future__ import annotations

import argparse
import json

from .breakout_qlib import BreakoutQlibRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qlib LightGBM on breakout events")
    parser.add_argument("config", help="Path to breakout Qlib YAML")
    args = parser.parse_args()
    print(json.dumps(BreakoutQlibRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
