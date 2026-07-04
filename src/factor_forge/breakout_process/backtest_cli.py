from __future__ import annotations

import argparse
import json

from .backtest import EventBacktestRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest frozen-box breakout event scores")
    parser.add_argument("config", help="Path to breakout event backtest YAML")
    args = parser.parse_args()
    print(json.dumps(EventBacktestRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
