from __future__ import annotations

import argparse
import json

from .breakout_qlib_backtest import QlibPredictionBacktestRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Qlib breakout predictions")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(QlibPredictionBacktestRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
