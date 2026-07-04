from __future__ import annotations

import argparse
import json

from .scenario_backtest import ScenarioBacktestRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the top conditional-IC scenarios")
    parser.add_argument("config", help="Path to scenario backtest YAML")
    args = parser.parse_args()
    print(json.dumps(ScenarioBacktestRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
