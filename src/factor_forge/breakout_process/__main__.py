from __future__ import annotations

import argparse
import json

from .research import BreakoutResearchRunner
from .validation import FrozenBreakoutValidationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-box breakout conditional IC research")
    parser.add_argument("config", help="Path to breakout YAML")
    parser.add_argument(
        "--mode", choices=("research", "validate"), default="research"
    )
    arguments = parser.parse_args()
    runner = (
        BreakoutResearchRunner()
        if arguments.mode == "research"
        else FrozenBreakoutValidationRunner()
    )
    result = runner.run(arguments.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
