from __future__ import annotations

import argparse
import json

from .research import BreakoutResearchRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-box breakout conditional IC research")
    parser.add_argument("config", help="Path to breakout research YAML")
    arguments = parser.parse_args()
    result = BreakoutResearchRunner().run(arguments.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
