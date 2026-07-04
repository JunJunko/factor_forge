from __future__ import annotations

import argparse
import json

from .core_validation import ContinuousMoveCoreValidationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate continuous_move as a frozen core factor")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(ContinuousMoveCoreValidationRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
