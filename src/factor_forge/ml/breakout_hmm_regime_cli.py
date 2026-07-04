from __future__ import annotations

import argparse
import json

from .breakout_hmm_regime import BreakoutHMMRegimeRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online HMM regime-gated breakout backtest")
    parser.add_argument("config")
    args = parser.parse_args()
    print(json.dumps(BreakoutHMMRegimeRunner().run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
