"""Run ATR lower-shadow reversion IC + LightGBM experiment."""

from __future__ import annotations

import sys

from factor_forge.ml.atr_reversion_runner import run_atr_reversion


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/ml/atr_reversion_lightgbm_v1.yaml"
    summary = run_atr_reversion(config)
    print(f"run_dir={summary['run_dir']}")

