# Architecture Changes

Industry research is isolated under `src/factor_forge/research/industry/`:

- `IndustryContextBuilder`: validates signal-date PIT membership and uniqueness.
- `IndustryFeatureBuilder`: builds industry returns, breadth, capital, and time-series features.
- `IndustryNeutralizer`: daily ridge residualization only.
- `IndustrySelector`: preset score and deterministic Top/Bottom selection.
- `IndustrySliceMapper`: validated many-to-one industry-to-stock mapping.
- `IndustryResidualReturnBuilder`: aligned `open_t1` targets with leave-one-out benchmarks.
- `IndustrySliceEvaluator`: reuses the existing L1 daily-correlation and summary primitives.
- `IndustrySlicePipeline`: orchestration and reporting.

The immutable stock data version is the cache boundary. Industry intermediates may be persisted in each run when diagnostics request it; they are never user inputs. A future shared cache must separately key base aggregation, features, and selections by data version/date range, membership version, stock-filter hash, standard/level, EMA windows, breadth window, member minimum, and selector parameters.
