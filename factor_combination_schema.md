# Factor Combination Schema V1

Required top-level values are `version: 1`, `kind: factor_combination`, and `factor_combination`. A combination needs a non-empty `id`, a `name`, and at least two uniquely identified components. Sources must be existing atomic YAML files; relative paths resolve from the combination YAML directory. Nested combinations are rejected.

Supported normalization: `cs_zscore`, `cs_rank`, `cs_percentile`. Supported missing policies: `intersection`, `require_minimum_components`, `zero_after_normalization`. Supported methods: `weighted_sum`, `equal_weight`. Filters support `gt`, `gte`, `lt`, `lte` with `exclude` or `score_penalty`. Variants may reference only declared component/filter IDs. Weighted sums reject all-zero weights and percentile bounds must satisfy `0 <= lower < upper <= 1`.

See `configs/combinations/short_term_alpha_combo_v1.yaml` for the runnable schema example.
