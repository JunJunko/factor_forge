# Factor Combination Leakage Check

- `full_sample_normalization`: daily group-by only; PASS
- `future_factor_data`: atomic DSL remains limited to current/past operators; PASS
- `scope_before_normalization`: scope mask is applied before preprocessing; PASS
- `factor_target_alignment`: combination receives no future-return columns; PASS
- `cached_data_version`: cache metadata includes data/date/membership versions; PASS
- `industry_slice_alignment`: each selected industry scope is recomputed; PASS

The run manifest records these checks. Any failed check produces `INVALID` and suppresses an Alpha conclusion.
