# Industry Slice Leakage Report

Implementation audit status: **VALID**

- `future_industry_membership`: PASS — stock-day membership comes from the immutable PIT panel.
- `full_sample_standardization`: PASS — z-scores and neutralization are grouped by trade date.
- `same_day_future_return_usage`: PASS — scores use close_t-or-earlier fields; targets begin at open_t1.
- `target_window_alignment`: PASS — stock and industry benchmark share entry and exit prices.
- `duplicate_stock_industry_mapping`: PASS — duplicates fail; mapping uses `validate="many_to_one"`.
- `industry_score_future_data`: PASS — rolling features are backward-looking.
- `industry_return_membership_alignment`: PASS — benchmarks use signal-date PIT membership.

Every run emits its own leakage report. Missing PIT coverage, duplicate mappings, empty usable results, or failed construction raises an explicit error or marks the report `INVALID`; no non-PIT fallback exists.
