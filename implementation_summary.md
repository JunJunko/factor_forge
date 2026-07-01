# Factor Combination Implementation Summary

Implemented typed V1 schema validation, kind dispatch, relative atomic-YAML references, nested-combination rejection, content-addressed atomic caching, scope-first daily winsorization/normalization, direction alignment, equal/fixed weighting, three missing-value policies, tail filters, variants, automatic leave-one-out generation, industry-slice recomputation, and combination diagnostics.

Run normally:

```bash
factor-forge run --factor configs/combinations/short_term_alpha_combo_v1.yaml --experiment configs/experiments/short_term_alpha_combo_l1.yaml
```

Industry slices:

```bash
factor-forge run --factor configs/combinations/short_term_alpha_combo_v1.yaml --experiment configs/experiments/short_term_alpha_combo_industry_slice.yaml
```

Not supported in V1: nested combinations, ML/IC/Sharpe optimized weights, regime weights, nonlinear models, genetic/grid search, or automatic combination search.
