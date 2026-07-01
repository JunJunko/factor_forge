# Industry Slice Schema

`industry_slice` is optional and defaults off. Supported values:

```yaml
industry_slice:
  enabled: false
  industry_standard: sw
  industry_level: l1
  membership_mode: point_in_time
  selector:
    preset: sw_l1_neutralized_rotation_v1
    overrides:
      short_ema: 3
      long_ema: 10
      breadth_change_window: 5
      ridge_alpha: 1.0
      minimum_industry_members: 8
  scopes: [all, top5, bottom5]
  diagnostics:
    evaluate_industry_selector: true
    save_industry_intermediate: true
```

Allowed scopes are `all`, `top2`, `top5`, `top10`, and `bottom5`. Allowed targets are `stock_return` and `stock_minus_sw_l1_return`. Unknown presets, standards, levels, membership modes, scopes, and targets fail schema validation rather than silently falling back.
