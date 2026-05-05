# Report Feature Contract

Core features used by morning and closing reports:

- `return_5d`
- `return_20d`
- `return_60d`
- `trading_value_ratio_20d`
- `volatility_20d`
- `near_52w_high_pct`
- `foreign_flow_direction`
- `short_ratio`
- `value_quality_score`

Guidance:

- wide-format report views should expose these features directly.
- non-report features should be disabled or moved to slower cadences.
