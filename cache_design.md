# Atomic Factor Cache Design

The cache stores atomic `factor_value` output before combination-scope normalization. Its SHA-256 key covers `factor_yaml_hash`, `data_version`, `start_date`, `end_date`, `adjustment_mode`, `base_universe`, `membership_version`, and `factor_engine_version`.

A YAML edit or data/membership version change creates a different entry. Data and metadata files must both exist and validate; partial, unreadable, missing-column, or duplicate-key entries fail explicitly. Users never supply a cache path through the CLI.
