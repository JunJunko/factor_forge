# Test Report

Status: **PASS**

Command: `pytest -q`

Result: `29 passed`.

Coverage includes all prior platform and industry-slice regression tests plus factor-kind dispatch, legacy no-kind behavior, schema validation, relative atomic references, nested-combination rejection, scope-first normalization, direction reversal, fixed/equal weights, all three missing-value policies, filters, variants, cache hits, YAML-hash invalidation, and data-version invalidation.

Additional checks: `python -m compileall -q src` passed; all three atomic examples, the combination example, and both experiment examples passed Pydantic schema validation; `python -m factor_forge.cli run --help` confirms the unchanged two-option CLI.
