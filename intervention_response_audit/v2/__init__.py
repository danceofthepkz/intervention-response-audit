"""Frozen-state authority/hedge audit (v2).

The v2 package is intentionally isolated from the cascade runner.  Stage-0 is
offline-only: it selects states, renders counterfactual prompts, and produces
integrity artifacts without importing or invoking an API client.
"""

from intervention_response_audit.v2.schema import V2FrozenConfig

__all__ = ["V2FrozenConfig"]
