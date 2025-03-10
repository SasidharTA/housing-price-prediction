"""Utilities for ``Attribution`` usecases.

The module provides custom ``Attribution Function`` can be integrated with any module.
"""

from .attribution import (
    set_baseline_value,
    _predict,
    get_var_contribution_variants,
    get_attribution
)
