from tigerml.core.utils._lib import DictObject
from tigerml.model_monitoring.core.metrics import (
    anderson,
    chiSquare,
    dsi,
    kl,
    ks,
    psi,
)

DRIFT_OPTIONS = DictObject(
    {
        "PSI": {
            "string": "population_stability_index",
            "measure_type": "information theory",
            "func": psi,
            "default_params": {"n_bins": 7},
            "applicable_drift_type": ["target_drift", "feature_drift"],
            "applicable_data_type": ["numerical", "categorical"],
        },
        "ChiSquare": {
            "string": "ChiSquareDrift",
            "measure_type": "statistical",
            "func": chiSquare,
            "default_params": {},
            "applicable_drift_type": ["target_drift", "feature_drift"],
            "applicable_data_type": ["categorical"],
        },
        # "Anderson": {
        #     "string": "AndersonDrift",
        #     "func": anderson,
        #     "default_params": {},
        #     "applicable_drift_type": ["target_drift", "feature_drift"],
        #     "applicable_data_type": ["numerical"],
        # },
        # "KLDivergence": {
        #     "string": "KLDivergenceDrift",
        #     "measure_type": "information theory",
        #     "func": kl,
        #     "default_params": {"approximation": "gaussian"},
        #     "applicable_drift_type": ["target_drift", "feature_drift"],
        #     "applicable_data_type": ["numerical"],
        # },
        "KS": {
            "string": "KSDrift",
            "measure_type": "statistical",
            "func": ks,
            "default_params": {},
            "applicable_drift_type": ["target_drift", "feature_drift"],
            "applicable_data_type": ["numerical"],
        },
        "DSI": {
            "string": "DSIDrift_v1",
            "measure_type": "information theory",
            "func": dsi,
            "default_params": {"n_feature_bins": 10, "n_target_bins": 5},
            "applicable_drift_type": ["concept_drift"],
            "applicable_data_type": ["numerical", "categorical"],
        },
    }
)
