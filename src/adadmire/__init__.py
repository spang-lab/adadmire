"""adadmire: Anomaly detection in mixed high-dimensional molecular data."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("adadmire")
except PackageNotFoundError:  # e.g. imported from a source checkout without installation
    __version__ = "unknown"

from adadmire.main import (
    get_threshold_continuous,
    get_threshold_discrete,
    loo_cv_cor,
    pred_continuous,
    pred_discrete,
    place_anomalies_continuous,
    impute,
    penalty,
    admire
)
