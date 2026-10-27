"""Error-gated Oja model and its experiment functions (ongoing work)."""

from .oja_variant_model import ErrorGatingModel
from .experiment import run_error_experiment, run_error_baseline

__all__ = [
    "ErrorGatingModel",
    "run_error_experiment",
    "run_error_baseline",
]
