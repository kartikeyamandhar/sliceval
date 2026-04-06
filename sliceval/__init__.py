"""sliceval — Slice-based model evaluation for ML engineers."""

from .evaluator import SliceEvaluator
from .slice import Slice, SliceMetrics
from .report import SliceReport

__version__ = "0.1.0"
__all__ = ["SliceEvaluator", "Slice", "SliceMetrics", "SliceReport"]
