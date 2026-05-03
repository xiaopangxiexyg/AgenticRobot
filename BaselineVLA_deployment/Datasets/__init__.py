"""Dataset discovery, canonicalization, and model input adapters."""

from .registry import build_data_pipeline

__all__ = ["build_data_pipeline"]
