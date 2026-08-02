"""Compatibility imports for the former version-named producer module."""

from .component_capture.producer import (
    ComponentStreamBindings,
    ProducedComponentSample,
    validate_produced_values,
    validate_producer_bindings,
)


V25pProducerBindings = ComponentStreamBindings
V25pProducedSample = ProducedComponentSample


__all__ = [
    "V25pProducedSample",
    "V25pProducerBindings",
    "validate_produced_values",
    "validate_producer_bindings",
]
