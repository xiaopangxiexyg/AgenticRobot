"""Compatibility exports for lower-owned Human composition adaptation."""

from teleop_runtime.action.human_composition_input import (
    HumanComponentActionAdaptation,
    HumanCompositionInputAdaptation,
    adapt_human_emission_to_component_action,
    adapt_human_emission_to_composition_input,
)


__all__ = [
    "HumanComponentActionAdaptation",
    "HumanCompositionInputAdaptation",
    "adapt_human_emission_to_component_action",
    "adapt_human_emission_to_composition_input",
]
