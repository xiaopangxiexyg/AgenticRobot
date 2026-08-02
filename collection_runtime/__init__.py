"""Device-, control-, and UI-neutral live collection runtime.

This package owns generic collection-time lifecycle and persistence mechanics.
It does not own device production, robot control, GUI interaction, dataset
schema definitions, or training/read-time projections.
"""

# Keep package import side-effect free. Public runtime objects remain owned by
# their submodules so importing one compatibility shim does not load unrelated
# stores or optional dependencies.
__all__: list[str] = []
