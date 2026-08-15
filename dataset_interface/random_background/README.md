# Random-background augmentation

This package owns the reusable, framework-neutral image augmentation, mask
reader, photometric, occlusion, and real-image provider semantics consumed by
`dataset_interface`.

Offline download, indexing, preview, and sample-generation commands remain in
`utils/random_background/`. The old Python import paths there are compatibility
re-exports only; new library code imports this package directly.
