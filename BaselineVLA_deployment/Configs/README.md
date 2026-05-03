# Configs Directory Structure

This directory uses a modularized configuration design. When adjusting variables or hyperparameters, please locate the corresponding subfolder and files below.

```text
Configs/
├── data/
│   └── mock_lerobot_meta.yaml
│       ├── name
│       ├── raw                                                           # Raw dataset configuration
│       │   ├── dataset_type, manifest_loader, root_dirs, discovery_glob  # Dataset formats, parsers, and search paths
│       │   ├── state_key, action_key, derived_val_ratio                  # Data dictionary keys and validation split ratio
│       │   ├── filters: state_dim, action_dim, robot_type                # Dimension filters and specific robot constraints
│       │   ├── limits: max_datasets, max_episodes... max_total_samples   # Data loading quantity limits and bounds
│       │   ├── sampling: frame_stride, max_video_tokens, state_scale...  # Token limits, strides, data scalings and mix ratios
│       │   ├── sequence.med.kind, side, future_*, emit_intermediate      # Selected med target kind, temporal window, and whether to materialize raw med tensors
│       │   ├── sequence.med.vector / sequence.med.point_cloud            # Vector fields or point-cloud field bundle
│       │   └── loader: batch_size, num_workers, pin_memory               # PyTorch DataLoader specific properties
│       └── med                                                           # Manipulation Embedding (MED) configuration
│           ├── provider_type, source_intermediate_key, flatten_sequence  # MED provider wiring and output layout
│           ├── enabled_for_experiments: gtmed, fulltrain, genmed, vanilla # Boolean flags indicating which runs use this feature
│           └── mode, scale, base_seed                                    # Embedding projection mode, scaling and RNG seed
│
├── model/
│   ├── basic_diffusion.yaml                                              # Diffusion policy model configuration
│   │   ├── name, family, activation, dropout                             # Core model identity and basic neural hyperparameters
│   │   ├── vision_token_dim, use_vision_tokens, context_dim              # Observation and context setups
│   │   ├── task: enabled, num_embeddings, id_embedding_dim...            # Task ID conditioning and encoding properties
│   │   ├── state_encoder_hidden_dims, vision_encoder_hidden_dims         # MLP shapes for input state/vision projection
│   │   ├── context_num_attention_heads, context_transformer_layers...    # Context Transformer encoder capacities
│   │   ├── diffusion_hidden_dim, time_embed_dim                          # Core diffusion dimensions and time embeddings
│   │   ├── diffusion_num_attention_heads, diffusion_transformer_layers...# Diffusion Transformer core processing capacities
│   │   ├── diffusion_head_hidden_dims, diffusion_steps, inference_steps  # Diffusion output bounds and noise progression steps
│   │   └── beta_start, beta_end, reconstruction_loss_weight              # Noise schedule bounds and auxiliary loss weights
│   │
│   ├── basic_vla.yaml                                                    # Vision-Language-Action policy configuration
│   │   ├── name, family, activation, dropout, token_dim                  # Core model identity and general token dimensions
│   │   ├── vision_token_dim, use_vision_tokens                           # Visual feature properties
│   │   ├── task: enabled, t5_model_path, max_text_tokens...              # Language task conditioning (e.g. T5 model paths)
│   │   ├── state_encoder_hidden_dims, vision_encoder_hidden_dims         # Input projectors for states and vision tokens
│   │   ├── num_attention_heads                                           # Number of attention heads across transformers
│   │   ├── observation_transformer_layers, policy_transformer_layers...  # Layer counts for observation and policy transformers
│   │   ├── med_branch.projector.type / predictor.type                    # MED codec choices for point-cloud/vector targets
│   │   └── action_decoder_hidden_dims                                    # Action output head MLP dimensions
│   │
│   └── minimal_mlp.yaml
│       └── (Contains standard MLP structure hyperparameters)
│
├── train/
│   └── smoke.yaml                                                        # Training loop and optimizer configurations
│       ├── run_name, epochs, learning_rate, weight_decay, device         # Global optimization parameters and target device
│       ├── validation_ratio, validation_seed                             # Validation subset sampling controls
│       ├── progress: enabled, leave, mininterval, postfix_update_interval# tqdm progress-bar behavior
│       ├── local_tracking: enabled, epoch_metrics_filename               # Local per-epoch JSONL metric persistence
│       ├── log_every_n_steps                                             # Legacy fallback for postfix update frequency
│       ├── save_last_checkpoint, save_best_checkpoint                    # Checkpoint saving toggles
│       └── wandb: enabled, project, entity, mode, group, job_type, tags  # Weights & Biases telemetry settings
│
├── experiment/
│   ├── fulltrain.yaml
│   ├── genmed.yaml
│   ├── gtmed.yaml
│   ├── pcvae.yaml                                                    # Single-frame tactile point-cloud VAE pretraining
│   └── vanilla.yaml
│       └── (Overrides main combinations. Direct parameters that combine data, model, and train configs above)
│
└── *.yaml (Root level)
    ├── train_runner.yaml
    ├── vanilla.yaml
    ├── gtmed.yaml
    ├── fulltrain.yaml
    ├── genmed.yaml
    └── pcvae.yaml
        └── (Top-level entry point files used directly to run training scripts. Uses `defaults` to assemble above modules)
```

## MED Configuration Conventions

- `data.raw.sequence.med.kind` controls whether the selected target is `vector` or `point_cloud`.
- `data.raw.sequence.med.side` can auto-fill the default right/left tactile field names.
- `data.raw.sequence.med.emit_intermediate=false` can be used by evaluation-only flows to skip constructing raw future-med tensors in the dataset.
- `model.family` stays focused on the observation backbone: `mlp`, `vla`, or `diffusion`.
- `model.med_branch.projector.type` is used for MED inputs in `gtmed/fulltrain`.
- `model.med_branch.predictor.type` is used for MED outputs in `genmed/fulltrain`.
- `pointnet` is only valid for point-cloud MED projector branches.
- `point_query` is only valid for point-cloud MED predictor branches.
- `pcvae` pretrains a single-frame tactile point-cloud VAE. The VAE uses `[137, 5]` input when `include_valid_feature=true`, reconstructs only `(x,y,z,value)`, and masks invalid points in the loss.
- `TactilePointVAE_Latent_Med` converts raw point-cloud windows such as `[8,137,5]` into latent windows such as `[8,32]`. Camera and wrist-local presets/checkpoints must be kept separate.
- `train.validation_ratio` only affects the validation split and can be used to randomly subsample large real-data validation sets.
- `data.raw.loader.val.*` can override DataLoader behavior only for validation, for example a lower worker count or disabling timeout.
