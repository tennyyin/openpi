# openpi bike-rotor fine-tuning on SageMaker (cv-wfm / H200)

Full fine-tune of **pi0_base** and **pi05_base** on the TRI/LBM `BimanualBikeRotorInstall`
task (bimanual dual-Panda). Configs live in `src/openpi/training/config.py`:
`pi0_bike_rotor`, `pi05_bike_rotor`.

- **Cameras** (pi0's 3 slots): `base_0_rgb`=scene_right_0, `left_wrist_0_rgb`=wrist_left_plus,
  `right_wrist_0_rgb`=wrist_right_plus.
- **State**: 16-d measured joint state (jpos L/R + gripper L/R).
- **Actions**: 20-d cartesian `xyzrot6g` (absolute EE pose+gripper per arm). No delta transform;
  normalization handles scale.
- **Data**: 534 teleop demos only (rollouts excluded).

## One-time local prep (do this before launching)

```bash
cd ~/workspace/openpi

# 1. Convert raw LBM teleop demos -> LeRobot dataset (~520k frames, h264 video).
#    Parallel: 32 shard processes over disjoint episodes, then auto-merge into one dataset.
#    (Single-process fallback: convert_bike_rotor_to_lerobot.py --repo-id tri/bike_rotor_cartesian)
bash examples/bike_rotor/run_conversion_parallel.sh tri/bike_rotor_cartesian 32

# 2. Compute normalization stats for BOTH configs, straight from the raw lowdim (no video
#    decode -> no local ffmpeg/torchcodec dependency, and much faster). Writes
#    ./assets/{pi0,pi05}_bike_rotor/tri/bike_rotor_cartesian/norm_stats.json, baked into the image.
#    (Equivalent to scripts/compute_norm_stats.py for these configs since we apply no delta transform.)
uv run examples/bike_rotor/compute_norm_stats_from_raw.py

# If you prefer the stock path instead, it needs a working video backend (system ffmpeg):
#   uv run scripts/compute_norm_stats.py --config-name pi0_bike_rotor
#   uv run scripts/compute_norm_stats.py --config-name pi05_bike_rotor

# 3. Stage dataset + base checkpoints to S3 (mounted offline by the jobs).
bash sagemaker/stage_to_s3.sh tri/bike_rotor_cartesian
```

## Launch (submits to the cv-wfm p5en H200 queue)

```bash
export SM_USER=tenny            # namespaces the ECR repo / job / S3 paths
export WANDB_API_KEY=...        # or have `wandb login` populate ~/.netrc

# pi0 (first launch builds+pushes the image; reuse it for pi05 with BUILD_TYPE=None):
bash sagemaker/run_sm.sh pi0_bike_rotor  bike-rotor-pi0  cv-wfm full
bash sagemaker/run_sm.sh pi05_bike_rotor bike-rotor-pi05 cv-wfm None
```

Add `--dry-run` (as an extra flag) to print the payload without submitting. Monitor with
open-world's `sagemaker/sqm.sh` (these are AWS Batch service-jobs).

## Validation loss
Both configs set `val_fraction=0.05`: a deterministic **episode-level** 5% holdout (27 of
534 episodes, seeded so train/val never share an episode → no frame leakage). Every
`val_interval` (1000) steps the trainer averages `num_val_batches` (20) no-grad batches and
logs `val_loss` to wandb (uses EMA params when available). Norm stats are computed over all
534 episodes (standard; the 5% holdout is negligible for stats). To disable, set
`val_fraction=0`.

## Notes
- **Full fine-tune**: no LoRA variant, no freeze filter. pi0 uses z-score norm; pi05 uses
  quantile norm (set automatically by model type).
- Base weights are passed via `--weight-loader.params-path` (mounted `base_ckpt` channel),
  overriding the config's `gs://` path so training needs no GCS egress.
- Checkpoints land in `.../sagemaker/<user>/<job>/checkpoints`. `action_horizon=16`,
  `batch_size=32`, `num_train_steps=30000` — tune in the config as needed.
- If a job OOMs, `--fsdp-devices` already defaults to the GPU count; also confirm
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.
