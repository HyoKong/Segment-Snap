# Released checkpoints

[Overview](../README.md) · [Data](../docs/DATA.md) · [Method](../docs/METHOD.md) · [Validation results](../docs/RESULTS_VAL.md)

The release provides three learned predictors for one 3D interaction-understanding pipeline: movable parts, dense handles, and joint part-handle proposals. The `t1`, `t2`, and `s2` directory names are retained because scripts and checkpoint paths use them. Download the published files from [Hugging Face: `imsuperkong/Segment-Snap`](https://huggingface.co/imsuperkong/Segment-Snap).

| Role | Size | Weight | Configuration |
|---|---:|---|---|
| movable-part predictor | 1.71 GB | `t1_spformer/model/epoch_14.pth` | [`t1_spformer/config.py`](t1_spformer/config.py) |
| dense-handle predictor | 1.50 GB | `t2_semantic/model/model_best.pth` | [`t2_semantic/config.py`](t2_semantic/config.py) |
| joint part-handle predictor | 1.71 GB | `s2_joint/model/model_last.pth` | [`s2_joint/config.py`](s2_joint/config.py) |

## Download and verify

From the repository root, download and verify all three weights:

```bash
python scripts/download_checkpoints.py --dest checkpoints
python scripts/download_checkpoints.py --dest checkpoints --verify-only
```

The downloader fetches three weights and their matching `config.py` files, then checks the
weights against the MD5 values bundled in the script. It requires `huggingface_hub`, included
in the [setup instructions](../README.md#quick-start). `--verify-only` checks existing weights
without downloading. A missing weight or digest mismatch fails the command. The weights are
intentionally gitignored.

<details>
<summary>Verification digests and exact byte sizes</summary>

| weight | bytes | MD5 | SHA-256 |
|---|---:|---|---|
| `t1_spformer/model/epoch_14.pth` | 1,705,908,057 | `6b44303b6b93618862f74f3620c5cb42` | `cbe543cd5229f635e99aba64450c3d384971d462b004be915891e766a7774930` |
| `t2_semantic/model/model_best.pth` | 1,504,787,251 | `2ab3ac49afee2a5fffcbf8f363e30a87` | `c96d78e53c047496276d59ab8ffdd0dc9f00a31193f8ce75e4245f2e347d87ad` |
| `s2_joint/model/model_last.pth` | 1,714,611,973 | `6a2c0867f8191b1d0b201a720e92b18b` | `c5f0be9c0667cd4a07e1f6420044bb964a1b08535dd8b44e8c1a20f3744144d6` |

</details>

## Which weights inference uses

The reproduction script reads each configuration from `checkpoints/<directory>/config.py`, beside the downloaded weight, rather than from `configs/`. It uses:

```text
t1_spformer/config.py  + model/epoch_14.pth
t2_semantic/config.py  + model/model_best.pth
s2_joint/config.py     + model/model_last.pth
```

All released checkpoints contain an EMA state dictionary, and `infer_t1.py` and `infer_t2_sem.py` default to `--weights ema`. Use `--weights raw` only for a compatible checkpoint that lacks or should not use EMA weights. The joint-child inference loader likewise selects `ema_state_dict` when it is present.

The released weights are sufficient for inference; they already contain the trained model parameters. The Volt-B initialization file is needed only for retraining. The training configurations expect it at `weights/volt-base-scannetpp.pth`:

```bash
mkdir -p weights
curl -L -o weights/volt-base-scannetpp.pth \
  https://huggingface.co/KadirYilmaz/Volt/resolve/main/Volt_experiments/joint_training_base/scannetpp/model/model_last.pth
```

## Training and selection disclosure

Articulate3D fine-tuning uses only the 195 training scenes; validation scenes do not contribute
fine-tuning losses or gradient updates. The predictors start from ScanNet++-pretrained Volt-B
weights. Validation was used for model, checkpoint, and hyperparameter selection, so this is not
a validation-blind release.

- The part checkpoint is `epoch_14`, the 14th of 20 evaluation checkpoints from the 400-epoch run. It was selected after training by the validation ranking column of the full pipeline, not by the trainer's segmentation criterion.
- The dense-handle checkpoint is `model_best`, selected by the trainer's validation mIoU criterion.
- The joint checkpoint is `model_last`, the final saved checkpoint.

See [Method: Training](../docs/METHOD.md#training) for architectures and training settings, and [Validation results](../docs/RESULTS_VAL.md) for reported scores.
