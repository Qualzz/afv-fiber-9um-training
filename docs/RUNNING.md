# Running locally

This release preserves the actual training files rather than replacing them with a newly written trainer. It excludes CT caches, remote execution, transfer scripts and machine supervisors.

## Dependencies

Historical run configurations record PyTorch 2.1.2 with BF16 CUDA training. The training host inspected for publication had Python 3.10.13 and PyTorch 2.1.2+cu121. Vendored dependencies included `dynamic-network-architectures==0.4.4`, NumPy 1.26.4 and SciPy 1.11.4; Pillow was 10.0.1. `requirements-training.txt` records these observed versions. This is not an exhaustive historical lockfile; see `provenance/environment-inspection.json`.

Use an appropriate CUDA-enabled PyTorch installation for your GPU. Install the remaining training requirements into the same Python environment. Label preparation additionally requires scikit-image with Lee skeletonization support; its original preparation-host version is not recovered or pinned here.

## Paths and inputs

V3–V5 use their script directory as the training root. They expect `weights/plans.json`, prepared `tiles/<region-id>/` inputs, and the run's `snapshot.json`, `validation.json` and `validation/*.npz`. Initial model weights are not included. V3 full initializes from expanded-01 best EMA at 1200. The earlier initial run starts from the published base backbone. V3 and V4 first load `weights/checkpoint_final.pth` even when `--initialize-from` subsequently replaces all heads. V4 additionally reads `models/358_finetuned_final.json`; its exact receipt is included at `provenance/358_finetuned_final.json` and must be placed at that expected local path. Supply `weights/plans.json` from the base model. V4 and V5 require the previous-phase checkpoint as described in the training chronology.

V6 `settings.py` retains the historical local paths `E:/fiber-full-8000-v5` and `D:/fiber-compression-multiscroll-v4`. These are local filesystem roots, not remote endpoints. Change `SOURCE` to the local V5 directory and `PARENT` to the local V4 directory when adapting the code. `SOURCE_RUN` must point at a V5 run containing the step-874 checkpoint with the recorded initialization hash. Its sampler loads V5 `data.py` and `material.py`. Dependencies may be installed normally; the historical `dependencies` path insertion can be removed in an adapted copy.

The V5/V6 snapshot includes absolute local region folders. Remap these folders to your prepared inputs. Changing settings or manifests changes their hashes, so do this in a working copy and record new provenance. Do not overwrite the archived configs to imply an identical original run.

## Training entrypoints

Examples of the original CLI interface, assuming all local inputs and parent checkpoints have been supplied:

```bash
# V3: inside original/fiber-compression-pherc0358-v3
python train.py full --steps 3000 --initialize-from /path/to/initialize-from-expanded-01.pt --workers 6

# V4: inside original/fiber-compression-multiscroll-v4
python train.py multiscroll --steps 12000 --initialize-from /path/to/358_finetuned_final.pt --validate-every 100 --workers 6

# V5: inside original/fiber-full-8000-v5
python train.py multiscroll-full-v5 --steps 12000 --initialize-from /path/to/50M_MULTISCROLL_8000.pt --learning-rate 2e-5 --validate-every 250 --checkpoint-every 50 --patience 20 --workers 8

# V6: inside original/fiber-centre-compression-v6, after remapping settings
python prepare.py
python train.py --workers 8
# For a continuation of a locally created V6 run:
python train.py --resume --workers 8
```

These demonstrate interfaces, not a promise that commands work against the metadata-only clone. V4 was configured for 12000 steps and stopped at 8000; specifying 8000 as its maximum would change the learning-rate schedule. V5 was configured for 12000 and paused at 874 before V6 began. Historical run directories contain receipts and metrics, not their model checkpoints or input arrays. Use a separate run directory or a working copy for a new experiment. Inspect each archived config for the actual parameters; CLI defaults may differ. Full recreation also needs the earlier parent checkpoints, which this repository does not distribute.

## Export

`original/hf-export/export.py` is the actual script that converted the portable V6 EMA into the four-class nnU-Net checkpoint layout. Its source paths refer to the author's original workspace; remap `SOURCE`, `OFFICIAL` and `OUT` for your own export. It validates the portable checkpoint SHA and backbone keys/shapes against the base model. Its use of `hashlib.file_digest` requires Python 3.11 or later (the export environment differed from the training host).

The public checkpoint is already available on Hugging Face. Download it for inference rather than rerunning all phases. No training or CT download is started by the repository's verification tools.

## Independently verifying released weights

After obtaining the public checkpoint, run (CPU only):

```bash
python tools/verify_checkpoint.py --checkpoint /path/to/checkpoint_final.pth
```

This checks the public full-file SHA and all 1160 tensor hashes/dtypes/shapes against the published manifest. With the original V6 `last.pt`, adding `--source /path/to/last.pt` also checks the EMA byte identities and embedded run config. Without that source checkpoint, the source-side hashes remain author-supplied evidence, not independent attestation of the training history.

The release inference plan uses 128³ patches instead of the base model's 256×256×224. Standard nnU-Net normalization is per inference case and does not reproduce every training sampler's normalization. Mirroring is allowed by the checkpoint although the fine-tuning sampler does not add flips. Loader defaults and precision can affect the probabilities; exact inference parity is not implied by weight identity. `provenance/inference-comparison-method.json` documents the older comparison receipt.

V5's validation preparation depended on excluded setup scripts; the exact frozen validation arrays are not included. The archived validation manifest describes those inputs but cannot recreate them alone. A fresh clean-environment installation has not been certified; the dependency versions are observed historical versions.
