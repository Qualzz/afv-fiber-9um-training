# Training recipe

## Model and objectives

The backbone is `ResidualEncoderUNet` from `dynamic-network-architectures==0.4.4`, initialized from `scrollprize/fiber_hz_vt`. Its architecture is recorded in `release/plans.json`. It takes one CT channel and produces four segmentation classes: background, vertical, horizontal and intersection. Two auxiliary heads predict a centreline score and six local instance affinities, ordered dz1, dy1, dx1, dz4, dy4, dx4.

Training uses 128³ patches from 512³ source regions on their native grid. Undeformed patches use region mean and standard deviation, with a standard-deviation floor of 10. V3–V5 compacted patches recompute patch statistics. V6 uses source-region statistics for both variants. No common-resolution resampling is applied.

The semantic objective combines cross-entropy with separate aggregate weighting for known foreground and background, and 0.5 times the sum of Dice losses for vertical and horizontal classes. Two coarse segmentation outputs contribute with weights 0.25 and 0.125. Coarse cells containing unknown fine-grid labels are ignored. The foreground recall objective at reference centre points has weight 0.25; balanced affinity BCE has weight 0.3. Unknown semantic label 255 and uncertain ownership are masked.

V3–V5 use a binary centre objective with coefficient 0.25. V6 replaces this with a Gaussian target of sigma 1.5 native voxels and coefficient 2.0. Its BCE is normalized separately in three valid distance zones: distance ≤1 voxel (weight 0.25), 1–6 voxels (0.65), and >6 voxels (0.10). See the second `loss_for` definition in V6 `network.py` for the exact implementation. Intersection-dominated voxels are ignored, but class 3 receives no positive targets and is penalized by cross-entropy on all known class-0/1/2 voxels. Its output should not be treated as a validated intersection detector after fine-tuning.

## Successive phases

| Phase | Initialization | Recorded endpoint | Peak LR / warmup |
| --- | --- | --- | --- |
| Pre-V3: initial, PHerc0358 | Base fibre backbone and new heads, inferred from code; no initialization receipt | Stopped at 1199; best EMA 1100 initializes next run | 5e-5 / 100 |
| Pre-V3: expanded-01, PHerc0358 | Initial best EMA; fresh optimizer | Stopped at 1283; best EMA 1200 initializes V3 | 5e-5 / 100 |
| V3, PHerc0358 full corpus | Expanded-01 best EMA, all heads; fresh optimizer | 3000 | 5e-5 / 100 updates |
| V4, multiscroll | V3 EMA; fresh optimizer | 8000, early stopped | 5e-5 / 100 |
| V5, expanded corpus | V4 step-8000 **final EMA**, all heads; fresh optimizer | Manually stopped at 874 | 2e-5 / 100 |
| V6, final centre/compaction objective | V5 step-874 **raw model**; optimizer and EMA reset | 7624 = 874 + 6750 | 2e-5 / 50 |

The numbers are run counters, not a single cumulative counter across every phase. V6's configured maximum is 12000; the released model is its final EMA at 7624. V6 ended by its early-stopping patience. The published final EMA is **not** the best EMA according to the run's predefined validation score; that was at 2624. No documented validation-based rationale for publishing the final EMA instead has been recovered. The release is identified faithfully rather than claimed to be optimal. The V5 raw model at 874 was the continuation state; its latest EMA validation was at 750, not a validation of those raw weights.

All phases use AdamW, weight decay 1e-4, batch size 2, four-step gradient accumulation (effective batch 8), BF16 autocast, gradient norm clipping at 1 and seed 3582026. All six recorded runs use cosine decay with a 2% learning-rate floor, tied to each run's configured maximum rather than its eventual stopping counter. Its EMA decay is `min(0.999, 1 - 1/(phase_step + 1))`. Sampling is deterministic by sample index; CUDA training is not claimed bitwise reproducible on different hardware.

The pre-V3 runs use the same four core source files as V3: their recovered run config hashes match. Their metadata and available metrics are under `provenance/pre-v3`; their CT/label arrays and checkpoints remain absent. `earlyStopped: true` in old status receipts can simply mean the run ended before its maximum; `stopped` indicates a manual stop, not convergence.

Initial-run base initialization is inferred from `model.initialize` in the preserved code and the base teacher SHA `a80d98f1baaaa09fac5e9d499ade91cb6f79982239cf8520fec9c217125ed899`. There is no initial-run initialization receipt, so this inference is not an attested checkpoint transition.

Parent checkpoint hashes:

- Initial best EMA (step 1100): `5dd2dbe447e3940d337751e236bc88ac64888f02592f74a87e4a5fbbd12828c5`
- Expanded-01 best EMA (step 1200): `6bcb9070d5c890fce66511e86f80b14e5fe7e4536b681e48c31faed109759903`

- V3 EMA: `7195cdaa831700d72c1fb3da98c8ce71d8ea87659d026a9f26c358b759f4ba59`
- V4 step 8000: `6a3ff424f67c5619a12b18fcb2598fbc36961fd851f9fb67e4d4d0c37d3e458b`
- V5 step 874: `847aa6c9b153d947a1c0939e7fb91b6f3021b5c38c5a4ed1590034dbfee5411e`
- V6 training checkpoint: `f0e8caf18cf4c4f5c576d0247fe450294c210bf0171f39517a1fe08f77dae4e9`
- Portable V6 EMA with auxiliary heads: `144a33b154c440ab4990bf05953e454a6cd8167cd28a8a846746a64cfd605498`

## Compression augmentation

### Earlier phases: V3–V5

`material.py` constructs examples from real local sheet-frame CT slabs, using same-scroll donors in the multiscroll phases. It compacts material with factors sampled from 0.15–0.65 and thresholds from 50–80, adds smooth spatial variation and masks uncertain seams. `deform` adds bending/shear and mild noise. These earlier slab-based augmentations differ from the final single-region V6 warp.

### Final phase: V6

Each augmented example starts from **one contiguous source region**. A local sheet normal is estimated from smoothed CT gradients in a 25³ neighbourhood. A smooth, strictly monotone mapping along that normal closes empty gaps first and compresses material for hard cases. Density guides the mapping, while sampled source CT supplies its texture. CT is inverse-warped and spline coordinates are forward-warped with the same mapping. Augmented CT averages three linearly interpolated subvoxel samples, so it is smoother than the direct undeformed crop. Augmented centres are restricted to [208,303] per region axis to retain context. Centre targets are rebuilt from transformed curves. Inconsistent semantic/instance labels are ignored.

Half the proposals are undeformed. Within augmented proposals, the hard fraction ramps from zero at phase update 100 to 40% at update 750. After that:

| Severity | Fraction of augmented proposals | Material factor | Void factor |
| --- | --- | --- | --- |
| Mild | 20% | 1.0 | 0.30–0.60 |
| Moderate | 40% | 0.85–1.0 | 0.08–0.25 |
| Hard | 40% | 0.55–0.80 | 0.02–0.10 |

Augmentations with fewer than 16 centre voxels or coverage below 0.97 fall back to an undeformed sample. Consequently the accepted augmented proportion can be lower than 50%.

The included original geometry checks reduce a 55-voxel slab spacing to 17.21875 voxels while retaining an 8-voxel material thickness. The oblique forward/inverse round-trip error is approximately 0.01262 voxels. These checks validate the mapping, not material physics.

## Validation and stopping

V6 freezes 16 locations with four matched variants (real, mild, moderate, hard), giving 64 paired cases, and preserves 72 earlier cases unchanged: 136 total. Validation uses PHerc0813 and separate PHerc1447 source regions. The paired locations come from the validation partition. The cohort called **real** mixes original CT from PHerc0813 and Volcomp q8 CT from PHerc1447. Here real means **undeformed**, not uncompressed CT; mild/moderate/hard denote additional geometric deformation, not storage codecs.

Validation runs after the first 100 phase updates and every 250 thereafter. Checkpoints are saved every 25 updates. A retention guard requires real-case foreground Dice and centre coverage to remain within 0.03 of the baseline before best-checkpoint promotion. Stopping patience is 20 validation events after at least 1000 phase updates. The exact score and aggregation are in V6 `validate`; recorded configurations and available metrics are under each `runs` directory.

These metrics compare against teacher-supported pseudo-labels. They are not independent human-annotated accuracy measurements.

## Recorded model-selection results

Lower selection score is better. These are internal agreement metrics against ancestor-generated pseudo-labels, not independent accuracy. In the table, paired real means undeformed PHerc0813 original CT plus PHerc1447 q8 CT.

| Metric | Baseline, step 874 | Best score, step 2624 | Published final EMA, step 7624 |
| --- | ---: | ---: | ---: |
| Selection score | 1.814837 | 1.688139 | 1.792878 |
| Paired real semantic loss | 0.818786 | 0.905586 | 0.991064 |
| PHerc0813 paired real semantic loss | 1.202240 | 1.341829 | 1.421297 |
| Legacy compressed centre coverage | 0.930632 | 0.885110 | 0.733033 |
| Legacy compressed semantic loss | 1.596179 | 1.753335 | 2.189142 |

The selection formula is `real.semanticLoss + 0.5*compressed.semanticLoss + 0.5*real.centreLoss + 0.5*hard.centreLoss`. The 72 legacy cases do not enter this score or its retention guard. The guard checks foreground Dice and centre coverage on 16 paired real cases; it does not constrain every semantic loss or legacy cohort. Fixed mild/moderate/hard validation factors are (1,.45), (.9,.15), (.65,.04), inside the augmentation ranges. The final legacy regressions are explicitly disclosed above. There is no independent human-labeled test set or test-accuracy result in this release.

V3 final EMA and V4 final EMA are identified by their preserved export receipts (`provenance/358_finetuned_final.json`, `50M_MULTISCROLL_8000.json`), distinct from their earlier best checkpoints. The final V6 per-case receipts at 2624 and 7624 are also included.
