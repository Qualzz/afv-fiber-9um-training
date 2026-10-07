---
license: apache-2.0
base_model: scrollprize/fiber_hz_vt
tags:
  - vesuvius-challenge
  - nnunet
---

# Fibre detector fine-tuned on ~9 µm CT

Fine-tuned from ScrollPrize fiber_hz_vt with automatic partial pseudo-labels, centreline/affinity auxiliary losses and geometric compression augmentation. Sources include original CT and native-grid Volcomp q8 CT. Native pitch is 8.64 or 9.362 µm; native grid does not imply uncompressed provenance.

Training lineage: initial → expanded-01 → PHerc0358 full (V3) → multiscroll (V4) → expanded corpus (V5) → centre/void-first augmentation (V6). Published weights are V6 final EMA at step 7624, not the run's best-scoring EMA at 2624.

Outputs retain the base layout: background, vertical, horizontal, intersection. Class 3 received no positive fine-tuning targets and was penalized on known class-0/1/2 voxels; do not rely on it as a validated intersection detector. Only the segmentation backbone is exported.

Training scans span PHerc0125, 0175A, 0191, 0211, 0257, 0268, 0306B, 0358, 0800 and 1447. PHerc0813 and spatially separate PHerc1447 regions provide ancestor-teacher pseudo-label validation, not independent ground truth. No independent test accuracy is claimed.

The validation cohort named real means undeformed: it combines PHerc0813 original CT and PHerc1447 Volcomp q8 CT. Geometric compression augmentation is distinct from the source storage codec.

Inference plans use 128³ windows; loader normalization, precision and mirroring can differ from training. Mirroring is allowed; disabling it trades speed against predictions, without a quantified accuracy claim here.

Training source, recorded configurations and provenance: [Qualzz/afv-fiber-9um-training](https://github.com/Qualzz/afv-fiber-9um-training). The archive excludes CT cache and machine orchestration infrastructure; exact reconstruction requires frozen inputs and parent checkpoints that are not bundled.

## Matched 3D voxel comparison

![Original ScrollPrize HZ/VT (left) and published AFV V6 EMA (right), on the same central 256³ PHerc0125 CT cube](assets/fiberpred-pherc0125-256.png)

**Left:** original ScrollPrize HZ/VT. **Right:** the published V6 EMA. Blue denotes vertical fibre predictions; orange denotes horizontal fibre predictions. Both use the same original CT cube, camera, 60% confidence threshold, 128³ inference windows, 64-voxel stride, Gaussian probability blending, BF16 precision and no test-time mirroring. Rendering uses categorical voxels without smoothing or MIP.

The 256³ cube is centred at native XYZ **(4032, 4060, 10913)** in PHerc0125, at **9.362 µm/voxel**. PHerc0125 is present in the training corpus; this is a qualitative illustration, not a held-out accuracy benchmark. [Comparison provenance](https://github.com/Qualzz/afv-fiber-9um-training/blob/main/docs/COMPARISON.md).

