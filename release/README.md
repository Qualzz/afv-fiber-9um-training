---
license: apache-2.0
base_model: scrollprize/fiber_hz_vt
tags:
  - vesuvius-challenge
  - nnunet
---

# fiber_hz_vt fine-tuned for ~9 µm scans

[`scrollprize/fiber_hz_vt`](https://huggingface.co/scrollprize/fiber_hz_vt) fine-tuned on native full-scroll scans at 8.64 µm and 9.362 µm. Same architecture, classes and file layout: tools that load `fiber_hz_vt` can load this repository.

Output classes: `0` background, `1` vertical fiber, `2` horizontal fiber, `3` intersection.

**Test-time mirroring** is enabled by default, as in `fiber_hz_vt`. It gives visibly better predictions but makes inference about 8× slower. For speed, pass `--disable_tta` to `nnUNetv2_predict` or `vesuvius.predict`.

**Training scans:** PHerc0125, 0191, 0211, 0257, 0358 (9.362 µm) and PHerc0175A, 0268, 0306B, 0800, 1447 (8.64 µm). PHerc0813 was used for validation only; PHerc0343 was never used. Labels are automatic pseudo-labels, not human-verified, and intersection regions were ignored in training.

Only evaluated on native ~9 µm scans.
