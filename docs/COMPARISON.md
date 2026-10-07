# Matched PHerc0125 3D comparison

This illustration compares the original ScrollPrize HZ/VT checkpoint with the published AFV V6 EMA, on the same central 256³ cube. It is a qualitative example from a training scroll, not a held-out accuracy estimate.

- Original checkpoint SHA-256: `a80d98f1baaaa09fac5e9d499ade91cb6f79982239cf8520fec9c217125ed899`.
- Published V6 checkpoint SHA-256: `6837f3fe8fca123e6ed0b91bb052ce5218ca6eb6889800e4d578b6c8aaf59767`.
- Native XYZ centre: (4032, 4060, 10913); core origin: (3904, 3932, 10785); shape: 256³; pitch: 9.362 µm.
- Input has a 32-voxel halo on each side, producing a 320³ inference region.
- Source: [original PHerc0125 CT](https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0/.zarray); verified compressor and filters are null. Source metadata and chunk SHA-256 receipts are included. Native CT scratch was bounded RAM storage and removed after inference/rendering.
- Identical 128³ patches, stride 64 in each axis, Gaussian probability blending, BF16 precision, no TTA, and shared input-region mean/std (std floor 10). These are matched comparison settings, not a claim that every official inference path uses identical normalization.
- Display: argmax vertical/horizontal class with confidence ≥0.60. Background and intersection class 3 are hidden. No spline extraction, MIP or surface smoothing is used.
- Same orthographic camera and categorical nearest-neighbour voxel rendering; blue = vertical, orange = horizontal.

![Matched voxel predictions](../assets/fiberpred-pherc0125-256.png)

[Source, chunk and input provenance](../provenance/comparison-pherc0125-256/provenance.json) · [Original inference](../provenance/comparison-pherc0125-256/original-provenance.json) · [V6 inference](../provenance/comparison-pherc0125-256/ours-provenance.json) · [Render parameters](../provenance/comparison-pherc0125-256/render-provenance.json).

## Training exposure

PHerc0125 contributes **25 of the 356 training regions (7.02%)** in the frozen final V5/V6 corpus, and **7.863 of 99.526 metres of pseudo-labelled source fibres (7.90%)**. The V6 sampler selects uniformly among nine training scrolls, so its expected share of sample proposals is **11.11%**. These are final-corpus and V6 sampling proportions, not cumulative shares across all fine-tuning phases or base-model pretraining. The displayed cube, including its 32-voxel inference halo, does not overlap any preserved PHerc0125 training-region box in V4–V6. This does not establish independence from base-model pretraining or correlations within the same scroll; the image remains a qualitative comparison, not an independent test.

[Counts and geometric overlap checks](../provenance/comparison-pherc0125-256/training-share.json).
