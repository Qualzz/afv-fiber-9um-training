# Data and pseudo-labels

The final frozen V5/V6 corpus contains:

| Split | Regions | Trace fragments | Recorded fragment arclength |
| --- | ---: | ---: | ---: |
| Train | 356 | 55,708 | 99.526304 m |
| Validation | 37 | 6,274 | 11.122821 m |
| Total | 393 | 61,982 | 110.649125 m |

Arclength sums observed trace fragments; partial overlaps can be counted. These figures are not certified unique physical fibre metres.

Final training scrolls: PHerc0125, PHerc0175A, PHerc0191, PHerc0211, PHerc0257, PHerc0268, PHerc0306B, PHerc0800 and PHerc1447. The pre-V3 and V3 phases additionally used PHerc0358. PHerc0813 is held out for validation; additional PHerc1447 validation regions are spatially separate from training regions. PHerc0343 is excluded from training; no test-performance claim is made here.

The region identifiers, split membership, input-file hashes and teacher checkpoint hashes are recorded in `original/fiber-centre-compression-v6/runs/multiscroll-centre-v6/snapshot.json`. Validation locations, donor IDs and augmentation metadata are in its `validation.json`. V6 retains the V5 corpus unchanged.

Source regions have native pitches of 9.362 or 8.64 µm. Earlier corpus collection read original CT; the newly added V5 PHerc1447 and PHerc0175A regions used native-grid Volcomp q8 CT. The added teacher-inference inputs were 576³ with a 32-voxel halo, cropped to 512³ cores. Thus "native grid" does **not** imply that every training input was uncompressed. This repository concerns fibre detection, not ink detection.

## Partial-label recipe

`annotation.build(predictions, traces)` takes teacher probability arrays and accepted trace geometry. The probability-array layout is vertical, horizontal, intersection, background; this differs from the model's class-index order. The original function is included in V4 and V5.

Known background is assigned where the quantized teacher background probability is at least 242. Family support uses its probability plus half the intersection probability, thresholded at 153. A Lee skeleton is extracted from that support. Accepted trace points seed skeleton ownership within 1.5 voxels, but **all** skeleton branches compete for nearest ownership. This prevents an accepted fragment from labeling an entire connected bundle through region growing.

Only supported voxels owned by accepted traces receive positive instance labels. Cross-family overlap and intersection-dominated voxels (intersection probability at least 128) are ignored. Unknown semantic targets are 255; unknown instance IDs are 0. Donor traces contain at least 30 points. Geometric inferred bridges are not positive training targets.

The sampler chooses scrolls uniformly, then horizontal/vertical families uniformly, then fragments proportional to observed arclength within that group. Patches are sampled near observed trace points. The initial single-scroll phase samples families and fragment arclength without a scroll selection step.

## Required local input format

Each prepared region has:

- `volume.u8`: uint8 CT, shape 512×512×512 in Z/Y/X order.
- `semantic.npy`: uint8 partial semantic labels.
- `instances.npy`: instance labels (uint16 when prepared).
- `centre.npy`: reference centre support.
- `corpus.json`: trace geometry including `pointsLocalXYZ`, `lineLocalXYZ`, family and fragment arclength.
- `hzvt-classes.json`: region normalization statistics.

The loader also supports earlier `teacher-labels.npz` target bundles. These files and fixed validation `.npz` arrays are not uploaded here. Downloading public source scans alone does not reconstruct the exact automatic trace set. Corpus acquisition and network/cache machinery are intentionally outside this release.

## Ancestor teachers and evaluation dependence

The initial PHerc0358 pseudo-labels derive from the ScrollPrize base model. V4 uses V3 final EMA (`7195…`) both as teacher and initialization. V5 adds regions labeled by V4 final EMA (`6a3f…`), while retaining V3-teacher regions. V6 reuses that frozen corpus. Validation labels also derive from these ancestor teachers and guide model selection/early stopping. Consequently these scores measure agreement with model ancestry and are selection-biased; they do not independently establish fibre correctness or improvement over the base detector. No independent test dataset is supplied or reported.

`provenance/region-metadata.json` supplies the 393 recovered region metadata records, source URLs, voxel pitches, splits, teacher identities and hashes checked against the frozen snapshot. 54 regions use Volcomp q8: 48 training and 6 PHerc1447 validation regions. PHerc1447 validation is therefore also compressed CT. The other 339 records name original CT source URLs. `ct-source-metadata.json` records metadata retrieved from those sources at packaging time; it is not a claim that public source contents can never change.

The original corpus collection ran the current ancestor teacher on native-grid source regions, extracted family-specific probability support into skeleton paths and observed trace fragments, retained qualifying traces, then used the conservative ownership recipe above to build partial labels. V5 used 576³ teacher context, cropped to 512³ cores, threshold 60%, trace-point spacing 4 voxels and endpoint merge-gap limit 30 voxels. This extraction/preparation infrastructure is excluded at the author's request. Exact regeneration would require the original extraction implementation and frozen trace/label arrays; the training core does not substitute for them. Inferred gaps are not accepted positive-label donors.

PHerc1447 appears in both training and validation. `spatial-split-check.json` reports the minimum separation of their 512³ core boxes and any overlaps. Spatial separation does not remove within-scroll correlation, teacher circularity or model-selection bias. No held-out accuracy claim follows from it.
