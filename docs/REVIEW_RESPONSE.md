# Responses to the first independent review

Reviewer: Cursor Claude Opus 5.5 High. Historical training code remains unchanged.

- **B1:** Corrected lineage: base → initial (stopped 1199, best EMA 1100) → expanded-01 (stopped 1283, best EMA 1200) → full V3 → V4 → V5 → V6. Recovered initial/expanded configs, snapshots, metrics and phase receipts are under provenance/pre-v3. Their tracked core source hashes match the V3 source files.
- **B2:** Removed the misleading arbitrary-best wording. Disclosed best EMA step 2624 and actual baseline/best/final table, final semantic/legacy regressions, patience stop, and lack of a recovered selection rationale for releasing final instead of best. Model was not changed.
- **B3:** Explicit ancestor teachers and circular validation/model-selection dependence. No independent human-labeled test accuracy claimed.
- **B4:** V3 initialization included; V4 nonexistent flags removed; V4/V5 maximum 12000 restored; V5 workers 8; required base checkpoint and V4 model receipt documented. Frozen validation/preparation inputs remain required and absent.
- **B5:** Recovered the training-host manifest, exact recorded SHA 553fc3e9…, with all 136 parsed records identical to the local differently formatted copy. Recovery receipt included; verification not weakened.
- **E1:** Verification now reads historical configs directly, verifies all phase snapshots/prepared manifests, V3 phase validation, V6 initialization/embedded config, checkpoint transition receipts, original HF metadata Git blobs, and 393 exact recovered/reconstructed metadata receipts. Explicit failure collection works under python -O. 1364 consistency checks passed, including 27 run-tracked core-file checks (19 V3–V6 + 8 pre-V3).
- **E2:** Added CPU verify_checkpoint.py, actual 1160 per-tensor dtype/shape/byte hashes, extracted original checkpoint config. Verified against local public file SHA and original V6 EMA. A third party can verify public-side tensor hashes; source history requires source checkpoint access and otherwise remains author-supplied evidence.
- **E3:** Recovered all 393 region metadata records, checked metadata hashes against snapshot, retained byte-exact metadata receipts; fetched source metadata confirms original source codecs and Volcomp q=8. 54 compressed regions include all 6 PHerc1447 validation regions. Proposed corrected HF card is drafted separately; historical card stays intact and remote card has not changed.
- **E4:** Class 3 has no positive targets and receives negative CE pressure on known voxels; explicitly disclaimed as a validated intersection output.
- **E5:** Documented patch-size/normalization/mirroring/precision differences and historical comparison method. 4.9% any quantized-channel differences is not 4.9% classification error; argmax differences are about 0.03%.
- **E6:** Export receipts identify V3 final EMA at 3000 and V4 final EMA at 8000. V5 raw step 874 was manually stopped and not itself validated (latest EMA validation 750).
- **Other:** Loss coefficient, augmented CT subvoxel smoothing/centre restriction and compacted-patch normalization corrected. Exact selection formula and 16-case guard scope documented; legacy cases excluded from selection/guard. PHerc1447 core boxes have no overlap; nearest train/validation core gap 1915.73 voxels, not an independence proof. Extraction described in prose, with exact regeneration limits. No fresh clean-environment or independent accuracy certification claimed.

The original trainer archive is not rewritten to erase limitations. Source release readiness means accurate, inspectable disclosure, not a claim of scientific validation or guaranteed prize eligibility.

## Follow-up review corrections

Cursor Opus 5.5 High's second review judged the archive publishable after five traceability corrections, while retaining the disclosed scientific limitations. Those corrections were applied:

1. Inventory now includes 492 files, with separate kinds for historical-hash-anchored manifests/metadata, recovered author receipts, checkpoint-derived receipts and fetched public metadata. Per-case metrics and export receipts are explicitly author receipts, not externally attested history.
2. Verification checks expanded-01 and initial validation against the phase receipt, and parent hashes in the training chronology against recovered receipts/configs. All 2221 checks passed with `python -O`; historical source files remain unchanged.
3. Manifest recovery identifies the original Windows training host and path, with the exact byte difference: 4228 LF endings versus CRLF endings, 4228 extra bytes. After line-ending normalization the files are byte-identical, including spaces and float representations.
4. Initial backbone initialization is explicitly a code/teacher-hash inference without an initialization receipt.
5. Real validation means undeformed: PHerc0813 original CT and PHerc1447 q8 CT, not exclusively uncompressed sources. Training docs and proposed model card say this.

Additional follow-up: upstream Hugging Face metadata declares Apache-2.0 at revision 0905e68f14b33d1b98fd32d726f2c97607dea80c; its receipt is included without claiming a legal rights audit. The exact CT provenance of the old export-comparison cubes is explicitly unverified, rather than inferred from native pitch. Pre-V3's use of PHerc0358 is corrected in DATA.md. Nothing has been published or updated remotely.

## Final focused review

Cursor Claude Opus 5.5 High completed a third, focused review of these changes and found no remaining concrete publication blocker for an honest source archive. The reviewer did not execute checks; the author-side 2221 checks are reported separately. Two non-blocking retouches were applied: PHerc0175A training used q8 only in part (24 regions), and the private host-address fragment was removed from the recovery receipt. This agreement does not certify independent accuracy, turnkey reproducibility or prize eligibility. Publication was paused during review; the user subsequently requested adding the comparison to Hugging Face and the GitHub release.
