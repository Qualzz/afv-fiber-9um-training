# Attribution

The model was fine-tuned from [ScrollPrize's fiber_hz_vt](https://huggingface.co/scrollprize/fiber_hz_vt). Base model architecture metadata is included under `provenance/base-model`.

The upstream Hugging Face API declares `apache-2.0` at revision `0905e68f14b33d1b98fd32d726f2c97607dea80c`; the API receipt is `provenance/base-model/license-metadata.json`. This records an upstream declaration, not an independent audit of all upstream rights.

The architecture uses MIC-DKFZ's [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures), version 0.4.4. This dependency is imported, not vendored. Its own license applies to that package.

PyTorch, NumPy, SciPy, Pillow and scikit-image are external dependencies with their own licenses. CT and annotation data retain their source terms and are not distributed in this repository. The Apache-2.0 license covers the author's code here and does not relicense third-party weights or data.
