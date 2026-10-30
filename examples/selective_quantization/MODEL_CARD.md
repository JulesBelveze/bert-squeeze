---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: dmis-lab/UltraBERT-150M
language:
- en
datasets:
- JulesBelveze/tldr_news
tags:
- bert-squeeze
- selective-quantization
- newsletter-section-classification
---

# UltraBERT newsletter-section classifier: quantization study checkpoint

This is the selected **FP32 fine-tuned checkpoint**, not the pretrained fill-mask model or a quantized artifact. It is intended to reproduce the selective dynamic INT8 case study. Custom model code is included and requires `trust_remote_code=True` with the pinned Transformers 5.16.1 runtime.

## Source and training

Fine-tuned from `dmis-lab/UltraBERT-150M` at revision `6563895568657abedabf6478435a250080c00440`, using the frozen `JulesBelveze/tldr_news` revision `5ee6c613484aa6ac96fac7827a5659409897cc5b`. The dataset was normalized for exact-text deduplication, conflicting-label groups excluded, and split 80/10/10 with seed 42: 9,764 training, 1,220 validation, and 1,221 test examples.

Three epochs, batch 32, maximum 256 tokens, AdamW learning rate 3e-5, betas (0.9, 0.98), epsilon 1e-6, weight decay 0.1, 5% warmup followed by linear decay, and gradient clipping at 1.0. Best full-validation macro-F1 selected the checkpoint. Recorded training run: `ultrabert-full-db1117fb7a32`, 918 optimizer steps.

Label IDs:

| ID | Mapped newsletter section |
| --- | --- |
| 0 | Business & Strategy |
| 1 | Guides & Resources |
| 2 | Headlines & News |
| 3 | Research & Engineering |
| 4 | Security |

## Recorded results and limitations

At batch 16 with fixed padding to 256 tokens, FP32 test macro-F1 was 0.7723174. An indiscriminate x86 dynamic INT8 conversion scored 0.2889913 on the same test split. Subsequent quantization choices used a 64-example validation diagnostic, not the test examples. The fixed exemption is `ultrabert.layers.0.mlp.down_proj`.

On the remaining 1,156 validation examples, FP32 macro-F1 was 0.7747189. The exempted INT8 recipe scored 0.7644595 at batch 16/fixed padding, 0.7717992 at batch one/fixed padding, and 0.7683864 at batch one/actual length. No test score is claimed for the exempted recipe. Full validation had already been used for checkpoint selection, so the remainder is not an independent model-selection holdout. INT8 predictions vary with batching and padding.

This target merges original newsletter sections; it is not a clean semantic topic annotation. Exact-text deduplication leaves some near-duplicate and source-URL overlap across splits. This study uses one training seed and does not establish future-newsletter generalization or a universal module exemption.

## Reproduction

Use `examples/selective_quantization/reproduce.py` in [bert-squeeze](https://github.com/JulesBelveze/bert-squeeze), with this release pinned to an immutable Hub revision. The script checks `release-manifest.json`, reconstructs the frozen data from its pinned public parquet, reproduces the diagnostic interventions, and evaluates the frozen recipe on the validation remainder. Latency depends on hardware; model predictions can also vary across backends. Use x86 to compare against the recorded quantization study.

The packager verifies architecture, label mapping, weight-file size, and file hashes. It does not infer the provenance of arbitrary weights or rerun quality; these recorded scores describe the original run. Inspect the reproduction report before treating a repackaged checkpoint as validated.

## Attribution

The upstream model card identifies UltraBERT as Apache-2.0 licensed. This checkpoint retains its model code, configuration, and tokenizer files. See `LICENSE` and `NOTICE`; no source-newsletter text is redistributed in this model package. The study metadata records source revisions, training settings, and the release file hashes.
