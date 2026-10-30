# Selective dynamic INT8 for a sequence classifier

Compare FP32, all-Linear INT8, and an explicit Linear-module exemption on your labelled examples. Export and reload the quantized artifacts, checking logits under three batching/padding policies. No training or automatic search takes place.

This is a small PyTorch example hosted in bert-squeeze. It does not use the Lightning quantization callback or require installing the full library. Inline dependencies pin the Torch 2.8 eager dynamic path and Transformers 5.16.1 used by the study, avoiding the project's separate adapter dependency boundary. This is not a comparison with other quantization runtimes.

## 1. Run a public-checkpoint demo

From the **bert-squeeze repository root**:

```bash
uv run --no-project --script examples/selective_quantization/compare.py \
  distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  --revision 714eb0fa89d2f80546fda750413ed43d93601a13 \
  --data examples/selective_quantization/sentiment-demo.jsonl \
  --exclude distilbert.transformer.layer.0.ffn.lin2 \
  --output /tmp/selective-int8-demo
```

The [public model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/714eb0fa89d2f80546fda750413ed43d93601a13/config.json) is a fine-tuned binary sentiment classifier: label 0 is negative and 1 positive. The eight included sentences are handwritten **plumbing examples**, not a benchmark or independent quality evaluation. The exclusion demonstrates named selection; we have not established that it improves this model. This demo does not reproduce the UltraBERT article's scores.

The first run downloads dependencies and model weights. Model download and output artifacts require several hundred MB of disk space. The script uses two Torch threads and selects x86/fbgemm/QNNPACK according to CPU availability; results from different backends are not interchangeable. It refuses to overwrite an output directory. Choose a new path for another run.

## 2. Use your own checkpoint and validation data

```bash
uv run --no-project --script examples/selective_quantization/compare.py \
  /path/to/finetuned/checkpoint \
  --data /path/to/validation.jsonl \
  --exclude ultrabert.layers.0.mlp.down_proj \
  --trust-remote-code \
  --output /tmp/ultrabert-selective-comparison
```

Use the exclusion and custom-code flag only for the appropriate trusted checkpoint. `--exclude` accepts exact names from `model.named_modules()` and may be repeated. A parameter name such as `classifier.weight` is rejected. Omit exclusions to compare only FP32 with all-Linear INT8. Native Hub checkpoints require a full revision commit; custom-code checkpoints must be local directories with their Python source files beside the HF config/tokenizer/weights, as in the study's saved checkpoint.

Input is JSONL with a nonempty `text` string and integer `label` matching the checkpoint's `id2label` mapping. The example supports single-label sequence classification, right padding, CPU execution, and a 256-token cap. All labels are included in macro-F1, with zero for absent classes; `label_counts` makes the supplied sample's support visible. Use genuinely representative validation data before selecting a recipe. Full batch-one evaluation can take many minutes on a large dataset; there is no cloud job or automatic retry.

## 3. Read the comparison

`comparison.json` records the source revision, input checksum, label mapping, backend, conversion counts, serialized deployment bytes, and predictions/quality under batch 16 with fixed padding, batch one with fixed padding, and batch one at actual length. It counts prediction changes against FP32 under the same policy and against each variant's own batch-16 predictions. A similar aggregate F1 score does not establish request-level equivalence.

`int8_all/` and, when requested, `int8_selective/` contain packed state dictionaries, tokenizer/config files, and exact module-selection metadata. The loader reconstructs that selection before loading weights. Reloaded logits must match pre-export logits under `torch.testing.assert_close` for every supplied example and policy. `fp32/` holds the reference checkpoint. Serialized bytes do not measure resident memory. This example does not benchmark latency; keep quality, padding policy, thread count, and actual inference timing together when making a deployment decision.

## 4. Reproducing the article's exact measurements

The [study runner and release instructions](STUDY.md) rebuild the frozen dataset and evaluate the measured checkpoint without the original private runner. The real checkpoint passed a fresh x86 package check; public checkpoint publication and the public-download check are still pending; the study manifest leaves the public model pin unset until those steps are completed.


The measured fine-tuned UltraBERT weights are not distributed here. The public demo and your own checkpoint can exercise the same conversion/export/evaluation workflow, but cannot establish the article's numerical result. The exact study also depends on its frozen data, split, and selected checkpoint.

For the study owner, the original checkpoint can be downloaded from the existing Modal volume without retraining:

```bash
modal volume get ultrabert-pilot-results \
  ultrabert-full-db1117fb7a32/best \
  /tmp/ultrabert-tldr-checkpoint
```

That command requires access to the experiment volume and does not publish anything. A public checkpoint release remains a separate step. The module exemption is a finding for that checkpoint, not a recommended default for other encoders.
