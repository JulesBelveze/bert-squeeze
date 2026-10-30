# Reproduce the UltraBERT case study

**Release status: the real checkpoint passed the fresh x86 package check; public checkpoint publication and the public-download check are pending.** `study.json` deliberately has `model_release: null`, so the no-checkpoint command fails clearly rather than substituting pretrained or random weights.

## Reader command, once the checkpoint release is pinned

From a fresh bert-squeeze checkout, on an x86 CPU environment supporting PyTorch's x86 quantization engine:

```bash
uv run --no-project --script examples/selective_quantization/reproduce.py \
  --output /tmp/ultrabert-study-reproduction
```

Once `study.json` records the published model repository, immutable commit, and manifest checksum, this downloads the measured checkpoint and pinned public parquet automatically. It verifies every release file hash, reconstructs the original deduplication/label mapping/split, and requires the exact frozen records checksum. No Modal account is needed by readers. The environment pins Python 3.12, Torch 2.8.0, Transformers 5.16.1, scikit-learn 1.5.2, and PyArrow 19.0.1.

The runner then checks the checkpoint's FP32 diagnostic score before further computation, reproduces the all-Linear / first-down-only / first-down-exempt interventions and rounded-FFN-weight control on the same ordered 64 examples, and evaluates FP32 versus the frozen exemption on the remaining 1,156 validation articles. The original ordering matters for dynamic quantization. No test examples are used for the final exemption and no training or automatic search occurs.

The final comparison checks all three original policies, verifies export/reload, and measures paired fixed-padding and actual-length latency on the same seeded 32 texts with three repeats. `report.json`, logits, timing samples, rebuilt data, and the quantized artifact remain in the output directory. It refuses to overwrite an existing output. Plan for **up to 60 minutes on a comparable two-core CPU**, several GB of temporary disk including caches, and model/dependency transfer time. It is a local process, with no cloud job or retry automatically launched. `--no-timing` omits only timing, not the batch-one quality checks.

This reproduces the headline targeted interventions, weight-rounding control, and final verification. The earlier discovery sweeps over groups/projections/individual blocks are retained as recorded evidence under `recorded-results/`; the runner does not rerun that entire search history. Latency is hardware-dependent, and backend/runtime changes can alter predictions. A mismatched FP32 diagnostic reference stops the run with a saved report to avoid benchmarking the wrong checkpoint. Matching that score is a sanity check, not a substitute for the release file hashes.

## Local packaged-checkpoint command

Before publication, or when the release is already downloaded:

```bash
uv run --no-project --script examples/selective_quantization/reproduce.py \
  --checkpoint /path/to/packaged-release \
  --output /tmp/ultrabert-study-reproduction
```

The checkpoint directory must contain `release-manifest.json`. Use `--source /path/to/train-00000-of-00001.parquet` to reuse an already-downloaded source file; its checksum is still mandatory. The eight-sentence DistilBERT demo in the main README is separate and cannot validate the article's result.

## Release preparation, for the study owner

1. Download the selected checkpoint from the existing experiment volume (no retraining):

   ```bash
   modal volume get ultrabert-pilot-results \
     ultrabert-full-db1117fb7a32/best \
     /tmp/ultrabert-study-source
   ```

2. Prepare the release folder from the bert-squeeze root:

   ```bash
   uv run --no-project --script examples/selective_quantization/package_checkpoint.py \
     /tmp/ultrabert-study-source /tmp/ultrabert-study-release
   ```

   The packager checks architecture, label order, and the recorded FP32 weight-file size, copies an explicit allowlist of inference files, adds the model card/license/attribution, and hashes the resulting files. It does not upload or prove numerical provenance by itself. Unlisted logs, credentials, optimizer states, and data files are not copied.

3. Run the local packaged-checkpoint command above on x86 and inspect the diagnostic/remainder results and reload checks. The workspace-only `ultrabert_pilot/modal_check_public_study.py` provides the same check in a fresh pinned x86 image using the owner's existing volume; it loads only this example folder, not the training package or other pilot modules. That owner verification is capped at one CPU worker/hour with no retries; it does not train or publish anything.

4. Publish and register a **new public model repository** using your authenticated Hub account:

   ```bash
   uv run --no-project --with huggingface-hub==1.33.0 --script \
     examples/selective_quantization/package_checkpoint.py \
     /tmp/ultrabert-study-release \
     --publish-repo JulesBelveze/ultrabert-tldr-section-classifier
   ```

   Choose the repository name before running. The command checks all local file hashes, refuses an existing repository, uploads only the manifest's files, downloads the immutable resulting revision, verifies those hashes again, and registers the actual commit in `study.json`. It does not rerun inference after upload. An upload failure after repository creation can leave an incomplete repository; inspect that state before retrying rather than enabling overwrite.

   If the release was uploaded separately, manual registration remains available with `--pin-repo OWNER/MODEL --revision FULL_40_CHARACTER_COMMIT`. Manual registration alone does not verify the public download.

5. Run the reader command from a fresh checkout using the pinned public release. This checks the public download path as well as local computation. Only after it succeeds should the release status at the top of this document be changed to verified and the article claim exact-study reproducibility.

## Evidence and boundaries

`study.json` records the source revisions, checksums, diagnostic hashes, label mapping, training recipe, exemption, and original reported results. The original five JSON files in `recorded-results/` retain the original baseline, group ablation, projection trace, targeted intervention, and final verification. Article bodies were removed from the group report copy; hashes/source rows retain their identity in the public data. These files are evidence to compare against, not new evaluations.

The target is merged newsletter sections, not independently annotated topics. Source overlap remains across the split. The final remainder was new to quantization diagnosis but had participated in checkpoint selection. The reader run does not correct those limitations or make a causal claim about activation outliers. Deployment-size reporting counts inference payload files, excluding release documentation/manifest overhead; it is not resident memory. Empirical timing percentiles do not establish a serving SLA.

## Fresh x86 verification result

`public-study-check-05e80228270a` completed using the packaged original checkpoint and this standalone runner. Rebuilt records and both subset orders matched exactly. FP32 diagnostic macro-F1, the single-down-projection failure, the rounded-weight control, and all six FP32/selective validation-remainder F1 scores matched their original reports. Reload checks passed on all 1,156 remainder articles at batch 16 plus 32 probes under both batch-one policies. The weight-file SHA256 is `639d053ec85a0d3862607261eac7af97a648252ed60a0203a755168f7c1ca1bf`.

Two diagnostic INT8 scores differed: all-Linear macro-F1 was 32.92 instead of 31.33, and the exempted recipe was 80.53 instead of 77.57. The precise cause of that difference has not been isolated. Original measurements have not been overwritten. The newer worker's paired actual-length medians were 130.9 ms FP32 and 68.6 ms selective INT8; these are a separate timing observation, not replacements for the article's original 146.3/112.7 ms pair.

The complete fresh report and release file hashes are in `recorded-results/release-verification.json`. The runner now pins the eight package versions observed in that successful run; this records the verified software environment but does not eliminate CPU or numerical-path variation. A public download of the published revision is still unverified because the model pin is not registered yet.
