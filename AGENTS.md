# Repository Guidelines

## Project Structure & Module Organization
- `bert_squeeze/`: core library.
  - `assistants/`: training orchestration. Entrypoints: `train_assistant.py`, `distil_assistant.py`; configs in `assistants/configs/` (`train_*.yaml`, `distil_*.yaml`).
  - `models/`: LightningModules (`lt_*.py`, `base_lt_module.py`), simple baselines (`lstm.py`, `lr.py`), and transformers under `custom_transformers/` (`bert.py`, `distilbert.py`, `theseus_bert.py`, `encoder_decoder.py`, `fastbert.py`, `deebert.py`). Layers in `layers/` (`mha.py`, `classifier.py`).
  - `distillation/`: distillers (`base_distiller.py`, `sequence_classification_distiller.py`, `seq2seq_distiller.py`) and helpers in `utils/` (`labeler.py`).
  - `data/`: LightningDataModules under `data/modules/` (`transformer_module.py`, `distillation_module.py`, `lstm_module.py`, `lr_module.py`, `base.py`).
  - `inference/`: lightweight inference API (`model.py`, `processors.py`).
  - `utils/`: shared utilities — `callbacks/` (e.g., `pruning.py`, `lottery_ticket.py`, `quantization.py`, `fastbert_logic.py`, `checkpointing.py`), `losses/` (`distillation_losses.py`, `losses.py`, `lsl.py`, `romebert_loss.py`), `optimizers/` (`bert_adam.py`), `schedulers/` (`theseus_schedulers.py`), `scorers/` (`sequence_classification_scorer.py`, `lm_scorer.py`), `artifacts/` (`transformer_artifacts.py`, `distillation_artifacts.py`), `errors/`, plus `types.py`, `vocabulary.py`, `utils_fct.py`.
- `tests/`: pytest suite and fixtures (`tests/test_*.py`, `tests/fixtures/`).
- `docs/`: Sphinx sources and notebooks; build output goes to `docs/_build/`.
- `images/`, `lightning_logs/`, `tmp_model/`: assets and local artifacts.

## Build, Test, and Development Commands
- Create env and install: `uv venv && source .venv/bin/activate && uv sync`
- Run tests: `uv run task test` (alias for `uv run pytest -sv tests/*`).
- Lint: `uv run task lint` (flake8 with max line length 90).
- Format: `uv run task format` (isort + black); check only: `uv run task check-formatting`.
- Build docs: `make -C docs html` or `uv run sphinx-build -b html docs docs/_build/html`.

## Coding Style & Naming Conventions
- Python 3.9–3.12 supported (`requires-python >=3.9,<3.13`).
- Black line length 90; isort profile "black"; 4‑space indentation.
- Follow flake8 (ignores: E203, W503, E501, F401, E266); run hooks via `pre-commit`.
- Naming: modules/files `snake_case.py`; classes `CapWords`; functions/vars `snake_case`.
- Prefer type hints and docstrings; keep public APIs stable under `bert_squeeze/`.

## Testing Guidelines
- Use `pytest`; place new tests under `tests/` as `test_*.py`.
- Keep tests deterministic (avoid network); use `tests/fixtures/` for small assets.
- Run `uv run task test` locally; add tests with bug fixes and new features.

## Docs & Examples
- Prefer adding user guides in `docs/` and small runnable snippets in docstrings.
- Build locally with the commands above; avoid committing generated `_build/`.
