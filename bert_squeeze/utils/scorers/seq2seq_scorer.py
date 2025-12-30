from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from evaluate import EvaluationModule, load


def _sanitize_group_label(label: Any) -> str:
    label_str = str(label).strip().lower()
    return label_str.replace(" ", "_")


def _length(tokens: str, unit: str) -> int:
    if unit == "words":
        return len(tokens.split())
    if unit == "chars":
        return len(tokens)
    raise ValueError(f"Unsupported length unit '{unit}'. Use 'words' or 'chars'.")


@dataclass
class MetricConfig:
    name: str
    loader_name: str
    compute_kwargs: Dict[str, Any] = field(default_factory=dict)


class Seq2SeqScorer:
    """Compute multiple seq2seq metrics with optional grouping support."""

    DEFAULT_METRICS = ("rouge", "bleu")
    METRIC_LOADERS: Mapping[str, MetricConfig] = {
        "rouge": MetricConfig(
            name="rouge",
            loader_name="rouge",
            compute_kwargs={"use_aggregator": True, "use_stemmer": True},
        ),
        "bleu": MetricConfig(name="bleu", loader_name="sacrebleu"),
        "bertscore": MetricConfig(name="bertscore", loader_name="bertscore"),
        "meteor": MetricConfig(name="meteor", loader_name="meteor"),
    }
    DEFAULT_BERTSCORE_KWARGS = {
        "lang": "en",
        "model_type": "microsoft/deberta-base-mnli",
        "rescale_with_baseline": True,
    }

    def __init__(
        self,
        metrics: Optional[Iterable[str]] = None,
        *,
        rouge_types: Iterable[str] = ("rouge1", "rouge2", "rougeL"),
        bertscore_kwargs: Optional[Dict[str, Any]] = None,
        compute_length_stats: bool = True,
        length_unit: str = "words",
    ) -> None:
        self.metrics = tuple(metrics or self.DEFAULT_METRICS)
        self._validate_metrics()
        self.rouge_types = tuple(rouge_types)
        self.bertscore_kwargs = bertscore_kwargs or self.DEFAULT_BERTSCORE_KWARGS
        self.compute_length_stats = compute_length_stats
        self.length_unit = length_unit
        self._metric_cache: MutableMapping[str, EvaluationModule] = {}

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        *,
        group_labels: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        if group_labels is not None and len(group_labels) != len(predictions):
            raise ValueError("Group labels must match the number of predictions.")

        scores: Dict[str, float] = {}
        scores.update(self._compute_metrics(predictions, references))

        if self.compute_length_stats:
            scores.update(self._compute_length_stats(predictions, references))

        if group_labels is not None:
            group_scores = self._compute_group_metrics(
                predictions, references, group_labels
            )
            scores.update(group_scores)

        return scores

    def _compute_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for metric_name in self.metrics:
            if metric_name == "rouge":
                scores.update(self._compute_rouge(predictions, references))
            elif metric_name == "bleu":
                scores.update(self._compute_bleu(predictions, references))
            elif metric_name == "bertscore":
                scores.update(self._compute_bertscore(predictions, references))
            elif metric_name == "meteor":
                scores.update(self._compute_meteor(predictions, references))
        return scores

    def _compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        metric = self._get_metric("rouge")
        result = metric.compute(
            predictions=predictions,
            references=references,
            rouge_types=list(self.rouge_types),
            **self.METRIC_LOADERS["rouge"].compute_kwargs,
        )
        return {metric_name: float(result[metric_name]) for metric_name in result}

    def _compute_bleu(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        metric = self._get_metric("bleu")
        formatted_refs = [[ref] for ref in references]
        result = metric.compute(predictions=predictions, references=formatted_refs)
        return {"bleu": float(result["score"])}

    def _compute_bertscore(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        metric = self._get_metric("bertscore")
        result = metric.compute(
            predictions=predictions,
            references=references,
            **self.bertscore_kwargs,
        )
        return {
            "bertscore_precision": float(result["precision"]),
            "bertscore_recall": float(result["recall"]),
            "bertscore_f1": float(result["f1"]),
        }

    def _compute_meteor(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        metric = self._get_metric("meteor")
        result = metric.compute(predictions=predictions, references=references)
        return {"meteor": float(result["meteor"])}

    def _compute_group_metrics(
        self,
        predictions: List[str],
        references: List[str],
        group_labels: List[Any],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        grouped_indices: Dict[str, List[int]] = {}
        for idx, label in enumerate(group_labels):
            group = _sanitize_group_label(label)
            grouped_indices.setdefault(group, []).append(idx)

        for group_name, indices in grouped_indices.items():
            group_preds = [predictions[i] for i in indices]
            group_refs = [references[i] for i in indices]
            group_scores = self._compute_metrics(group_preds, group_refs)
            prefix = f"{group_name}_"
            for metric_name, value in group_scores.items():
                scores[f"{prefix}{metric_name}"] = value

        return scores

    def _compute_length_stats(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        pred_lengths = [_length(pred, self.length_unit) for pred in predictions]
        ref_lengths = [_length(ref, self.length_unit) for ref in references]
        avg_pred = sum(pred_lengths) / max(len(pred_lengths), 1)
        avg_ref = sum(ref_lengths) / max(len(ref_lengths), 1)
        length_ratio = avg_pred / (avg_ref + 1e-8)
        mae = sum(abs(p - r) for p, r in zip(pred_lengths, ref_lengths)) / max(
            len(pred_lengths), 1
        )
        return {
            f"length_avg_prediction_{self.length_unit}": float(avg_pred),
            f"length_avg_reference_{self.length_unit}": float(avg_ref),
            f"length_ratio_{self.length_unit}": float(length_ratio),
            f"length_mae_{self.length_unit}": float(mae),
        }

    def _get_metric(self, name: str) -> EvaluationModule:
        if name not in self._metric_cache:
            config = self.METRIC_LOADERS[name]
            self._metric_cache[name] = load(config.loader_name)
        return self._metric_cache[name]

    def _validate_metrics(self) -> None:
        unknown = set(self.metrics) - set(self.METRIC_LOADERS)
        if unknown:
            raise ValueError(
                f"Unsupported metrics {sorted(unknown)}. "
                f"Available metrics: {sorted(self.METRIC_LOADERS)}"
            )


class LightningSeq2SeqScorer:
    """Thin Lightning-friendly wrapper for Seq2SeqScorer."""

    def __init__(self, prefix: str, **scorer_kwargs: Any) -> None:
        self.prefix = prefix.rstrip("_")
        self.scorer = Seq2SeqScorer(**scorer_kwargs)

    def get_log_dict(
        self,
        predictions: List[str],
        references: List[str],
        *,
        group_labels: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        scores = self.scorer.compute(
            predictions=predictions,
            references=references,
            group_labels=group_labels,
        )
        prefixed: Dict[str, float] = {}
        for key, value in scores.items():
            prefixed_key = f"{self.prefix}_{key}" if self.prefix else key
            prefixed[prefixed_key] = value
        return prefixed
