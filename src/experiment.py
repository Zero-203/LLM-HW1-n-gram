"""Train and evaluate unigram through 6-gram language models."""

import csv
import gzip
import json
import math
import struct
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
K_CANDIDATES = (1.0, 0.1, 0.01, 0.001, 0.0001)
KATZ_UNIGRAM_BASE_K = 0.01
MIN_WORD_COUNT = 2
UNKNOWN_TOKEN = "<UNK>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
MODEL_DIRECTORY_NAMES = {
    1: "unigram",
    2: "bigram",
    3: "trigram",
    4: "4-gram",
    5: "5-gram",
    6: "6-gram",
}
SPLIT_FILENAMES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def model_directory_name(n: int) -> str:
    if n not in MODEL_DIRECTORY_NAMES:
        raise ValueError("n 必须是 1 到 6 之间的整数")
    return MODEL_DIRECTORY_NAMES[n]


def _iter_token_sequences(path: Path) -> Iterator[List[str]]:
    if not path.is_file():
        raise FileNotFoundError("缺少数据集文件：{}".format(path))

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError("{} 第 {} 行为空".format(path, line_number))
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "{} 第 {} 行 JSON 无效：{}".format(path, line_number, error.msg)
                ) from error
            if not isinstance(record, dict):
                raise ValueError("{} 第 {} 行必须是 JSON 对象".format(path, line_number))
            tokens = record.get("tokens")
            if not isinstance(tokens, list) or any(
                not isinstance(token, str) for token in tokens
            ):
                raise ValueError(
                    "{} 第 {} 行的 tokens 必须是字符串数组".format(path, line_number)
                )
            if any(token in (UNKNOWN_TOKEN, START_TOKEN, END_TOKEN) for token in tokens):
                raise ValueError(
                    "{} 第 {} 行正文含模型保留符号".format(path, line_number)
                )
            yield tokens


def _pack_ids(ids: Tuple[int, ...]) -> bytes:
    if not ids:
        return b""
    return struct.pack(">" + "I" * len(ids), *ids)


def _event_keys(tokens: List[str], n: int, token_to_id: Dict[str, int], unk_id: int,
                end_id: int, start_id: int) -> Iterator[Tuple[bytes, bytes]]:
    history = [start_id] * (n - 1)
    for token in tokens:
        target_id = token_to_id.get(token, unk_id)
        ngram_key = _pack_ids(tuple(history) + (target_id,))
        yield ngram_key, ngram_key[: 4 * (n - 1)]
        if n > 1:
            history = history[1:] + [target_id]

    ngram_key = _pack_ids(tuple(history) + (end_id,))
    yield ngram_key, ngram_key[: 4 * (n - 1)]


def _train_counts(
    train_path: Path,
    n: int,
    token_to_id: Dict[str, int],
    unk_id: int,
    end_id: int,
    start_id: int,
) -> Tuple[Counter, Counter, int, int]:
    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()
    paragraph_count = 0
    event_count = 0

    for tokens in _iter_token_sequences(train_path):
        paragraph_count += 1
        for ngram_key, context_key in _event_keys(
            tokens, n, token_to_id, unk_id, end_id, start_id
        ):
            ngram_counts[ngram_key] += 1
            context_counts[context_key] += 1
            event_count += 1

    if paragraph_count == 0:
        raise ValueError("训练集为空，无法训练模型")
    return ngram_counts, context_counts, paragraph_count, event_count


def _train_counts_by_order(
    train_path: Path,
    maximum_order: int,
    token_to_id: Dict[str, int],
    unk_id: int,
    end_id: int,
    start_id: int,
) -> Dict[int, Counter]:
    counts_by_order: Dict[int, Counter] = {
        order: Counter() for order in range(1, maximum_order + 1)
    }
    for tokens in _iter_token_sequences(train_path):
        for order in range(1, maximum_order + 1):
            for ngram_key, _context_key in _event_keys(
                tokens, order, token_to_id, unk_id, end_id, start_id
            ):
                ids = struct.unpack(">" + "I" * order, ngram_key)
                counts_by_order[order][ids] += 1
    return counts_by_order


def _build_katz_probability(
    n: int,
    counts_by_order: Dict[int, Counter],
    vocabulary_size: int,
    smoothing_k: Optional[float] = None,
    cache_size: Optional[int] = None,
) -> Callable[[int, Tuple[int, ...], int], float]:
    if cache_size is None:
        cache_size = max(1024, 2 * vocabulary_size)
    unigram_count_total = sum(counts_by_order[1].values())
    context_counts_by_order = {1: Counter({(): unigram_count_total})}
    words_by_context = {order: defaultdict(list) for order in range(2, n + 1)}
    for order in range(2, n + 1):
        order_contexts = Counter()
        for key, count in counts_by_order[order].items():
            order_contexts[key[:-1]] += count
            words_by_context[order][key[:-1]].append(key[-1])
        context_counts_by_order[order] = order_contexts
    frequency_of_frequency = {
        order: Counter(counts_by_order[order].values())
        for order in range(1, n + 1)
    }

    @lru_cache(maxsize=cache_size)
    def katz_probability(order: int, context: Tuple[int, ...], word_id: int) -> float:
        if order == 1:
            count = counts_by_order[1].get((word_id,), 0)
            base_k = (
                smoothing_k
                if smoothing_k and smoothing_k > 0
                else KATZ_UNIGRAM_BASE_K
            )
            return (count + base_k) / (
                unigram_count_total + base_k * vocabulary_size
            )

        context = context[-(order - 1):]
        count = counts_by_order[order].get(context + (word_id,), 0)
        total = context_counts_by_order[order].get(context, 0)
        if not total:
            return katz_probability(order - 1, context[1:], word_id)
        if count:
            fof = frequency_of_frequency[order]
            discount_ratio = (
                (count + 1) * fof[count + 1] / (count * fof[count])
                if fof[count] and fof[count + 1]
                else 0.5
            )
            discount = min(1.0, max(0.0, discount_ratio))
            return count * (1.0 - discount) / total

        alpha = katz_backoff_weight(order, context)
        return alpha * katz_probability(order - 1, context[1:], word_id)

    @lru_cache(maxsize=cache_size)
    def katz_backoff_weight(order: int, context: Tuple[int, ...]) -> float:
        total = context_counts_by_order[order].get(context, 0)
        if not total:
            return 1.0
        observed = words_by_context[order].get(context, ())
        seen_probability = 0.0
        lower_seen_probability = 0.0
        for word_id in observed:
            count = counts_by_order[order][context + (word_id,)]
            fof = frequency_of_frequency[order]
            discount_ratio = (
                (count + 1) * fof[count + 1] / (count * fof[count])
                if fof[count] and fof[count + 1]
                else 0.5
            )
            discount = min(1.0, max(0.0, discount_ratio))
            seen_probability += count * (1.0 - discount) / total
            lower_seen_probability += katz_probability(
                order - 1, context[1:], word_id
            )
        denominator = 1.0 - lower_seen_probability
        return (
            max(0.0, 1.0 - seen_probability) / denominator
            if denominator > 0
            else 0.0
        )

    return katz_probability


def _evaluate(
    path: Path,
    n: int,
    token_to_id: Dict[str, int],
    unk_id: int,
    end_id: int,
    start_id: int,
    ngram_counts: Counter,
    context_counts: Counter,
    vocabulary_size: int,
    smoothing_k: Optional[float],
    smoothing_method: str = "add_k",
    lower_order_counts: Optional[Dict[int, Counter]] = None,
) -> Dict[str, Any]:
    total_events = 0
    total_body_tokens = 0
    unknown_tokens = 0
    zero_probability_events = 0
    negative_log_likelihood = 0.0
    ngram_count_total = sum(ngram_counts.values())

    good_turing_adjusted_counts: Dict[int, float] = {}
    good_turing_context_totals: Dict[Tuple[int, ...], float] = {}
    good_turing_context_types: Dict[Tuple[int, ...], int] = Counter()
    if smoothing_method == "good_turing":
        global_frequency_counts = Counter(ngram_counts.values())
        singleton_count = global_frequency_counts.get(1, 0)
        doubleton_count = global_frequency_counts.get(2, 0)
        fallback_discount = (
            singleton_count / (singleton_count + 2 * doubleton_count)
            if singleton_count + 2 * doubleton_count
            else 0.5
        )
        for count in global_frequency_counts:
            adjusted_count = (
                (count + 1) * global_frequency_counts[count + 1]
                / global_frequency_counts[count]
                if global_frequency_counts.get(count + 1)
                else count * (1.0 - fallback_discount)
            )
            good_turing_adjusted_counts[count] = min(float(count), max(0.0, adjusted_count))

        for ngram_key, count in ngram_counts.items():
            context = (
                struct.unpack(">" + "I" * (n - 1), ngram_key[: 4 * (n - 1)])
                if n > 1
                else ()
            )
            good_turing_context_totals[context] = (
                good_turing_context_totals.get(context, 0.0)
                + good_turing_adjusted_counts[count]
            )
            good_turing_context_types[context] += 1

    if smoothing_method == "katz":
        if lower_order_counts is None:
            raise ValueError("Katz smoothing 需要低阶 n-gram 计数")
        katz_probability = _build_katz_probability(
            n, lower_order_counts, vocabulary_size, smoothing_k
        )

    for tokens in _iter_token_sequences(path):
        total_body_tokens += len(tokens)
        unknown_tokens += sum(token not in token_to_id for token in tokens)
        for ngram_key, context_key in _event_keys(
            tokens, n, token_to_id, unk_id, end_id, start_id
        ):
            count = ngram_counts.get(ngram_key, 0)
            context_count = context_counts.get(context_key, 0)
            total_events += 1
            if smoothing_method == "good_turing":
                context_key_ids = (
                    struct.unpack(">" + "I" * (n - 1), context_key)
                    if n > 1
                    else ()
                )
                context_total = context_counts.get(context_key, 0)
                if count == 0:
                    observed_types = good_turing_context_types.get(context_key_ids, 0)
                    unseen_types = vocabulary_size - observed_types
                    adjusted_total = good_turing_context_totals.get(context_key_ids, 0.0)
                    leftover_mass = max(
                        1e-12,
                        1.0 - adjusted_total / context_total,
                    ) if context_total else 1.0
                    probability = leftover_mass / max(1, unseen_types)
                elif context_total:
                    probability = (
                        good_turing_adjusted_counts.get(count, float(count))
                        / context_total
                    )
                else:
                    probability = 1.0 / (vocabulary_size * max(1, ngram_count_total))
            elif smoothing_method == "katz":
                ids = struct.unpack(">" + "I" * n, ngram_key)
                probability = katz_probability(n, ids[:-1], ids[-1])
                if probability <= 0:
                    probability = 1.0 / (vocabulary_size * max(1, ngram_count_total))
            elif smoothing_k is None:
                if count == 0 or context_count == 0:
                    zero_probability_events += 1
                    continue
                probability = count / context_count
            else:
                probability = (count + smoothing_k) / (
                    context_count + smoothing_k * vocabulary_size
                )
            if probability <= 0:
                zero_probability_events += 1
                continue
            negative_log_likelihood -= math.log(probability)

    if total_events == 0:
        raise ValueError("{} 为空，无法计算评估指标".format(path.name))

    has_zero_probability = zero_probability_events > 0
    mean_nll = (
        None
        if has_zero_probability
        else negative_log_likelihood / total_events
    )
    perplexity = None if mean_nll is None else math.exp(mean_nll)

    return {
        "smoothing_method": smoothing_method,
        "smoothing_k": smoothing_k if smoothing_method == "add_k" else None,
        "parameters": (
            {"unigram_base_k": KATZ_UNIGRAM_BASE_K}
            if smoothing_method == "katz"
            else {
                "count_of_counts_scope": "all observed n-grams of this order",
                "unseen_mass": "redistributed within context",
            }
            if smoothing_method == "good_turing"
            else {}
        ),
        "event_count": total_events,
        "body_token_count": total_body_tokens,
        "unknown_token_count": unknown_tokens,
        "unknown_token_rate": (
            unknown_tokens / total_body_tokens if total_body_tokens else 0.0
        ),
        "zero_probability_event_count": zero_probability_events,
        "zero_probability_event_rate": zero_probability_events / total_events,
        "ngram_coverage": (total_events - zero_probability_events) / total_events,
        "negative_log_likelihood": (
            None if has_zero_probability else negative_log_likelihood
        ),
        "cross_entropy": mean_nll,
        "perplexity": perplexity,
    }


def _evaluate_add_k_candidates(
    path: Path,
    n: int,
    token_to_id: Dict[str, int],
    unk_id: int,
    end_id: int,
    start_id: int,
    ngram_counts: Counter,
    context_counts: Counter,
    vocabulary_size: int,
    k_candidates: Tuple[float, ...],
) -> Dict[str, Dict[str, Any]]:
    negative_log_likelihoods = {str(k): 0.0 for k in k_candidates}
    total_events = 0
    total_body_tokens = 0
    unknown_tokens = 0

    for tokens in _iter_token_sequences(path):
        total_body_tokens += len(tokens)
        unknown_tokens += sum(token not in token_to_id for token in tokens)
        for ngram_key, context_key in _event_keys(
            tokens, n, token_to_id, unk_id, end_id, start_id
        ):
            count = ngram_counts.get(ngram_key, 0)
            context_count = context_counts.get(context_key, 0)
            total_events += 1
            for k in k_candidates:
                probability = (count + k) / (context_count + k * vocabulary_size)
                negative_log_likelihoods[str(k)] -= math.log(probability)

    if total_events == 0:
        raise ValueError("{} 为空，无法计算评估指标".format(path.name))

    unknown_rate = unknown_tokens / total_body_tokens if total_body_tokens else 0.0
    return {
        str(k): {
            "smoothing_method": "add_k",
            "smoothing_k": k,
            "event_count": total_events,
            "body_token_count": total_body_tokens,
            "unknown_token_count": unknown_tokens,
            "unknown_token_rate": unknown_rate,
            "zero_probability_event_count": 0,
            "zero_probability_event_rate": 0.0,
            "ngram_coverage": 1.0,
            "negative_log_likelihood": negative_log_likelihoods[str(k)],
            "cross_entropy": negative_log_likelihoods[str(k)] / total_events,
            "perplexity": math.exp(negative_log_likelihoods[str(k)] / total_events),
        }
        for k in k_candidates
    }


def _write_ngram_counts(
    path: Path, n: int, ngram_counts: Counter, context_counts: Counter,
    vocabulary: List[str],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as output:
        for key, count in ngram_counts.items():
            ids = struct.unpack(">" + "I" * n, key)
            output.write(
                json.dumps(
                    {
                        "token_ids": ids,
                        "count": count,
                        "context_count": context_counts[key[: 4 * (n - 1)]],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def run_experiment(
    n: int,
    project_root: Path = PROJECT_ROOT,
    output_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Compare smoothing candidates on validation and evaluate each on test."""
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 6:
        raise ValueError("n 必须是 1 到 6 之间的整数")

    project_root = Path(project_root)
    data_dir = project_root / "data" / "processed"
    model_name = model_directory_name(n)
    model_dir = project_root / "model" / model_name
    metrics_path = project_root / "result" / "metrics" / "{}_metrics.json".format(
        model_name
    )
    train_path = data_dir / SPLIT_FILENAMES["train"]
    validation_path = data_dir / SPLIT_FILENAMES["val"]
    test_path = data_dir / SPLIT_FILENAMES["test"]

    experiment_started = time.perf_counter()
    vocabulary_started = time.perf_counter()
    word_frequencies: Counter = Counter()
    train_body_token_count = 0
    for tokens in _iter_token_sequences(train_path):
        word_frequencies.update(tokens)
        train_body_token_count += len(tokens)
    if train_body_token_count == 0:
        raise ValueError("训练集没有正文 token")

    lexical_tokens = sorted(
        token
        for token, frequency in word_frequencies.items()
        if frequency >= MIN_WORD_COUNT
    )
    vocabulary = lexical_tokens + [UNKNOWN_TOKEN, END_TOKEN]
    if len(vocabulary) >= 2**32:
        raise ValueError("词表过大，无法使用当前 token ID 编码")
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    unk_id = token_to_id[UNKNOWN_TOKEN]
    end_id = token_to_id[END_TOKEN]
    start_id = len(vocabulary)
    vocabulary_scan_seconds = time.perf_counter() - vocabulary_started

    counting_started = time.perf_counter()
    ngram_counts, context_counts, train_paragraph_count, train_event_count = (
        _train_counts(
            train_path, n, token_to_id, unk_id, end_id, start_id
        )
    )
    lower_order_counts = _train_counts_by_order(
        train_path, n, token_to_id, unk_id, end_id, start_id
    )
    ngram_counting_seconds = time.perf_counter() - counting_started

    validation_started = time.perf_counter()
    validation_metrics_by_k = _evaluate_add_k_candidates(
        validation_path,
        n,
        token_to_id,
        unk_id,
        end_id,
        start_id,
        ngram_counts,
        context_counts,
        len(vocabulary),
        K_CANDIDATES,
    )

    validation_smoothing_metrics = {
        method: _evaluate(
            validation_path,
            n,
            token_to_id,
            unk_id,
            end_id,
            start_id,
            ngram_counts,
            context_counts,
            len(vocabulary),
            None,
            smoothing_method=method,
            lower_order_counts=lower_order_counts if method == "katz" else None,
        )
        for method in ("good_turing", "katz")
    }
    validation_by_candidate = {
        "add_k:{}".format(k): validation_metrics_by_k[str(k)]
        for k in K_CANDIDATES
    }
    validation_by_candidate.update(validation_smoothing_metrics)
    selected_candidate = min(
        validation_by_candidate,
        key=lambda candidate: validation_by_candidate[candidate]["cross_entropy"],
    )
    if selected_candidate.startswith("add_k:"):
        selected_method = "add_k"
        selected_k = float(selected_candidate.split(":", 1)[1])
    else:
        selected_method = selected_candidate
        selected_k = None
    selected_validation_metrics = validation_by_candidate[selected_candidate]
    selected_parameters = (
        {"k": selected_k}
        if selected_method == "add_k"
        else selected_validation_metrics.get("parameters", {})
    )
    best_add_k = min(
        K_CANDIDATES,
        key=lambda k: validation_metrics_by_k[str(k)]["cross_entropy"],
    )

    validation_evaluation_seconds = time.perf_counter() - validation_started
    test_started = time.perf_counter()
    test_metrics_by_k = _evaluate_add_k_candidates(
        test_path,
        n,
        token_to_id,
        unk_id,
        end_id,
        start_id,
        ngram_counts,
        context_counts,
        len(vocabulary),
        K_CANDIDATES,
    )
    test_smoothing_metrics = {
        method: _evaluate(
            test_path,
            n,
            token_to_id,
            unk_id,
            end_id,
            start_id,
            ngram_counts,
            context_counts,
            len(vocabulary),
            None,
            smoothing_method=method,
            lower_order_counts=lower_order_counts if method == "katz" else None,
        )
        for method in ("good_turing", "katz")
    }
    test_by_candidate = {
        "add_k:{}".format(k): test_metrics_by_k[str(k)] for k in K_CANDIDATES
    }
    test_by_candidate.update(test_smoothing_metrics)
    selected_test_metrics = test_by_candidate[selected_candidate]
    test_metrics = {
        "add_k_by_k": test_metrics_by_k,
        "add_k": test_metrics_by_k[str(best_add_k)],
        "mle": _evaluate(
            test_path,
            n,
            token_to_id,
            unk_id,
            end_id,
            start_id,
            ngram_counts,
            context_counts,
            len(vocabulary),
            None,
        ),
        **test_smoothing_metrics,
        "by_candidate": test_by_candidate,
        "selected": selected_test_metrics,
    }
    test_evaluation_seconds = time.perf_counter() - test_started

    model_save_started = time.perf_counter()
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path = model_dir / "vocabulary.json"
    counts_path = model_dir / "ngram_counts.jsonl.gz"
    metadata_path = model_dir / "metadata.json"
    vocabulary_path.write_text(
        json.dumps(
            {
                "tokens": vocabulary,
                "unknown_token": UNKNOWN_TOKEN,
                "end_token": END_TOKEN,
                "start_token": START_TOKEN,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_ngram_counts(counts_path, n, ngram_counts, context_counts, vocabulary)
    model_artifact_save_seconds = time.perf_counter() - model_save_started
    model_artifact_size_bytes = vocabulary_path.stat().st_size + counts_path.stat().st_size
    timing = {
        "vocabulary_scan_seconds": vocabulary_scan_seconds,
        "ngram_counting_seconds": ngram_counting_seconds,
        "validation_evaluation_seconds": validation_evaluation_seconds,
        "test_evaluation_seconds": test_evaluation_seconds,
        "model_artifact_save_seconds": model_artifact_save_seconds,
        "measured_total_seconds": time.perf_counter() - experiment_started,
        "measured_total_scope": (
            "词表扫描、n-gram 计数、验证/测试评估及词表/计数文件写入；"
            "不含最终指标与元数据 JSON 写入"
        ),
    }

    metrics = {
        "n": n,
        "model_name": model_name,
        "k_candidates": list(K_CANDIDATES),
        "selected_candidate": selected_candidate,
        "selected_method": selected_method,
        "selected_k": selected_k,
        "selected_parameters": selected_parameters,
        "selection_metric": "validation cross_entropy",
        "validation_by_k": validation_metrics_by_k,
        "validation_by_smoothing": validation_smoothing_metrics,
        "validation_by_candidate": validation_by_candidate,
        "selected_validation_metrics": selected_validation_metrics,
        "selected_test_metrics": selected_test_metrics,
        "test": test_metrics,
        "training": {
            "paragraph_count": train_paragraph_count,
            "body_token_count": train_body_token_count,
            "event_count": train_event_count,
            "vocabulary_size": len(vocabulary),
            "unique_ngram_count": len(ngram_counts),
            "unique_context_count": len(context_counts),
            "model_artifact_size_bytes": model_artifact_size_bytes,
            "singleton_word_count": sum(
                frequency < MIN_WORD_COUNT
                for frequency in word_frequencies.values()
            ),
        },
        "timing": timing,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "n": n,
        "model_name": model_name,
        "training_data": str(train_path.relative_to(project_root)),
        "validation_data": str(validation_path.relative_to(project_root)),
        "test_data": str(test_path.relative_to(project_root)),
        "vocabulary_file": vocabulary_path.name,
        "ngram_counts_file": counts_path.name,
        "metrics_file": str(metrics_path.relative_to(project_root)),
        "minimum_word_frequency": MIN_WORD_COUNT,
        "rare_word_mapping": "training frequency below 2 and evaluation OOV map to <UNK>",
        "boundary_tokens": {
            "context_start": START_TOKEN,
            "predicted_end": END_TOKEN,
            "sequence_unit": "paragraph",
        },
        "prediction_vocabulary_size": len(vocabulary),
        "k_candidates": list(K_CANDIDATES),
        "selected_candidate": selected_candidate,
        "selected_method": selected_method,
        "selected_k": selected_k,
        "selected_parameters": selected_parameters,
        "validation_cross_entropy": selected_validation_metrics["cross_entropy"],
        "test_selected_perplexity": selected_test_metrics["perplexity"],
        "timing": timing,
        "test_add_k_perplexity": test_metrics["add_k"]["perplexity"],
        "test_mle_zero_probability_event_count": test_metrics["mle"][
            "zero_probability_event_count"
        ],
        "ngram_count_encoding": "gzip JSONL; token_ids are unsigned big-endian 32-bit vocabulary indices",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output_fn(
        "n={} 实验完成：选择 {}，验证集交叉熵={:.6f}，测试集困惑度={:.6f}。".format(
            n,
            selected_candidate,
            selected_validation_metrics["cross_entropy"],
            selected_test_metrics["perplexity"],
        )
    )
    output_fn(
        "耗时统计（秒；合计不含最终指标/元数据 JSON 写入）：词表扫描 {:.3f}，"
        "n-gram 计数 {:.3f}，验证评估 {:.3f}，"
        "测试评估 {:.3f}，模型文件写入 {:.3f}，合计 {:.3f}。".format(
            timing["vocabulary_scan_seconds"],
            timing["ngram_counting_seconds"],
            timing["validation_evaluation_seconds"],
            timing["test_evaluation_seconds"],
            timing["model_artifact_save_seconds"],
            timing["measured_total_seconds"],
        )
    )
    output_fn(
        "模型规模：词表 {}，唯一 n-gram {}，上下文 {}，模型文件 {:.2f} MiB。".format(
            len(vocabulary),
            len(ngram_counts),
            len(context_counts),
            model_artifact_size_bytes / (1024 * 1024),
        )
    )
    output_fn("模型目录：{}".format(model_dir))
    output_fn("指标文件：{}".format(metrics_path))
    return {
        "n": n,
        "model_name": model_name,
        "selected_candidate": selected_candidate,
        "selected_method": selected_method,
        "selected_k": selected_k,
        "timing": timing,
        "training": metrics["training"],
        "validation": selected_validation_metrics,
        "selected_test": selected_test_metrics,
        "test": test_metrics,
        "model_dir": model_dir,
        "metrics_path": metrics_path,
        "metadata_path": metadata_path,
        "vocabulary_path": vocabulary_path,
        "counts_path": counts_path,
    }


def export_metrics_csv(project_root: Path = PROJECT_ROOT) -> Path:
    """Export all smoothing candidates and MLE baselines as one long-form CSV."""
    metrics_dir = Path(project_root) / "result" / "metrics"
    metric_fields = (
        "event_count",
        "body_token_count",
        "unknown_token_count",
        "unknown_token_rate",
        "zero_probability_event_count",
        "zero_probability_event_rate",
        "ngram_coverage",
        "negative_log_likelihood",
        "cross_entropy",
        "perplexity",
    )
    timing_fields = (
        "vocabulary_scan_seconds",
        "ngram_counting_seconds",
        "validation_evaluation_seconds",
        "test_evaluation_seconds",
        "model_artifact_save_seconds",
        "measured_total_seconds",
    )
    rows: List[Dict[str, Any]] = []

    for n in MODEL_DIRECTORY_NAMES:
        model_name = model_directory_name(n)
        metrics_path = metrics_dir / "{}_metrics.json".format(model_name)
        if not metrics_path.is_file():
            raise FileNotFoundError("缺少阶数 {} 的指标文件：{}".format(n, metrics_path))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation_by_candidate = metrics.get("validation_by_candidate")
        test_by_candidate = metrics.get("test", {}).get("by_candidate")
        selected_candidate = metrics.get("selected_candidate")
        if not isinstance(validation_by_candidate, dict) or not isinstance(
            test_by_candidate, dict
        ):
            raise ValueError("{} 缺少候选平滑指标".format(metrics_path.name))
        if set(validation_by_candidate) != set(test_by_candidate):
            raise ValueError("{} 验证/测试候选集合不一致".format(metrics_path.name))
        if selected_candidate not in validation_by_candidate:
            raise ValueError("{} 的获选候选不在验证指标中".format(metrics_path.name))

        training = metrics.get("training", {})
        timing = metrics.get("timing", {})
        common = {
            "n": n,
            "model_name": model_name,
            "selected_candidate": selected_candidate,
            "selected_method": metrics.get("selected_method"),
            "selected_k": metrics.get("selected_k"),
            "training_paragraph_count": training.get("paragraph_count"),
            "training_body_token_count": training.get("body_token_count"),
            "training_event_count": training.get("event_count"),
            "vocabulary_size": training.get("vocabulary_size"),
            "unique_ngram_count": training.get("unique_ngram_count"),
            "unique_context_count": training.get("unique_context_count"),
            "model_artifact_size_bytes": training.get("model_artifact_size_bytes"),
        }
        common.update(
            {"timing_{}".format(key): timing.get(key) for key in timing_fields}
        )

        for candidate, validation_metrics in validation_by_candidate.items():
            test_metrics = test_by_candidate[candidate]
            row = {
                **common,
                "candidate": candidate,
                "smoothing_method": validation_metrics.get("smoothing_method"),
                "k": validation_metrics.get("smoothing_k"),
                "selected": candidate == selected_candidate,
            }
            row.update(
                {
                    "validation_{}".format(field): validation_metrics.get(field)
                    for field in metric_fields
                }
            )
            row.update(
                {
                    "test_{}".format(field): test_metrics.get(field)
                    for field in metric_fields
                }
            )
            rows.append(row)

        mle_metrics = metrics.get("test", {}).get("mle")
        if isinstance(mle_metrics, dict):
            row = {
                **common,
                "candidate": "mle_baseline",
                "smoothing_method": "mle",
                "k": None,
                "selected": False,
            }
            row.update({"validation_{}".format(field): None for field in metric_fields})
            row.update(
                {
                    "test_{}".format(field): mle_metrics.get(field)
                    for field in metric_fields
                }
            )
            rows.append(row)

    columns = (
        "n",
        "model_name",
        "candidate",
        "smoothing_method",
        "k",
        "selected",
        "selected_candidate",
        "selected_method",
        "selected_k",
        *("validation_{}".format(field) for field in metric_fields),
        *("test_{}".format(field) for field in metric_fields),
        "training_paragraph_count",
        "training_body_token_count",
        "training_event_count",
        "vocabulary_size",
        "unique_ngram_count",
        "unique_context_count",
        "model_artifact_size_bytes",
        *("timing_{}".format(field) for field in timing_fields),
    )
    output_path = metrics_dir / "model_comparison.csv"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path
