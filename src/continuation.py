"""Text continuation entry points."""

import bisect
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.experiment import (
    END_TOKEN,
    MODEL_DIRECTORY_NAMES,
    UNKNOWN_TOKEN,
    _build_katz_probability,
    model_directory_name,
)


def _choose_unseen_id(observed_ids: List[int], rank: int, vocabulary_size: int) -> int:
    """Return the rank-th vocabulary ID not present in the sorted observed IDs."""
    low, high = 0, vocabulary_size - 1
    while low < high:
        middle = (low + high) // 2
        missing_through_middle = middle + 1 - bisect.bisect_right(observed_ids, middle)
        if missing_through_middle > rank:
            high = middle
        else:
            low = middle + 1
    return low


def _sample_next_id(
    counts: Dict[int, int], vocabulary_size: int, k: float, rng: random.Random
) -> int:
    observed_ids = sorted(counts)
    total = sum(counts.values()) + k * vocabulary_size
    choice = rng.random() * total
    for token_id in observed_ids:
        choice -= counts[token_id] + k
        if choice < 0:
            return token_id

    unseen_count = vocabulary_size - len(observed_ids)
    if unseen_count <= 0:
        return observed_ids[-1]
    rank = rng.randrange(unseen_count)
    return _choose_unseen_id(observed_ids, rank, vocabulary_size)


def _sample_weighted_next_id(
    observed_weights: Dict[int, float],
    vocabulary_size: int,
    unseen_weight: float,
    rng: random.Random,
) -> int:
    observed_ids = sorted(observed_weights)
    unseen_count = vocabulary_size - len(observed_ids)
    total = sum(observed_weights.values()) + unseen_count * unseen_weight
    if total <= 0:
        return rng.randrange(vocabulary_size)

    choice = rng.random() * total
    for token_id in observed_ids:
        choice -= observed_weights[token_id]
        if choice < 0:
            return token_id

    if unseen_count <= 0 or unseen_weight <= 0:
        return observed_ids[-1]
    rank = min(unseen_count - 1, int(choice / unseen_weight))
    return _choose_unseen_id(observed_ids, rank, vocabulary_size)


def continue_text(
    text_path: Path,
    length: int,
    n: int,
    project_root: Optional[Path] = None,
) -> str:
    """Generate up to ``length`` tokens using a saved add-k n-gram model."""
    if length <= 0:
        raise ValueError("续写长度必须大于 0")
    if n not in MODEL_DIRECTORY_NAMES:
        raise ValueError("n 必须是 1 到 6 之间的整数")

    try:
        import jieba
    except ImportError as error:
        raise ImportError("续写需要 jieba，请先运行 pip install -r requirements.txt") from error

    project_root = project_root or Path(__file__).resolve().parent.parent
    model_dir = project_root / "model" / model_directory_name(n)
    metadata_path = model_dir / "metadata.json"
    vocabulary_path = model_dir / "vocabulary.json"
    counts_path = model_dir / "ngram_counts.jsonl.gz"
    for path in (metadata_path, vocabulary_path, counts_path):
        if not path.is_file():
            raise FileNotFoundError("缺少 n-gram 模型文件：{}".format(path))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("n") != n:
        raise ValueError("模型元数据中的 n 与所选阶数不一致")
    selected_method = metadata.get("selected_method", "add_k")
    k = metadata.get("selected_k")
    if selected_method == "add_k" and (
        not isinstance(k, (int, float)) or k <= 0
    ):
        raise ValueError("模型元数据中的 selected_k 无效")
    if selected_method not in ("add_k", "good_turing", "katz"):
        raise ValueError("模型元数据中的 selected_method 无效")

    vocabulary_data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary = vocabulary_data.get("tokens")
    if not isinstance(vocabulary, list) or not all(
        isinstance(token, str) for token in vocabulary
    ):
        raise ValueError("模型词表格式无效")
    token_to_id = {token: token_id for token_id, token in enumerate(vocabulary)}
    if UNKNOWN_TOKEN not in token_to_id or END_TOKEN not in token_to_id:
        raise ValueError("模型词表缺少保留符号")
    unk_id = token_to_id[UNKNOWN_TOKEN]
    end_id = token_to_id[END_TOKEN]
    start_id = len(vocabulary)

    counts_by_context: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(dict)
    counts_by_order = {order: Counter() for order in range(1, n + 1)}
    ngram_frequency_counts: Counter = Counter()
    with gzip.open(counts_path, "rt", encoding="utf-8") as counts_file:
        for line_number, line in enumerate(counts_file, start=1):
            try:
                record = json.loads(line)
                ids = record["token_ids"]
                count = record["count"]
                if (
                    not isinstance(ids, list)
                    or len(ids) != n
                    or any(type(token_id) is not int for token_id in ids)
                    or any(token_id < 0 or token_id > start_id for token_id in ids)
                    or ids[-1] >= start_id
                    or type(count) is not int
                    or count <= 0
                ):
                    raise ValueError
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "模型计数文件第 {} 行格式无效".format(line_number)
                ) from error
            context = tuple(ids[:-1])
            counts_by_context[context][ids[-1]] = count
            ngram_frequency_counts[count] += 1
            for order in range(1, n + 1):
                counts_by_order[order][tuple(ids[-order:])] += count

    if selected_method == "good_turing":
        singleton_count = ngram_frequency_counts.get(1, 0)
        doubleton_count = ngram_frequency_counts.get(2, 0)
        fallback_discount = (
            singleton_count / (singleton_count + 2 * doubleton_count)
            if singleton_count + 2 * doubleton_count
            else 0.5
        )
        adjusted_counts = {
            count: min(
                float(count),
                max(
                    0.0,
                    (count + 1) * ngram_frequency_counts[count + 1]
                    / ngram_frequency_counts[count]
                    if ngram_frequency_counts.get(count + 1)
                    else count * (1.0 - fallback_discount),
                ),
            )
            for count in ngram_frequency_counts
        }
    elif selected_method == "katz":
        katz_probability = _build_katz_probability(
            n, counts_by_order, len(vocabulary), cache_size=2 * len(vocabulary)
        )

    text = text_path.read_text(encoding="utf-8-sig")
    try:
        tokens = [token for token in jieba.cut(text) if token.strip()]
    except Exception as error:
        raise ValueError("jieba 无法处理输入文本：{}".format(error)) from error
    history = [token_to_id.get(token, unk_id) for token in tokens]
    if n > 1:
        history = ([start_id] * max(0, n - 1 - len(history)) + history)[-(n - 1):]
    else:
        history = []

    generated: List[str] = []
    rng = random.Random()
    for _ in range(length):
        context = tuple(history[-(n - 1):]) if n > 1 else ()
        context_counts = counts_by_context.get(context, {})
        if selected_method == "add_k":
            next_id = _sample_next_id(
                context_counts, len(vocabulary), float(k), rng
            )
        elif selected_method == "good_turing":
            context_total = sum(context_counts.values())
            observed_weights = {
                token_id: adjusted_counts[count] / context_total
                for token_id, count in context_counts.items()
            } if context_total else {}
            adjusted_total = sum(observed_weights.values())
            unseen_count = len(vocabulary) - len(context_counts)
            leftover_mass = (
                max(1e-12, 1.0 - adjusted_total)
                if context_total
                else 1.0
            )
            unseen_weight = leftover_mass / max(1, unseen_count)
            next_id = _sample_weighted_next_id(
                observed_weights, len(vocabulary), unseen_weight, rng
            )
        else:
            probabilities = [
                max(0.0, katz_probability(n, context, token_id))
                for token_id in range(len(vocabulary))
            ]
            if not any(probabilities):
                probabilities = [1.0] * len(vocabulary)
            next_id = rng.choices(range(len(vocabulary)), weights=probabilities, k=1)[0]
        if next_id == end_id:
            break
        generated.append(vocabulary[next_id])
        if n > 1:
            history.append(next_id)
    return "".join(generated)
