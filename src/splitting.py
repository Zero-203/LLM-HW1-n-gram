"""Split cleaned paragraph JSONL records into article-disjoint datasets."""

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SPLIT_NAMES = ("train", "val", "test")
MONTHLY_FILE_PATTERN = re.compile(r"^(\d{6})\.cleaned\.jsonl$")


def _read_articles(
    input_files: Sequence[Path],
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], int, List[Dict[str, str]]]:
    articles: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for input_file in input_files:
        match = MONTHLY_FILE_PATTERN.fullmatch(input_file.name)
        if match is None:
            raise ValueError(
                "清洗文件名应为 YYYYMM.cleaned.jsonl：{}".format(input_file.name)
            )
        file_month = match.group(1)

        with input_file.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(
                        "{} 第 {} 行为空，不是有效 JSONL 记录".format(
                            input_file.name, line_number
                        )
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "{} 第 {} 行 JSON 格式无效：{}".format(
                            input_file.name, line_number, error.msg
                        )
                    ) from error

                source_file = record.get("source_file")
                article_id = record.get("article_id")
                paragraph_id = record.get("paragraph_id")
                tokens = record.get("tokens")
                if source_file != file_month + ".txt":
                    raise ValueError(
                        "{} 第 {} 行 source_file 与月份文件不匹配".format(
                            input_file.name, line_number
                        )
                    )
                if not isinstance(article_id, str) or not article_id:
                    raise ValueError(
                        "{} 第 {} 行缺少有效 article_id".format(
                            input_file.name, line_number
                        )
                    )
                if not isinstance(paragraph_id, str) or not paragraph_id.startswith(
                    article_id + "-"
                ):
                    raise ValueError(
                        "{} 第 {} 行 paragraph_id 与 article_id 不匹配".format(
                            input_file.name, line_number
                        )
                    )
                if not isinstance(tokens, list) or any(
                    not isinstance(token, str) for token in tokens
                ):
                    raise ValueError(
                        "{} 第 {} 行 tokens 必须是字符串数组".format(
                            input_file.name, line_number
                        )
                    )

                key = (input_file.name, article_id)
                article = articles.setdefault(
                    key,
                    {
                        "source_file": source_file,
                        "month": file_month,
                        "content_hasher": hashlib.sha256(),
                        "paragraph_count": 0,
                        "paragraph_ids": set(),
                        "duplicate_paragraph_ids": set(),
                        "duplicate_paragraph_id_count": 0,
                    },
                )
                if article["source_file"] != source_file:
                    raise ValueError("同一 article_id 对应多个 source_file：{}".format(key))
                if paragraph_id in article["paragraph_ids"]:
                    article["duplicate_paragraph_ids"].add(paragraph_id)
                    article["duplicate_paragraph_id_count"] += 1
                else:
                    article["paragraph_ids"].add(paragraph_id)

                article["paragraph_count"] += 1
                for token in tokens:
                    token_bytes = json.dumps(token, ensure_ascii=False).encode("utf-8")
                    article["content_hasher"].update(len(token_bytes).to_bytes(8, "big"))
                    article["content_hasher"].update(token_bytes)

    if not articles:
        raise FileNotFoundError(
            "{} 中没有 YYYYMM.cleaned.jsonl 清洗文件".format(
                input_files[0].parent if input_files else "输入目录"
            )
        )

    duplicate_paragraph_id_count = sum(
        article["duplicate_paragraph_id_count"] for article in articles.values()
    )
    duplicate_paragraph_id_examples: List[Dict[str, str]] = []
    for key, article in articles.items():
        for paragraph_id in sorted(article["duplicate_paragraph_ids"]):
            if len(duplicate_paragraph_id_examples) >= 10:
                break
            duplicate_paragraph_id_examples.append(
                {
                    "source_file": article["source_file"],
                    "article_id": key[1],
                    "paragraph_id": paragraph_id,
                }
            )
        article["content_digest"] = article["content_hasher"].hexdigest()
        del article["content_hasher"]
        del article["paragraph_ids"]
        del article["duplicate_paragraph_ids"]
        del article["duplicate_paragraph_id_count"]

    return articles, duplicate_paragraph_id_count, duplicate_paragraph_id_examples


def _assign_splits(
    articles: Dict[Tuple[str, str], Dict[str, Any]],
    proportions: Sequence[float],
    seed: int,
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, List[List[Tuple[str, str]]]]]:
    totals_by_month: Dict[str, int] = defaultdict(int)
    articles_by_digest: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for key, article in articles.items():
        totals_by_month[article["month"]] += 1
        articles_by_digest[article["content_digest"]].append(key)

    proportion_sum = sum(proportions)
    normalized = tuple(value / proportion_sum for value in proportions)
    targets = {
        split: {
            month: totals_by_month[month] * normalized[index]
            for month in totals_by_month
        }
        for index, split in enumerate(SPLIT_NAMES)
    }
    assigned_counts = {
        split: {month: 0 for month in totals_by_month} for split in SPLIT_NAMES
    }

    components = list(articles_by_digest.values())
    randomizer = random.Random(seed)
    randomizer.shuffle(components)
    components.sort(key=len, reverse=True)

    assignments: Dict[Tuple[str, str], str] = {}
    duplicate_groups: List[List[Tuple[str, str]]] = []
    for component in components:
        if len(component) > 1:
            duplicate_groups.append(sorted(component))

        component_counts: Dict[str, int] = defaultdict(int)
        for key in component:
            component_counts[articles[key]["month"]] += 1

        scores = {}
        for split in SPLIT_NAMES:
            score = 0.0
            for month, count in component_counts.items():
                target = targets[split][month]
                current = assigned_counts[split][month]
                scale = max(target, 1.0)
                score += (
                    (current + count - target) ** 2 - (current - target) ** 2
                ) / scale
            scores[split] = score

        best_score = min(scores.values())
        best_splits = [
            split
            for split in SPLIT_NAMES
            if abs(scores[split] - best_score) < 1e-12
        ]
        selected_split = randomizer.choice(best_splits)
        for key in component:
            assignments[key] = selected_split
        for month, count in component_counts.items():
            assigned_counts[selected_split][month] += count

    return assignments, {"duplicate_groups": duplicate_groups, "counts": assigned_counts}


def split_dataset(
    input_dir: Path, output_dir: Path, proportions: Sequence[float], seed: int
) -> Dict[str, Any]:
    """Split paragraphs by article, stratified by source month, with a seeded ratio."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if len(proportions) != 3 or any(
        not isinstance(value, (int, float)) or value <= 0 for value in proportions
    ):
        raise ValueError("必须提供三个大于 0 的 train:val:test 比例")
    if any(not math.isfinite(value) for value in proportions):
        raise ValueError("数据集比例必须是有限数值")
    if not math.isfinite(sum(proportions)):
        raise ValueError("数据集比例之和必须是有限数值")
    if not isinstance(seed, int):
        raise ValueError("随机种子必须是整数")

    input_files = sorted(input_dir.glob("*.cleaned.jsonl"))
    if not input_files:
        raise FileNotFoundError(
            "{} 中没有 YYYYMM.cleaned.jsonl 清洗文件".format(input_dir)
        )

    articles, duplicate_paragraph_id_count, duplicate_paragraph_id_examples = (
        _read_articles(input_files)
    )
    assignments, assignment_details = _assign_splits(articles, proportions, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        split: output_dir / "{}.jsonl".format(split) for split in SPLIT_NAMES
    }
    temporary_paths = {
        split: output_dir / ".{}.jsonl.tmp".format(split) for split in SPLIT_NAMES
    }
    manifest_path = output_dir / "split_manifest.json"
    temporary_manifest_path = output_dir / ".split_manifest.json.tmp"

    try:
        with ExitStack() as stack:
            outputs = {
                split: stack.enter_context(
                    path.open("w", encoding="utf-8", newline="\n")
                )
                for split, path in temporary_paths.items()
            }
            for input_file in input_files:
                with input_file.open("r", encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        key = (input_file.name, record["article_id"])
                        split = assignments[key]
                        outputs[split].write(line if line.endswith("\n") else line + "\n")

        monthly_counts = {
            month: {
                split: assignment_details["counts"][split][month]
                for split in SPLIT_NAMES
            }
            for month in sorted(assignment_details["counts"][SPLIT_NAMES[0]])
        }
        split_counts = {
            split: sum(assignment_details["counts"][split].values())
            for split in SPLIT_NAMES
        }
        manifest = {
            "random_seed": seed,
            "requested_proportions": list(proportions),
            "normalized_proportions": [
                value / sum(proportions) for value in proportions
            ],
            "source_files": [path.name for path in input_files],
            "article_count": len(articles),
            "split_article_counts": split_counts,
            "monthly_article_counts": monthly_counts,
            "exact_duplicate_group_count": len(assignment_details["duplicate_groups"]),
            "exact_duplicate_article_count": sum(
                len(group) for group in assignment_details["duplicate_groups"]
            ),
            "duplicate_paragraph_id_count": duplicate_paragraph_id_count,
            "duplicate_paragraph_id_examples": duplicate_paragraph_id_examples,
            "duplicate_groups": [
                [
                    {"source_file": articles[key]["source_file"], "article_id": key[1]}
                    for key in group
                ]
                for group in assignment_details["duplicate_groups"]
            ],
            "assignments": [
                {
                    "source_file": articles[key]["source_file"],
                    "article_id": key[1],
                    "split": assignments[key],
                }
                for key in sorted(assignments)
            ],
        }
        temporary_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for split in SPLIT_NAMES:
            temporary_paths[split].replace(output_paths[split])
        temporary_manifest_path.replace(manifest_path)
    except Exception:
        for path in list(temporary_paths.values()) + [temporary_manifest_path]:
            path.unlink(missing_ok=True)
        raise

    return {
        "article_count": len(articles),
        "split_article_counts": split_counts,
        "monthly_article_counts": monthly_counts,
        "duplicate_paragraph_id_count": duplicate_paragraph_id_count,
        "manifest_path": manifest_path,
        "output_paths": output_paths,
    }
