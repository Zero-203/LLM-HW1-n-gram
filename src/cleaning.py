"""Clean the tagged PFR corpus into paragraph-level JSONL."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


RECORD_PATTERN = re.compile(
    r"^(?P<date>\d{8})-(?P<page>\d+)-(?P<article>\d+)-"
    r"(?P<paragraph>\d+)/m(?:\s+(?P<body>.*))?$"
)


def decode_source(path: Path) -> Tuple[str, str]:
    """Decode a source file without silently replacing undecodable bytes."""
    content = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError("Unable to decode as UTF-8 or GB18030")


def tokenize_tagged_body(body: str) -> Tuple[List[str], int]:
    """Remove POS and bracket-group labels while retaining all word tokens."""
    tokens: List[str] = []
    untagged_count = 0

    for raw_token in body.split():
        token = raw_token.lstrip("[")
        if "]" in token:
            token = token.split("]", 1)[0]
        if not token:
            continue

        if "/" in token:
            word, _pos = token.rsplit("/", 1)
            if word:
                tokens.append(word)
            else:
                untagged_count += 1
        else:
            untagged_count += 1
            tokens.append(token)

    return tokens, untagged_count


def clean_file(source_path: Path, output_dir: Path) -> Dict[str, Any]:
    text, encoding = decode_source(source_path)
    output_path = output_dir / (source_path.stem + ".cleaned.jsonl")
    rejected_path = output_dir / (source_path.stem + ".unparsed.txt")

    article_ids = set()
    blank_lines = 0
    paragraph_count = 0
    empty_paragraph_count = 0
    token_count = 0
    token_annotation_issue_count = 0
    token_annotation_issue_examples: List[str] = []
    malformed_examples: List[str] = []
    malformed_count = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        with rejected_path.open("w", encoding="utf-8", newline="\n") as rejected:
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    blank_lines += 1
                    continue

                match = RECORD_PATTERN.match(line)
                if match is None:
                    malformed_count += 1
                    rejected.write(raw_line + "\n")
                    if len(malformed_examples) < 5:
                        malformed_examples.append(
                            "line {}: {}".format(line_number, line[:200])
                        )
                    continue

                date = match.group("date")
                page = match.group("page")
                article_number = match.group("article")
                paragraph_number = match.group("paragraph")
                body = match.group("body") or ""
                tokens, annotation_issue_count = tokenize_tagged_body(body)
                if annotation_issue_count and len(token_annotation_issue_examples) < 5:
                    token_annotation_issue_examples.append(
                        "line {}: {}".format(line_number, body[:200])
                    )

                article_id = "{}-{}-{}".format(date, page, article_number)
                paragraph_id = "{}-{}-{}-{}".format(
                    date, page, article_number, paragraph_number
                )
                article_ids.add(article_id)
                paragraph_count += 1
                token_count += len(tokens)
                token_annotation_issue_count += annotation_issue_count
                if not tokens:
                    empty_paragraph_count += 1

                record = {
                    "source_file": source_path.name,
                    "article_id": article_id,
                    "paragraph_id": paragraph_id,
                    "tokens": tokens,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    if malformed_count == 0:
        rejected_path.unlink(missing_ok=True)

    return {
        "source_file": source_path.name,
        "source_encoding": encoding,
        "output_file": output_path.name,
        "line_count": len(text.splitlines()),
        "blank_line_count": blank_lines,
        "article_count": len(article_ids),
        "paragraph_count": paragraph_count,
        "empty_paragraph_count": empty_paragraph_count,
        "token_count": token_count,
        "token_annotation_issue_count": token_annotation_issue_count,
        "token_annotation_issue_examples": token_annotation_issue_examples,
        "malformed_line_count": malformed_count,
        "malformed_examples": malformed_examples,
        "rejected_file": rejected_path.name if malformed_count else None,
    }


def clean_corpus(input_dir: Path, output_dir: Path, pattern: str) -> Dict[str, Any]:
    source_files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not source_files:
        raise FileNotFoundError(
            "No source files matching {!r} in {}".format(pattern, input_dir)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_summaries = [clean_file(path, output_dir) for path in source_files]
    totals = {
        "source_file_count": len(file_summaries),
        "article_count": sum(item["article_count"] for item in file_summaries),
        "paragraph_count": sum(item["paragraph_count"] for item in file_summaries),
        "empty_paragraph_count": sum(
            item["empty_paragraph_count"] for item in file_summaries
        ),
        "token_count": sum(item["token_count"] for item in file_summaries),
        "token_annotation_issue_count": sum(
            item["token_annotation_issue_count"] for item in file_summaries
        ),
        "malformed_line_count": sum(
            item["malformed_line_count"] for item in file_summaries
        ),
    }
    summary = {
        "cleaning_rules": {
            "record_format": "date-page-article-paragraph/m",
            "output_format": "one UTF-8 JSON object per paragraph (JSONL)",
            "preserved": ["body words", "punctuation", "numbers", "dates", "bylines"],
            "removed": ["paragraph identifiers from token sequences", "POS tags", "group brackets and outer labels"],
            "paragraph_boundaries": "one JSONL record per paragraph; tokens are not joined across records",
            "unknown_word_mapping": "not applied during cleaning",
            "train_validation_test_split": "not applied during cleaning",
        },
        "totals": totals,
        "files": file_summaries,
    }
    summary_path = output_dir / "cleaning_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
