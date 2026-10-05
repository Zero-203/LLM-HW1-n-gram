"""Interactive command-line entry point for the n-gram project."""

import json
import math
import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from src.cleaning import clean_corpus
from src.continuation import continue_text
from src.experiment import model_directory_name, run_experiment
from src.splitting import split_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_proportions(value: str) -> Tuple[float, float, float]:
    """Parse three positive train/validation/test proportions."""
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("比例格式应为 train:val:test，例如 8:1:1")

    try:
        proportions = tuple(float(part.strip()) for part in parts)
    except ValueError as error:
        raise ValueError("比例必须是数字") from error

    if any(not math.isfinite(part) or part <= 0 for part in proportions):
        raise ValueError("三个比例都必须是有限的正数")

    return proportions


def _read_input(
    prompt: str, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> Optional[str]:
    try:
        return input_fn(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        output_fn("\n已取消当前操作。")
        return None


def _report_previous_result(
    result_path: Path, output_fn: Callable[[str], None]
) -> None:
    if not result_path.is_file():
        output_fn("未找到上次操作的统计清单，可能是上次操作未正常完成。")
        return

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        output_fn("无法读取上次操作结果：{}".format(error))
        return
    if not isinstance(result, dict):
        output_fn("上次操作结果格式无效，无法生成摘要。")
        return

    if isinstance(result.get("totals"), dict):
        totals = result["totals"]
        fields = (
            ("source_file_count", "语料文件"),
            ("article_count", "文章"),
            ("paragraph_count", "段落"),
            ("token_count", "token"),
            ("malformed_line_count", "格式异常行"),
            ("token_annotation_issue_count", "标注异常"),
        )
        summary = "，".join(
            "{}={}".format(label, totals[key])
            for key, label in fields
            if key in totals
        )
    elif isinstance(result.get("split_article_counts"), dict):
        fields = (
            ("random_seed", "随机种子"),
            ("normalized_proportions", "归一化比例"),
            ("article_count", "文章总数"),
            ("split_article_counts", "train/val/test"),
            ("exact_duplicate_group_count", "重复文章组"),
            ("duplicate_paragraph_id_count", "重复段落编号"),
        )
        summary = "，".join(
            "{}={}".format(label, result[key])
            for key, label in fields
            if key in result
        )
    elif "n" in result and "model_name" in result:
        validation = result.get("selected_validation_metrics", {})
        if not isinstance(validation, dict):
            validation = {}
        if not validation and isinstance(result.get("validation"), dict):
            validation = result["validation"]
        test = result.get("test", {})
        if not isinstance(test, dict):
            test = {}
        selected_test = result.get("selected_test_metrics", test.get("selected", {}))
        if not isinstance(selected_test, dict):
            selected_test = {}
        add_k_metrics = test.get("add_k", {})
        if not isinstance(add_k_metrics, dict):
            add_k_metrics = {}
        training = result.get("training", {})
        if not isinstance(training, dict):
            training = {}
        timing = result.get("timing", {})
        if not isinstance(timing, dict):
            timing = {}
        validation_cross_entropy = result.get(
            "validation_cross_entropy", validation.get("cross_entropy")
        )
        test_perplexity = result.get(
            "test_selected_perplexity",
            selected_test.get("perplexity", add_k_metrics.get("perplexity")),
        )
        elapsed = timing.get("measured_total_seconds")
        fields = ["n={}".format(result["n"])]
        selected_method = result.get("selected_method", "add_k")
        fields.append("平滑={}".format(selected_method))
        if selected_method == "add_k" and result.get("selected_k") is not None:
            fields.append("k={}".format(result["selected_k"]))
        if "event_count" in training:
            fields.append("训练事件={}".format(training["event_count"]))
        if validation_cross_entropy is not None:
            fields.append("验证交叉熵={:.6f}".format(validation_cross_entropy))
        if test_perplexity is not None:
            fields.append("测试困惑度={:.6f}".format(test_perplexity))
        fields.append(
            "总耗时={:.3f}s".format(elapsed) if elapsed is not None else "耗时未记录"
        )
        summary = "，".join(fields)
    else:
        summary = "上次结果文件存在，但没有可展示的摘要字段。"

    output_fn(
        "上次操作结果：{}".format(
            summary if summary else "上次结果文件存在，但统计清单为空。"
        )
    )


def prepare_output_directory(
    output_dir: Path,
    operation: str,
    artifact_patterns: Sequence[str],
    result_files: Sequence[str],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> bool:
    """Ask before deleting recognized outputs from a previous run."""
    return prepare_output_directories(
        operation,
        ((output_dir, artifact_patterns, result_files),),
        input_fn,
        output_fn,
    )


def prepare_output_directories(
    operation: str,
    output_specs: Sequence[Tuple[Path, Sequence[str], Sequence[str]]],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> bool:
    """Check and optionally replace outputs across one or more destinations."""
    artifacts = []
    previous_results = []
    for output_dir, artifact_patterns, result_files in output_specs:
        output_dir = Path(output_dir)
        candidates = {
            path
            for pattern in artifact_patterns
            for path in output_dir.glob(pattern)
            if path.is_file() or path.is_dir()
        }
        selected = []
        for path in sorted(candidates, key=lambda item: (len(item.parts), str(item))):
            if not any(parent in selected for parent in path.parents):
                selected.append(path)
        artifacts.extend(selected)
        previous_results.extend(output_dir / filename for filename in result_files)

    if not artifacts:
        return True

    output_fn(
        "检测到上次{}遗留的 {} 个文件/目录：{}".format(
            operation,
            len(artifacts),
            "、".join(
                "{}/{}".format(path.parent.name, path.name) for path in artifacts[:10]
            ),
        )
    )
    if len(artifacts) > 10:
        output_fn("其余 {} 个项目略。".format(len(artifacts) - 10))
    for result_path in previous_results:
        _report_previous_result(result_path, output_fn)

    while True:
        try:
            answer = input_fn(
                "是否删除上述上次生成的数据并重新执行？(y/n)："
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            output_fn("已取消操作，保留现有数据。")
            return False

        if answer in ("y", "yes"):
            break
        if answer in ("n", "no"):
            output_fn("已取消本次{}，保留现有数据。".format(operation))
            return False
        output_fn("请输入 y 或 n。")

    try:
        for path in artifacts:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    except OSError as error:
        output_fn("清理上次数据失败，已停止操作：{}".format(error))
        return False

    output_fn("已删除上次{}的生成文件，开始本次操作。".format(operation))
    return True


def _handle_clean(
    input_dir: Path,
    interim_dir: Path,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> None:
    if not prepare_output_directory(
        interim_dir,
        "清洗",
        ("*.cleaned.jsonl", "*.unparsed.txt", "cleaning_summary.json"),
        ("cleaning_summary.json",),
        input_fn,
        output_fn,
    ):
        return

    try:
        summary = clean_corpus(input_dir, interim_dir, "1998*.txt")
    except (FileNotFoundError, UnicodeError, OSError) as error:
        output_fn("清洗失败：{}".format(error))
        return

    totals = summary["totals"]
    output_fn(
        "清洗完成：{} 个源文件，{} 篇文章，{} 个段落，{} 个 token。".format(
            totals["source_file_count"],
            totals["article_count"],
            totals["paragraph_count"],
            totals["token_count"],
        )
    )
    output_fn("统计文件：{}".format(interim_dir / "cleaning_summary.json"))
    if totals["malformed_line_count"]:
        output_fn(
            "注意：有 {} 行无法解析，已保存在对应的 .unparsed.txt 文件中。".format(
                totals["malformed_line_count"]
            )
        )


def _handle_split(
    input_fn: Callable[[str], str], output_fn: Callable[[str], None], project_root: Path
) -> None:
    raw_proportions = _read_input(
        "输入 train:val:test 比例 [8:1:1]：", input_fn, output_fn
    )
    if raw_proportions is None:
        return
    if not raw_proportions:
        raw_proportions = "8:1:1"

    try:
        proportions = parse_proportions(raw_proportions)
    except ValueError as error:
        output_fn("参数无效：{}".format(error))
        return

    raw_seed = _read_input("输入随机种子 [42]：", input_fn, output_fn)
    if raw_seed is None:
        return
    try:
        seed = int(raw_seed) if raw_seed else 42
    except ValueError:
        output_fn("参数无效：随机种子必须是整数。")
        return

    output_dir = project_root / "data" / "processed"
    if not prepare_output_directory(
        output_dir,
        "数据集划分",
        (
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "split_manifest.json",
            ".*.jsonl.tmp",
            ".split_manifest.json.tmp",
        ),
        ("split_manifest.json",),
        input_fn,
        output_fn,
    ):
        return

    try:
        summary = split_dataset(
            project_root / "data" / "interim",
            output_dir,
            proportions,
            seed,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        output_fn("数据集划分失败：{}".format(error))
        return

    output_fn(
        "数据集划分完成：{} 篇文章；train/val/test 分别为 {}。".format(
            summary["article_count"], summary["split_article_counts"]
        )
    )
    output_fn("划分清单：{}".format(summary["manifest_path"]))


def _handle_experiment(
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    project_root: Path,
) -> None:
    raw_n = _read_input("输入 n 值（1-6）：", input_fn, output_fn)
    if raw_n is None:
        return
    try:
        n = int(raw_n)
    except ValueError:
        output_fn("参数无效：n 必须是整数。")
        return
    if not 1 <= n <= 6:
        output_fn("参数无效：当前计划支持 n=1 到 n=6。")
        return

    model_name = model_directory_name(n)
    model_dir = project_root / "model" / model_name
    metrics_dir = project_root / "result" / "metrics"
    metrics_filename = "{}_metrics.json".format(model_name)
    if not prepare_output_directories(
        "n={} 实验".format(n),
        (
            (
                model_dir,
                ("vocabulary.json", "ngram_counts.jsonl.gz", "metadata.json"),
                ("metadata.json",),
            ),
            (metrics_dir, (metrics_filename,), (metrics_filename,)),
        ),
        input_fn,
        output_fn,
    ):
        return

    try:
        run_experiment(n, project_root=project_root, output_fn=output_fn)
    except (FileNotFoundError, OSError, ValueError) as error:
        output_fn("实验运行失败：{}".format(error))
        return
    except NotImplementedError as error:
        output_fn(str(error))


def _handle_continuation(
    input_fn: Callable[[str], str], output_fn: Callable[[str], None], project_root: Path
) -> None:
    raw_n = _read_input("选择模型阶数 n（1-6）：", input_fn, output_fn)
    if raw_n is None:
        return
    try:
        n = int(raw_n)
    except ValueError:
        output_fn("参数无效：n 必须是 1 到 6 之间的整数。")
        return
    if not 1 <= n <= 6:
        output_fn("参数无效：当前计划支持 n=1 到 n=6。")
        return

    raw_path = _read_input("输入待续写的 txt 文件路径：", input_fn, output_fn)
    if raw_path is None:
        return
    raw_path = raw_path.strip('"').strip("'")
    text_path = Path(raw_path)
    if not text_path.is_absolute():
        text_path = project_root / text_path
    if text_path.suffix.lower() != ".txt" or not text_path.is_file():
        output_fn("参数无效：请指定一个存在的 .txt 文件。")
        return

    raw_length = _read_input("输入续写长度（token 数）：", input_fn, output_fn)
    if raw_length is None:
        return
    try:
        length = int(raw_length)
    except ValueError:
        output_fn("参数无效：续写长度必须是正整数。")
        return
    if length <= 0:
        output_fn("参数无效：续写长度必须大于 0。")
        return

    model_name = model_directory_name(n)
    output_dir = project_root / "result" / "continuation"
    output_path = output_dir / "{}_{}.txt".format(text_path.stem, model_name)
    if not prepare_output_directories(
        "n={} 文本续写".format(n),
        ((output_dir, (output_path.name,), ()),),
        input_fn,
        output_fn,
    ):
        return

    try:
        continuation = continue_text(text_path, length, n, project_root)
        original_text = text_path.read_text(encoding="utf-8-sig")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(original_text + continuation, encoding="utf-8")
    except (FileNotFoundError, ImportError, OSError, ValueError) as error:
        output_fn("文本续写失败：{}".format(error))
        return
    output_fn("续写结果：{}".format(continuation))
    output_fn("结果已保存：{}".format(output_path))


def run_interactive(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    project_root: Path = PROJECT_ROOT,
) -> int:
    """Run the numbered interactive command menu."""
    project_root = Path(project_root)
    while True:
        output_fn(
            "\n=== n-gram 实验菜单 ===\n"
            "1. 清洗语料\n"
            "2. 划分数据集\n"
            "3. 运行 n-gram 实验\n"
            "4. 文本续写\n"
            "0. 退出"
        )
        choice = _read_input("请选择命令：", input_fn, output_fn)
        if choice is None or choice == "0":
            output_fn("退出程序。")
            return 0
        if choice == "1":
            _handle_clean(
                project_root / "data" / "raw",
                project_root / "data" / "interim",
                input_fn,
                output_fn,
            )
        elif choice == "2":
            _handle_split(input_fn, output_fn, project_root)
        elif choice == "3":
            _handle_experiment(input_fn, output_fn, project_root)
        elif choice == "4":
            _handle_continuation(input_fn, output_fn, project_root)
        else:
            output_fn("无效命令，请输入 0 到 4。")


def main() -> int:
    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
