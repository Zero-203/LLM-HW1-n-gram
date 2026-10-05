import json
import tempfile
import unittest
from pathlib import Path

from n_gram import parse_proportions, run_interactive
from src.cleaning import clean_corpus, tokenize_tagged_body


class TokenizeTaggedBodyTests(unittest.TestCase):
    def test_removes_pos_and_group_labels_but_keeps_words(self):
        tokens, issues = tokenize_tagged_body(
            "[中国/ns 政府/n]nt １９９８年/t ，/w （/w"
        )

        self.assertEqual(tokens, ["中国", "政府", "１９９８年", "，", "（"])
        self.assertEqual(issues, 0)

    def test_discards_empty_word_marker_and_counts_issue(self):
        tokens, issues = tokenize_tagged_body("同一/v /m 篇/q")

        self.assertEqual(tokens, ["同一", "篇"])
        self.assertEqual(issues, 1)


class CleanCorpusTests(unittest.TestCase):
    def test_outputs_paragraph_records_without_joining_articles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            interim_dir = root / "interim"
            raw_dir.mkdir()
            source = raw_dir / "199801.txt"
            source.write_text(
                "19980101-01-001-001/m 标题/n 。/w\n"
                "\n"
                "19980101-01-002-001/m 正文/n １２３/m\n",
                encoding="utf-8",
            )

            summary = clean_corpus(raw_dir, interim_dir, "1998*.txt")
            output = interim_dir / "199801.cleaned.jsonl"
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["totals"]["article_count"], 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["tokens"], ["标题", "。"])
        self.assertEqual(records[1]["tokens"], ["正文", "１２３"])
        self.assertNotEqual(records[0]["article_id"], records[1]["article_id"])


class CommandLineTests(unittest.TestCase):
    def test_parses_proportions(self):
        self.assertEqual(parse_proportions("8:1:1"), (8.0, 1.0, 1.0))
        self.assertEqual(parse_proportions("0.8:0.1:0.1"), (0.8, 0.1, 0.1))

    def test_clean_command_reports_previous_result_and_preserves_on_no(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interim_dir = root / "data" / "interim"
            interim_dir.mkdir(parents=True)
            stale_output = interim_dir / "199801.cleaned.jsonl"
            stale_output.write_text("old result\n", encoding="utf-8")
            (interim_dir / "cleaning_summary.json").write_text(
                json.dumps({"totals": {"article_count": 23}}), encoding="utf-8"
            )
            choices = iter(["1", "n", "0"])
            messages = []

            result = run_interactive(
                input_fn=lambda _prompt: next(choices),
                output_fn=messages.append,
                project_root=root,
            )
            preserved_output = stale_output.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertEqual(preserved_output, "old result\n")
        self.assertTrue(any("上次操作结果：文章=23" in message for message in messages))
        self.assertTrue(any("保留现有数据" in message for message in messages))

    def test_clean_command_deletes_known_outputs_and_reruns_on_yes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "data" / "raw"
            interim_dir = root / "data" / "interim"
            raw_dir.mkdir(parents=True)
            interim_dir.mkdir(parents=True)
            (raw_dir / "199801.txt").write_text(
                "19980101-01-001-001/m 标题/n 。/w\n", encoding="utf-8"
            )
            (interim_dir / "199801.cleaned.jsonl").write_text(
                "old result\n", encoding="utf-8"
            )
            (interim_dir / "cleaning_summary.json").write_text(
                json.dumps({"totals": {"article_count": 23}}), encoding="utf-8"
            )
            unrelated_file = interim_dir / "notes.txt"
            unrelated_file.write_text("keep", encoding="utf-8")
            choices = iter(["1", "y", "0"])
            messages = []

            run_interactive(
                input_fn=lambda _prompt: next(choices),
                output_fn=messages.append,
                project_root=root,
            )

            cleaned = json.loads(
                (interim_dir / "199801.cleaned.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()[0]
            )
            unrelated_content = unrelated_file.read_text(encoding="utf-8")

        self.assertEqual(cleaned["tokens"], ["标题", "。"])
        self.assertEqual(unrelated_content, "keep")
        self.assertTrue(any("已删除上次清洗" in message for message in messages))

    def test_split_command_uses_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            interim_dir = root / "data" / "interim"
            interim_dir.mkdir(parents=True)
            with (interim_dir / "199801.cleaned.jsonl").open(
                "w", encoding="utf-8"
            ) as output:
                for index in range(1, 11):
                    article_id = "19980101-01-{:03d}".format(index)
                    record = {
                        "source_file": "199801.txt",
                        "article_id": article_id,
                        "paragraph_id": article_id + "-001",
                        "tokens": ["正文{}".format(index)],
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True)
            (processed_dir / "train.jsonl").write_text("old split\n", encoding="utf-8")
            (processed_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "random_seed": 7,
                        "article_count": 5,
                        "split_article_counts": {"train": 5, "val": 0, "test": 0},
                    }
                ),
                encoding="utf-8",
            )
            choices = iter(["2", "", "", "y", "0"])
            messages = []

            run_interactive(
                input_fn=lambda _prompt: next(choices),
                output_fn=messages.append,
                project_root=root,
            )
            manifest = json.loads(
                (root / "data" / "processed" / "split_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["random_seed"], 42)
        self.assertEqual(manifest["split_article_counts"], {"train": 8, "val": 1, "test": 1})
        self.assertTrue(any("数据集划分完成" in message for message in messages))
        self.assertTrue(any("随机种子=7" in message for message in messages))
        self.assertTrue(any("已删除上次数据集划分" in message for message in messages))

    def test_experiment_command_accepts_planned_n_range(self):
        choices = iter(["3", "6", "0"])
        messages = []

        run_interactive(
            input_fn=lambda _prompt: next(choices),
            output_fn=messages.append,
            project_root=Path(".").resolve() / "missing-experiment-test",
        )

        self.assertTrue(any("缺少数据集文件" in message for message in messages))

    def test_continuation_command_collects_text_path_and_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.txt").write_text("测试文本", encoding="utf-8")
            choices = iter(["4", "4", "sample.txt", "30", "0"])
            messages = []

            run_interactive(
                input_fn=lambda _prompt: next(choices),
                output_fn=messages.append,
                project_root=root,
            )

        self.assertTrue(any("文本续写失败：缺少 n-gram 模型文件" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
