import csv
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from n_gram import _report_previous_result
from src.experiment import (
    MODEL_DIRECTORY_NAMES,
    export_metrics_csv,
    model_directory_name,
    run_experiment,
)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.processed_dir = self.root / "data" / "processed"
        self.processed_dir.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_split(self, name, sequences):
        path = self.processed_dir / "{}.jsonl".format(name)
        with path.open("w", encoding="utf-8") as output:
            for index, tokens in enumerate(sequences, 1):
                record = {
                    "article_id": "article-{}".format(index),
                    "paragraph_id": "article-{}-001".format(index),
                    "tokens": tokens,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_all_splits(self):
        self.write_split("train", [["a", "b", "a"], ["a", "a"]])
        self.write_split("val", [["a", "unseen"]])
        self.write_split("test", [["b", "a"]])

    def test_trains_selects_k_and_saves_model_vocabulary_and_metadata(self):
        self.write_all_splits()
        messages = []

        result = run_experiment(2, self.root, messages.append)
        metadata = json.loads(result["metadata_path"].read_text(encoding="utf-8"))
        vocabulary = json.loads(result["vocabulary_path"].read_text(encoding="utf-8"))
        metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))

        self.assertEqual(result["model_name"], "bigram")
        self.assertIn("<UNK>", vocabulary["tokens"])
        self.assertIn("</s>", vocabulary["tokens"])
        self.assertNotIn("b", vocabulary["tokens"])
        self.assertEqual(metadata["selected_k"], metrics["selected_k"])
        self.assertEqual(metadata["selected_method"], metrics["selected_method"])
        self.assertEqual(metadata["selected_candidate"], metrics["selected_candidate"])
        self.assertEqual(len(metrics["validation_by_candidate"]), 7)
        self.assertEqual(len(metrics["test"]["by_candidate"]), 7)
        self.assertEqual(len(metrics["test"]["add_k_by_k"]), 5)
        self.assertEqual(
            metrics["selected_candidate"],
            min(
                metrics["validation_by_candidate"],
                key=lambda candidate: metrics["validation_by_candidate"][candidate][
                    "cross_entropy"
                ],
            ),
        )
        self.assertEqual(
            metrics["selected_test_metrics"],
            metrics["test"]["by_candidate"][metrics["selected_candidate"]],
        )
        self.assertEqual(
            metadata["test_selected_perplexity"],
            metrics["selected_test_metrics"]["perplexity"],
        )
        if metrics["selected_method"] == "add_k":
            self.assertIn(metadata["selected_k"], (1.0, 0.1, 0.01, 0.001, 0.0001))
        else:
            self.assertIsNone(metadata["selected_k"])
        self.assertEqual(metrics["test"]["add_k"]["event_count"], 3)
        self.assertEqual(metrics["test"]["add_k"]["unknown_token_count"], 1)
        self.assertEqual(metrics["test"]["mle"]["zero_probability_event_count"], 1)
        self.assertIsNone(metrics["test"]["mle"]["perplexity"])
        for method in ("good_turing", "katz"):
            self.assertIn(method, metrics["validation_by_smoothing"])
            self.assertEqual(metrics["test"][method]["event_count"], 3)
            self.assertTrue(math.isfinite(metrics["test"][method]["cross_entropy"]))
            self.assertTrue(math.isfinite(metrics["test"][method]["perplexity"]))
        self.assertEqual(metrics["k_candidates"], [1.0, 0.1, 0.01, 0.001, 0.0001])
        self.assertEqual(metadata["timing"], metrics["timing"])
        for key in (
            "vocabulary_scan_seconds",
            "ngram_counting_seconds",
            "validation_evaluation_seconds",
            "test_evaluation_seconds",
            "model_artifact_save_seconds",
            "measured_total_seconds",
        ):
            self.assertGreaterEqual(metrics["timing"][key], 0)
        self.assertGreater(metrics["training"]["unique_ngram_count"], 0)
        self.assertIn("耗时统计", "\n".join(messages))
        self.assertFalse(any(message.lstrip().startswith("{") for message in messages))
        self.assertTrue(result["counts_path"].is_file())
        with gzip.open(result["counts_path"], "rt", encoding="utf-8") as counts_file:
            first_count = json.loads(next(counts_file))
        self.assertIn("token_ids", first_count)
        self.assertIn("context_count", first_count)
        self.assertTrue(any("n=2 实验完成" in message for message in messages))

    def test_previous_experiment_result_is_reported_as_readable_summary(self):
        metrics_path = self.root / "experiment_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "n": 3,
                    "model_name": "trigram",
                    "selected_k": 0.1,
                    "selected_validation_metrics": {"cross_entropy": 1.25},
                    "test": {"add_k": {"perplexity": 3.5}},
                    "training": {"event_count": 100},
                    "timing": {"measured_total_seconds": 2.75},
                }
            ),
            encoding="utf-8",
        )
        messages = []

        _report_previous_result(metrics_path, messages.append)

        self.assertIn("n=3", messages[0])
        self.assertIn("测试困惑度=3.500000", messages[0])
        self.assertIn("总耗时=2.750s", messages[0])
        self.assertNotIn("{", messages[0])

    def test_saves_each_supported_order_to_its_named_model_directory(self):
        self.assertEqual(
            MODEL_DIRECTORY_NAMES,
            {
                1: "unigram",
                2: "bigram",
                3: "trigram",
                4: "4-gram",
                5: "5-gram",
                6: "6-gram",
            },
        )

    def test_rejects_unsupported_n(self):
        with self.assertRaises(ValueError):
            run_experiment(7, self.root, lambda _message: None)

    def test_requires_all_split_files(self):
        with self.assertRaises(FileNotFoundError):
            run_experiment(1, self.root, lambda _message: None)

    def test_exports_one_row_per_candidate_and_mle_baseline(self):
        metrics_dir = self.root / "result" / "metrics"
        metrics_dir.mkdir(parents=True)
        for n in MODEL_DIRECTORY_NAMES:
            model_name = model_directory_name(n)
            candidate_metrics = {
                "smoothing_method": "add_k",
                "smoothing_k": 0.1,
                "cross_entropy": 1.0,
                "perplexity": math.e,
            }
            metrics = {
                "selected_candidate": "add_k:0.1",
                "selected_method": "add_k",
                "selected_k": 0.1,
                "validation_by_candidate": {"add_k:0.1": candidate_metrics},
                "test": {
                    "by_candidate": {"add_k:0.1": candidate_metrics},
                    "mle": {"perplexity": None},
                },
                "training": {},
                "timing": {},
            }
            (metrics_dir / "{}_metrics.json".format(model_name)).write_text(
                json.dumps(metrics), encoding="utf-8"
            )

        output_path = export_metrics_csv(self.root)
        with output_path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))

        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(row["selected"] == "True" for row in rows), 6)
        self.assertEqual(sum(row["candidate"] == "mle_baseline" for row in rows), 6)


if __name__ == "__main__":
    unittest.main()
