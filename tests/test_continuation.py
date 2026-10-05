import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from n_gram import _handle_continuation
from src.continuation import (
    _choose_unseen_id,
    _sample_next_id,
    _sample_weighted_next_id,
    continue_text,
)


class ContinuationTests(unittest.TestCase):
    def test_choose_unseen_id_skips_observed_ids(self):
        self.assertEqual(_choose_unseen_id([0, 2, 4], 0, 6), 1)
        self.assertEqual(_choose_unseen_id([0, 2, 4], 1, 6), 3)
        self.assertEqual(_choose_unseen_id([0, 2, 4], 2, 6), 5)

    def test_sampling_observed_only_distribution_returns_observed_token(self):
        import random

        result = _sample_next_id({2: 5}, 1, 0.1, random.Random(7))
        self.assertEqual(result, 2)

    def test_weighted_sampler_handles_observed_and_unseen_tokens(self):
        import random

        self.assertEqual(
            _sample_weighted_next_id({2: 1.0}, 3, 0.0, random.Random(7)), 2
        )
        self.assertIn(
            _sample_weighted_next_id({}, 3, 1.0, random.Random(7)), range(3)
        )

    def test_continuation_loads_selected_non_add_k_smoothing(self):
        import jieba

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_dir = root / "model" / "unigram"
            model_dir.mkdir(parents=True)
            (model_dir / "metadata.json").write_text(
                json.dumps({"n": 1, "selected_method": "katz"}), encoding="utf-8"
            )
            (model_dir / "vocabulary.json").write_text(
                json.dumps({"tokens": ["词", "<UNK>", "</s>"]}),
                encoding="utf-8",
            )
            with gzip.open(model_dir / "ngram_counts.jsonl.gz", "wt", encoding="utf-8") as counts:
                counts.write(json.dumps({"token_ids": [0], "count": 2}) + "\n")
                counts.write(json.dumps({"token_ids": [2], "count": 1}) + "\n")
            text_path = root / "prompt.txt"
            text_path.write_text("词", encoding="utf-8")

            with patch.object(jieba, "cut", return_value=["词"]):
                result = continue_text(text_path, 3, 1, root)

            self.assertIsInstance(result, str)

    def test_interactive_continuation_saves_prefix_and_generated_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "test.txt"
            input_path.write_text("前缀", encoding="utf-8")
            answers = iter(("1", str(input_path), "20"))
            messages = []

            with patch("n_gram.continue_text", return_value="续写"):
                _handle_continuation(lambda _prompt: next(answers), messages.append, root)

            result_path = root / "result" / "continuation" / "test_unigram.txt"
            self.assertEqual(result_path.read_text(encoding="utf-8"), "前缀续写")
            self.assertTrue(any(str(result_path) in message for message in messages))

    def test_existing_continuation_result_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "test.txt"
            input_path.write_text("前缀", encoding="utf-8")
            output_path = root / "result" / "continuation" / "test_unigram.txt"
            output_path.parent.mkdir(parents=True)
            output_path.write_text("旧结果", encoding="utf-8")
            answers = iter(("1", str(input_path), "20", "n"))

            with patch("n_gram.continue_text") as continue_text:
                _handle_continuation(lambda _prompt: next(answers), lambda _message: None, root)

            continue_text.assert_not_called()
            self.assertEqual(output_path.read_text(encoding="utf-8"), "旧结果")


if __name__ == "__main__":
    unittest.main()