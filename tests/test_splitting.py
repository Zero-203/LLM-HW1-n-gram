import json
import tempfile
import unittest
from pathlib import Path

from src.splitting import split_dataset


class SplitDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "interim"
        self.output_dir = self.root / "processed"
        self.input_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_month(self, month, article_count):
        source_file = "{}.txt".format(month)
        output_path = self.input_dir / "{}.cleaned.jsonl".format(month)
        with output_path.open("w", encoding="utf-8") as output:
            for index in range(1, article_count + 1):
                article_id = "{}01-01-{:03d}".format(month, index)
                for paragraph in (1, 2):
                    record = {
                        "source_file": source_file,
                        "article_id": article_id,
                        "paragraph_id": "{}-{:03d}".format(article_id, paragraph),
                        "tokens": ["词{}".format(index), "段{}".format(paragraph)],
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_splits_articles_per_month_and_keeps_paragraphs_together(self):
        self.write_month("199801", 10)
        self.write_month("199802", 10)

        result = split_dataset(self.input_dir, self.output_dir, (8, 1, 1), 42)
        manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["monthly_article_counts"],
            {
                "199801": {"train": 8, "val": 1, "test": 1},
                "199802": {"train": 8, "val": 1, "test": 1},
            },
        )
        seen_articles = {}
        for split in ("train", "val", "test"):
            output_path = result["output_paths"][split]
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            for record in records:
                seen_articles.setdefault(record["article_id"], set()).add(split)
            self.assertEqual(len(records) % 2, 0)

        self.assertEqual(len(seen_articles), 20)
        self.assertTrue(all(len(splits) == 1 for splits in seen_articles.values()))

    def test_identical_articles_are_assigned_to_the_same_split(self):
        for month in ("199801", "199802"):
            record = {
                "source_file": "{}.txt".format(month),
                "article_id": "{}01-01-001".format(month),
                "paragraph_id": "{}01-01-001-001".format(month),
                "tokens": ["完全", "相同", "的", "文章"],
            }
            path = self.input_dir / "{}.cleaned.jsonl".format(month)
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

        result = split_dataset(self.input_dir, self.output_dir, (8, 1, 1), 42)
        manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
        duplicate_assignments = [
            item["split"]
            for item in manifest["assignments"]
            if item["article_id"].endswith("001")
        ]

        self.assertEqual(manifest["exact_duplicate_group_count"], 1)
        self.assertEqual(len(set(duplicate_assignments)), 1)

    def test_duplicate_articles_with_different_paragraph_boundaries_are_grouped(self):
        first = {
            "source_file": "199801.txt",
            "article_id": "19980101-01-001",
            "paragraph_id": "19980101-01-001-001",
            "tokens": ["相同"],
        }
        second = {
            "source_file": "199801.txt",
            "article_id": "19980101-01-001",
            "paragraph_id": "19980101-01-001-002",
            "tokens": ["文章"],
        }
        duplicate = {
            "source_file": "199802.txt",
            "article_id": "19980201-01-001",
            "paragraph_id": "19980201-01-001-001",
            "tokens": ["相同", "文章"],
        }
        (self.input_dir / "199801.cleaned.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (first, second)),
            encoding="utf-8",
        )
        (self.input_dir / "199802.cleaned.jsonl").write_text(
            json.dumps(duplicate, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        result = split_dataset(self.input_dir, self.output_dir, (8, 1, 1), 42)
        manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
        duplicate_assignments = [
            item["split"]
            for item in manifest["assignments"]
            if item["article_id"].endswith("001")
        ]

        self.assertEqual(manifest["exact_duplicate_group_count"], 1)
        self.assertEqual(len(set(duplicate_assignments)), 1)

    def test_duplicate_paragraph_ids_are_counted_in_manifest(self):
        record = {
            "source_file": "199801.txt",
            "article_id": "19980101-01-001",
            "paragraph_id": "19980101-01-001-001",
            "tokens": ["重复段落"],
        }
        (self.input_dir / "199801.cleaned.jsonl").write_text(
            (json.dumps(record, ensure_ascii=False) + "\n") * 2, encoding="utf-8"
        )

        result = split_dataset(self.input_dir, self.output_dir, (8, 1, 1), 42)
        manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))

        self.assertEqual(manifest["duplicate_paragraph_id_count"], 1)
        self.assertEqual(
            manifest["duplicate_paragraph_id_examples"][0]["paragraph_id"],
            "19980101-01-001-001",
        )

    def test_same_seed_produces_identical_outputs(self):
        self.write_month("199801", 10)
        first = split_dataset(self.input_dir, self.output_dir, (8, 1, 1), 7)
        first_manifest = first["manifest_path"].read_text(encoding="utf-8")
        first_outputs = {
            split: path.read_text(encoding="utf-8")
            for split, path in first["output_paths"].items()
        }

        second_dir = self.root / "processed-again"
        second = split_dataset(self.input_dir, second_dir, (8, 1, 1), 7)

        self.assertEqual(
            first_manifest, second["manifest_path"].read_text(encoding="utf-8")
        )
        self.assertEqual(
            first_outputs,
            {
                split: path.read_text(encoding="utf-8")
                for split, path in second["output_paths"].items()
            },
        )

    def test_rejects_invalid_proportions(self):
        self.write_month("199801", 1)
        with self.assertRaises(ValueError):
            split_dataset(self.input_dir, self.output_dir, (8, 0, 1), 42)


if __name__ == "__main__":
    unittest.main()
