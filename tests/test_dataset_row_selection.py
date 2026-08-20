import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.tk_app import _select_dataset_records


class DatasetRowSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [(number, f"row {number}") for number in range(1, 101)]

    def test_returns_all_rows_when_requested_count_reaches_dataset_size(self) -> None:
        self.assertEqual(_select_dataset_records(self.records, 100), self.records)
        self.assertEqual(_select_dataset_records(self.records, 500), self.records)

    def test_returns_requested_random_sample_in_source_order(self) -> None:
        selected = _select_dataset_records(self.records, 10)

        self.assertEqual(len(selected), 10)
        self.assertEqual(selected, sorted(selected, key=lambda item: item[0]))
        self.assertNotEqual(selected, self.records[:10])

    def test_random_sample_is_reproducible(self) -> None:
        self.assertEqual(
            _select_dataset_records(self.records, 12),
            _select_dataset_records(self.records, 12),
        )

    def test_rejects_non_positive_row_count(self) -> None:
        with self.assertRaises(ValueError):
            _select_dataset_records(self.records, 0)


if __name__ == "__main__":
    unittest.main()
