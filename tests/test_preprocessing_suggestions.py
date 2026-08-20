import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import PreprocessConfig, TextPreprocessor


class PunctuationSuggestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = TextPreprocessor()

    def test_normal_sentence_punctuation_does_not_trigger_removal(self) -> None:
        texts = [
            "This is a normal sentence.",
            "Is everything working correctly?",
            "Yes, it is working correctly.",
        ] * 20

        config, stats = self.preprocessor.suggest(texts)

        self.assertFalse(config.remove_punctuation)
        self.assertEqual(stats["rows_with_punctuation_only"], 0.0)
        self.assertEqual(stats["rows_with_repeated_punctuation"], 0.0)

    def test_punctuation_only_rows_trigger_removal_even_when_globally_rare(self) -> None:
        texts = ["ordinary words without symbols"] * 199 + ['. """" ==']

        config, stats = self.preprocessor.suggest(texts)

        self.assertTrue(config.remove_punctuation)
        self.assertEqual(stats["rows_with_punctuation_only"], 0.005)

    def test_repeated_punctuation_clusters_trigger_removal(self) -> None:
        texts = ["ordinary words without symbols"] * 48 + ["?????", '""""" ==']

        config, stats = self.preprocessor.suggest(texts)

        self.assertTrue(config.remove_punctuation)
        self.assertEqual(stats["rows_with_repeated_punctuation"], 0.04)

    def test_remove_punctuation_removes_ascii_and_unicode_marks(self) -> None:
        result = self.preprocessor.apply(
            'Text... "quoted" == test！',
            PreprocessConfig(remove_punctuation=True, normalize_whitespace=True),
        )

        self.assertEqual(result, "Text quoted test")


if __name__ == "__main__":
    unittest.main()
