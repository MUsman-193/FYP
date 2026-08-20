import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moderation.local_toxicity import (
    ToxicityModelError,
    _analysis_from_parsed,
    _completed_json_row_ids,
    _has_complete_json_container,
    _parse_json_array,
)


class ParseBatchJsonTests(unittest.TestCase):
    def test_parses_complete_array(self) -> None:
        payload = [
            {
                "row_id": 1,
                "overall_score": 0,
                "scores": {
                    "toxicity": 0,
                    "severe_toxicity": 0,
                    "identity_attack": 0,
                    "insult": 0,
                    "profanity": 0,
                    "threat": 0,
                },
                "category": ["None"],
                "toxicity_level": "Low",
                "toxicity_type": [],
            }
        ]

        self.assertEqual(_parse_json_array(json.dumps(payload)), payload)

    def test_salvages_complete_rows_from_truncated_array(self) -> None:
        truncated = """[
        {"row_id":1,"category":["None"],"toxicity_level":"Low","toxicity_type":[]},
        {"row_id":2,"category":["Harassment"],"toxicity_level":"High","toxicity_type":["Insult"]},
        {"row_id":3,"category":
        """

        parsed = _parse_json_array(truncated)

        self.assertEqual([row["row_id"] for row in parsed], [1, 2])

    def test_rejects_output_without_complete_rows(self) -> None:
        with self.assertRaises(ToxicityModelError):
            _parse_json_array('[{"row_id":1,"category":')


class DirectScoreTests(unittest.TestCase):
    def test_maps_compact_score_positions_without_changing_values(self) -> None:
        parsed = {
            "overall_score": 12,
            "scores": [12, 1, 2, 3, 4, 5],
            "category": ["None"],
            "toxicity_level": "Low",
            "toxicity_type": [],
        }

        result = _analysis_from_parsed(parsed, raw_model_output=json.dumps(parsed))

        self.assertEqual(result.overall_score, 12)
        self.assertEqual(
            list(result.scores.values()),
            [12, 1, 2, 3, 4, 5],
        )

    def test_uses_model_scores_without_label_mapping(self) -> None:
        parsed = {
            "overall_score": 7,
            "scores": {
                "toxicity": 8,
                "severe_toxicity": 1,
                "identity_attack": 0,
                "insult": 3,
                "profanity": 0,
                "threat": 0,
            },
            "category": ["Hate Speech"],
            "toxicity_level": "High",
            "toxicity_type": ["Identity Attack"],
            "explanation": "Test payload.",
        }

        result = _analysis_from_parsed(parsed, raw_model_output=json.dumps(parsed))

        self.assertEqual(result.overall_score, 7)
        self.assertEqual(result.scores["Identity Attack"], 0)
        self.assertEqual(result.scores["Toxicity"], 8)

    def test_requires_every_direct_score(self) -> None:
        parsed = {
            "overall_score": 0,
            "scores": {"toxicity": 0},
            "category": ["None"],
            "toxicity_level": "Low",
            "toxicity_type": [],
        }

        with self.assertRaises(ToxicityModelError):
            _analysis_from_parsed(parsed, raw_model_output=json.dumps(parsed))


class JsonCompletionTests(unittest.TestCase):
    def test_detects_complete_array_and_ignores_brackets_in_strings(self) -> None:
        value = '[{"row_id":1,"explanation":"contains ] safely"}] trailing text'

        self.assertTrue(_has_complete_json_container(value, opening="["))

    def test_rejects_truncated_array(self) -> None:
        self.assertFalse(_has_complete_json_container('[{"row_id":1}', opening="["))

    def test_collects_only_complete_row_objects(self) -> None:
        value = '[{"row_id":1,"scores":{}},{"row_id":2,"scores":'

        self.assertEqual(_completed_json_row_ids(value), {1})


if __name__ == "__main__":
    unittest.main()
