from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from aporee_source import (  # noqa: E402
    CONTINUOUS_SCENE_OFFSETS_SECONDS,
    candidate_from_document,
    hard_negative_from_document,
    has_minimum_audio,
    is_commercial_license,
)


class AporeeSourceTest(unittest.TestCase):
    def test_continuous_scene_protocol_uses_thirty_seconds(self) -> None:
        self.assertEqual(CONTINUOUS_SCENE_OFFSETS_SECONDS, (0, 10, 20))

    def test_minimum_audio_rejects_short_cached_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            short = Path(directory) / "short.wav"
            long = Path(directory) / "long.wav"
            for path, seconds in ((short, 1), (long, 6)):
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16_000)
                    stream.writeframes(b"\0\0" * 16_000 * seconds)
            self.assertFalse(has_minimum_audio(short))
            self.assertTrue(has_minimum_audio(long))

    def test_accepts_public_domain_and_attribution_only(self) -> None:
        self.assertTrue(is_commercial_license(
            "https://creativecommons.org/publicdomain/mark/1.0/"
        ))
        self.assertTrue(is_commercial_license(
            "http://creativecommons.org/licenses/by/3.0/"
        ))
        self.assertFalse(is_commercial_license(
            "http://creativecommons.org/licenses/by-nc/3.0/"
        ))
        self.assertFalse(is_commercial_license(
            "http://creativecommons.org/licenses/by-sa/3.0/"
        ))

    def test_high_speed_title_and_location_become_ground_truth_group(self) -> None:
        candidate = candidate_from_document("high_speed_train", {
            "identifier": "aporee_65051_75140",
            "title": "KTX high speed train to Busan",
            "description": "Interior where two carriages are connected",
            "creator": "abyssence",
            "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
        })
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.location_group, "aporee-location:65051")
        self.assertEqual(candidate.label, "high_speed_train")

    def test_rejects_description_only_and_noncommercial_matches(self) -> None:
        description_only = candidate_from_document("concert", {
            "identifier": "aporee_1_2",
            "title": "Playground",
            "description": "Near a concert hall",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/",
        })
        noncommercial = candidate_from_document("metro", {
            "identifier": "aporee_1_3",
            "title": "Metro train",
            "description": "",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by-nc/3.0/",
        })
        self.assertIsNone(description_only)
        self.assertIsNone(noncommercial)

    def test_rejects_ambiguous_pre_event_and_outdoor_titles(self) -> None:
        preconcert = candidate_from_document("concert", {
            "identifier": "aporee_2_3",
            "title": "Pre-concert atmosphere",
            "description": "Audience arriving",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
        })
        parking = candidate_from_document("shopping_mall", {
            "identifier": "aporee_2_4",
            "title": "Shopping mall parking lot",
            "description": "Cars passing",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
        })
        self.assertIsNone(preconcert)
        self.assertIsNone(parking)

    def test_rejects_nearby_transport_and_nonmusical_concerts(self) -> None:
        nearby_train = candidate_from_document("high_speed_train", {
            "identifier": "aporee_3_4",
            "title": "Birds with high-speed rail",
            "description": "The train passes in the distance",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/",
        })
        bird_concert = candidate_from_document("concert", {
            "identifier": "aporee_3_5",
            "title": "Nature night concert",
            "description": "Crickets and frogs",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
        })
        self.assertIsNone(nearby_train)
        self.assertIsNone(bird_concert)

    def test_accepts_actual_live_performance(self) -> None:
        candidate = candidate_from_document("concert", {
            "identifier": "aporee_4_5",
            "title": "Open air concert of the youth orchestra",
            "description": "The orchestra performs in the city",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by/3.0/",
        })
        self.assertIsNotNone(candidate)

    def test_turns_only_unambiguous_nearby_scenes_into_hard_negatives(self) -> None:
        nearby = hard_negative_from_document("high_speed_train", {
            "identifier": "aporee_5_6",
            "title": "Birds with high-speed rail",
            "description": "Birds dominate before a passing train",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/",
        })
        entrance = hard_negative_from_document("metro", {
            "identifier": "aporee_5_7",
            "title": "Subway station entrance",
            "description": "People enter the station",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
        })
        self.assertIsNotNone(nearby)
        assert nearby is not None
        self.assertEqual(nearby.label, "other")
        self.assertIsNone(entrance)

    def test_description_can_disqualify_a_weak_positive(self) -> None:
        document = {
            "identifier": "aporee_6_7",
            "title": "THSR Hsinchu Station - In the square",
            "description": "In the square outside the station with traffic",
            "creator": "tester",
            "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
        }
        self.assertIsNone(candidate_from_document("high_speed_train", document))
        negative = hard_negative_from_document("high_speed_train", document)
        self.assertIsNotNone(negative)
        assert negative is not None
        self.assertEqual(negative.label, "other")


if __name__ == "__main__":
    unittest.main()
