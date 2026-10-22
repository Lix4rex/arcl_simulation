from __future__ import annotations

import unittest

from agent.architecture.world_observer import SnapshotAssembler


class SuctionObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assembler = SnapshotAssembler({
            "arena": {"bounds": {"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 2}},
        })

    def snapshot(self, now: float = 10) -> dict:
        return self.assembler.build(
            sequence=1, timestamp=now, now=now, carton_source={},
        )

    def test_no_message_means_unknown(self) -> None:
        snapshot = self.snapshot()
        self.assertIsNone(snapshot["robot"]["suction_state"])
        self.assertEqual(snapshot["sources"]["suction"]["status"], "unknown")

    def test_transitions_translate_without_inventing_payload_or_contacts(self) -> None:
        for raw, expected in (
            ("libre", "free"), ("aspiration", "aspirating"),
            ("tenu", "holding"), ("descente", "lowering"),
        ):
            with self.subTest(state=raw):
                self.assembler.update_suction(raw, received_at=9)
                snapshot = self.snapshot()
                self.assertEqual(snapshot["robot"]["suction_state"], expected)
                self.assertEqual(snapshot["sources"]["suction"]["age_s"], 1)
                for key in ("payload_state", "held_object", "contact_confirmed", "contact_target_id"):
                    self.assertIsNone(snapshot["robot"][key])

    def test_state_is_retained_between_transition_messages(self) -> None:
        self.assembler.update_suction("tenu", received_at=1)
        snapshot = self.snapshot(now=100)
        self.assertEqual(snapshot["robot"]["suction_state"], "holding")
        self.assertEqual(snapshot["sources"]["suction"]["update_mode"], "on_change")

    def test_invalid_message_or_bridge_loss_clears_previous_state(self) -> None:
        for value in ("invalid", None):
            with self.subTest(value=value):
                self.assembler.update_suction("tenu", received_at=1)
                self.assembler.update_suction(value, received_at=2)
                snapshot = self.snapshot()
                self.assertIsNone(snapshot["robot"]["suction_state"])
                self.assertEqual(snapshot["sources"]["suction"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
