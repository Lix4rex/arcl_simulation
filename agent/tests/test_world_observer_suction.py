from __future__ import annotations

import unittest

from agent.architecture.world_observer import SnapshotAssembler, _velocity_to_map


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
        self.assertIsNone(snapshot["robot"]["suction_states"].get("1"))
        self.assertEqual(snapshot["sources"]["suction_1"]["status"], "unknown")

    def test_transitions_translate_without_inventing_payload_or_contacts(self) -> None:
        for raw, expected in (
            ("libre", "free"), ("aspiration", "aspirating"),
            ("tenu", "holding"), ("descente", "lowering"),
        ):
            with self.subTest(state=raw):
                self.assembler.update_actuator_suction(1, raw, received_at=9)
                snapshot = self.snapshot()
                self.assertEqual(snapshot["robot"]["suction_states"]["1"], expected)
                self.assertEqual(snapshot["sources"]["suction_1"]["age_s"], 1)
                for key in ("payload_state", "held_object", "contact_confirmed", "contact_target_id"):
                    self.assertIsNone(snapshot["robot"][key])

    def test_state_is_retained_between_transition_messages(self) -> None:
        self.assembler.update_actuator_suction(2, "tenu", received_at=99)
        snapshot = self.snapshot(now=100)
        self.assertEqual(snapshot["robot"]["suction_states"]["2"], "holding")
        self.assertEqual(snapshot["sources"]["suction_2"]["age_s"], 1)

    def test_invalid_message_or_bridge_loss_clears_previous_state(self) -> None:
        for value in ("invalid", None):
            with self.subTest(value=value):
                self.assembler.update_actuator_suction(3, "tenu", received_at=1)
                self.assembler.update_actuator_suction(3, value, received_at=2)
                snapshot = self.snapshot()
                self.assertIsNone(snapshot["robot"]["suction_states"]["3"])
                self.assertEqual(snapshot["sources"]["suction_3"]["status"], "unknown")

    def test_odometry_velocity_is_rotated_into_map(self) -> None:
        velocity = _velocity_to_map(1.0, 0.0, 0.5, 3.141592653589793 / 2.0)
        self.assertAlmostEqual(velocity["x"], 0.0, places=7)
        self.assertAlmostEqual(velocity["y"], 1.0, places=7)
        self.assertAlmostEqual(velocity["yaw"], 0.5)


if __name__ == "__main__":
    unittest.main()
