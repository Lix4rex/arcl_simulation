from __future__ import annotations

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from agent.architecture.demonstrations import (
    DATASET_FORMAT,
    DemonstrationRecorder,
    operator_choice_to_decision,
)
from agent.architecture.policy import AgentPolicy
from agent.architecture.record_strategy import StrategyRecordingSession
from agent.architecture.targets import TARGET_DATASET_FORMAT, one_hot_action
from agent.architecture.train_targets import read_target_datasets, target_loss, train


def observation() -> dict:
    return {
        "schema_version": 2,
        "sequence": 12,
        "timestamp": "2026-10-22T12:00:00+00:00",
        "arena": {"bounds": {"x_min": 0.0, "x_max": 3.0, "y_min": 0.0, "y_max": 2.0}},
        "robot": {
            "pose": {"x": 1.0, "y": 1.0, "yaw": 0.25},
            "pose_age_s": 0.0,
            "velocity": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "wrist_angle_rad": 0.0,
            "suction_states": {"1": "free", "2": "free", "3": "free"},
        },
        "cartons": [],
        "obstacles": [],
        "active_operation": None,
        "last_action_result": None,
        "sources": {"robot_pose": {"fresh": True, "age_s": 0.0}},
    }


class DemonstrationTests(unittest.TestCase):
    def test_operator_choices_map_to_policy_actions(self) -> None:
        world = observation()
        choices = (
            ("suction", {"group": "all", "enabled": True}, "SUCTION_ALL_ON"),
            ("suction", {"group": "all", "enabled": False}, "SUCTION_ALL_OFF"),
            ("suction", {"group": 2, "enabled": True}, "SUCTION_2_ON"),
            ("suction", {"group": 3, "enabled": False}, "SUCTION_3_OFF"),
            ("cancel", {"operation_type": "MOVE"}, "CANCEL"),
            ("idle", {}, "IDLE"),
        )
        for command, parameters, expected in choices:
            with self.subTest(command=command, parameters=parameters):
                self.assertEqual(
                    operator_choice_to_decision(command, parameters, world)["action"], expected
                )
        move = operator_choice_to_decision(
            "move", {"x": 1.2, "y": 0.7, "theta": math.pi / 3}, world
        )
        self.assertAlmostEqual(move["target_pose"]["sin"], math.sin(math.pi / 3))
        self.assertAlmostEqual(move["target_pose"]["cos"], math.cos(math.pi / 3))
        self.assertEqual(
            operator_choice_to_decision("wrist", {"target_angle_rad": 0.0}, world)["action"],
            "IDLE",
        )
        self.assertEqual(
            operator_choice_to_decision("wrist", {"target_angle_rad": 1.57}, world)["action"],
            "TOGGLE_WRIST",
        )
        world["robot"]["wrist_angle_rad"] = None
        self.assertEqual(
            operator_choice_to_decision("wrist", {"target_angle_rad": 1.57}, world)["action"],
            "UNSUPPORTED_WRIST_TARGET",
        )

    def test_recorder_publishes_only_flat_training_samples_after_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = DemonstrationRecorder(temporary, session_id="session-a")
            move = operator_choice_to_decision(
                "move", {"x": 0.1, "y": 0.7, "theta": -0.2}, observation()
            )
            decision_id = recorder.record_decision(
                observation(), move, command="move", parameters={"x": 0.1},
            )
            recorder.record_decision(
                observation(), {"action": "IDLE"}, command="idle", requires_result=False,
            )
            recorder.stop()
            self.assertFalse(recorder.path.exists())
            self.assertEqual(recorder.pending_results, 1)
            self.assertTrue(recorder.record_result(decision_id, {
                "action": "MOVE", "status": "failed", "failed": True,
            }))
            self.assertTrue(recorder.path.is_file())

            lines = recorder.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            rows = [json.loads(line) for line in lines]
            self.assertTrue(all(row["format"] == TARGET_DATASET_FORMAT == DATASET_FORMAT for row in rows))
            self.assertTrue(all("kind" not in row for row in rows))
            self.assertEqual(rows[0]["chosen_action"], "MOVE")
            self.assertEqual(rows[0]["target_action_probs"], one_hot_action("MOVE"))
            self.assertEqual(rows[0]["target_move_pose"], move["target_pose"])
            self.assertEqual(rows[0]["target_move_pose"]["x"], 0.1)
            self.assertTrue(rows[0]["move_target_valid"])
            self.assertEqual(rows[0]["result"]["status"], "failed")
            self.assertEqual(rows[1]["chosen_action"], "IDLE")
            self.assertEqual(rows[1]["target_action_probs"], one_hot_action("IDLE"))
            self.assertEqual(rows[1]["result"]["status"], "not_applicable")

            sessions, source_counts = read_target_datasets([recorder.path])
            self.assertEqual(source_counts, {"operator": 2})
            samples = sessions["session-a"]
            self.assertEqual(len(samples), 2)
            self.assertTrue(samples[0]["result"]["failed"])

    def test_recorder_appends_multiple_sources_to_one_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target_file = Path(temporary) / "targets.jsonl"
            for session_id, source in (("operator-a", "operator"), ("strategy-a", "strategy")):
                recorder = DemonstrationRecorder(
                    target_file, session_id=session_id, source=source
                )
                recorder.record_decision(
                    observation(), {"action": "IDLE"}, command="idle", requires_result=False,
                )
                recorder.stop()

            rows = [json.loads(line) for line in target_file.read_text().splitlines()]
            groups, source_counts = read_target_datasets([target_file])
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(groups), {"operator-a", "strategy-a"})
            self.assertEqual(source_counts, {"operator": 1, "strategy": 1})

    def test_strategy_events_record_actions_and_idle_but_not_centering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session = StrategyRecordingSession(temporary, session_id="strategy-a")
            session.update_observation(observation())
            session.handle_event({
                "event": "start", "id": 1, "command": "move",
                "parameters": {"x": 1.2, "y": 0.7, "theta": 0.5},
                "operation_type": "MOVE", "cancelable": True,
            }, now=0.0)
            self.assertFalse(session.sample_idle(now=0.05))
            self.assertTrue(session.sample_idle(now=0.1))
            session.handle_event({"event": "result", "id": 1, "status": "succeeded"})
            session.handle_event({
                "event": "start", "id": 2, "command": "rock_centering",
                "parameters": {}, "operation_type": "CENTRAGE", "cancelable": False,
            }, now=0.2)
            self.assertFalse(session.sample_idle(now=0.4))
            session.handle_event({"event": "result", "id": 2, "status": "succeeded"})
            session.handle_event({
                "event": "start", "id": 3, "command": "suction",
                "parameters": {"group": "all", "enabled": True},
                "operation_type": "SUCTION_ALL_ON", "cancelable": False,
            }, now=0.5)
            self.assertTrue(session.sample_idle(now=0.6))
            session.handle_event({"event": "result", "id": 3, "status": "succeeded"})
            session.handle_event({"event": "finish", "status": "succeeded"})

            rows = [json.loads(line) for line in session.path.read_text().splitlines()]
            self.assertEqual([row["chosen_action"] for row in rows], [
                "MOVE", "IDLE", "SUCTION_ALL_ON", "IDLE",
            ])
            self.assertTrue(all(row["source"] == "strategy" for row in rows))
            self.assertEqual(rows[0]["result"]["status"], "succeeded")
            self.assertEqual(rows[1]["observation"]["active_operation"]["type"], "MOVE")
            self.assertEqual(rows[3]["observation"]["active_operation"]["type"], "SUCTION_ALL_ON")
            groups, _ = read_target_datasets([session.path])
            self.assertEqual(len(groups["strategy-a"]), 4)

    def test_direct_demonstration_target_has_finite_loss_and_move_gradients(self) -> None:
        policy = AgentPolicy()
        sample = {
            "observation": observation(),
            "target_action_probs": one_hot_action("MOVE"),
            "target_move_pose": {
                "x": 1.25,
                "y": 0.8,
                "sin": math.sin(0.6),
                "cos": math.cos(0.6),
            },
            "move_target_valid": True,
        }
        weights = torch.ones(len(policy.network.action_head[-1].bias))
        loss, _ = target_loss(policy, sample, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in policy.network.parameters()
        ))
        decoded = policy.decode_move_tensor(observation(), torch.zeros((1, 4)))
        self.assertEqual(tuple(decoded.shape), (1, 4))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(decoded[0, 2:4])), 1.0)

    def test_flat_session_files_train_and_reload_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "demonstrations"
            dataset.mkdir()
            for session_id in ("session-train", "session-validation"):
                recorder = DemonstrationRecorder(dataset, session_id=session_id)
                recorder.record_decision(
                    observation(),
                    {"action": "MOVE", "target_pose": {
                        "x": 1.25, "y": 0.8, "sin": math.sin(0.6), "cos": math.cos(0.6),
                    }},
                    command="move",
                    requires_result=False,
                )
                recorder.stop()
            checkpoint = root / "imitation.pt"
            report = train(Namespace(
                datasets=[dataset], validation_data=None, validation_fraction=0.5,
                seed=7, test_data=None, model=None, device="cpu", position_weight=1.0,
                orientation_weight=1.0, learning_rate=0.0003, output=checkpoint,
                batch_size=2, epochs=1,
            ))
            self.assertEqual(report["original_train_samples"], 1)
            self.assertEqual(report["balanced_train_samples"], 1)
            self.assertEqual(report["train_samples"], 2)
            self.assertEqual(report["validation_samples"], 1)
            self.assertTrue(checkpoint.is_file())
            policy = AgentPolicy()
            policy.load_checkpoint(checkpoint)
            self.assertEqual(policy.training_status, "trained_targets")


if __name__ == "__main__":
    unittest.main()
