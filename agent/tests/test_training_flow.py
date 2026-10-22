from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from agent.architecture.policy import ACTIONS, AgentPolicy
from agent.architecture.action_executor import ActionExecutor
from agent.architecture.train_policy import ppo_loss, read_episodes, split_episodes, train
from agent.records import EpisodeRecords
from agent.rl.environment import TwoRobotArenaEnv


class ToyBackend:
    """Only test data: no ROS, Gazebo or production scoring rules."""

    def __init__(self, seconds: int = 6) -> None:
        self.seconds = seconds
        self.actions: list[dict] = []

    def reset(self) -> dict:
        self.time = 0
        self.pose = {"x": 0.25, "y": 0.5, "yaw": 0.4}
        self.loaded = False
        return self.state()

    def state(self) -> dict:
        carton = {"id": "carton", "posture": "horizontal",
                  "pose": {"x": 1.0, "y": 0.5, "z": 0.055, "yaw": 0},
                  "size": {"x": 0.32, "y": 0.11, "z": 0.11}, "accessible": True,
                  "approach_pose": {"x": 0.6, "y": 0.5, "yaw": 0}}
        held = {**copy.deepcopy(carton), "pose": {"x": self.pose["x"] + 0.3, "y": self.pose["y"], "z": 0.1, "yaw": 0}}
        observations = {}
        for robot in ("blue", "red"):
            observations[robot] = {
                "sequence": self.time + 1, "timestamp": self.time,
                "arena": {"bounds": {"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 2}},
                "robot": {"id": robot, "pose": copy.deepcopy(self.pose) if robot == "blue" else {"x": 2.7, "y": 1.7, "yaw": 0},
                          "pose_age_s": 0, "payload_state": ("loaded" if self.loaded else "empty") if robot == "blue" else None,
                          "held_object": held if robot == "blue" and self.loaded else None},
                "cartons": [copy.deepcopy(carton)], "obstacles": [], "other_robots": [], "aruco_detections": [],
                "zones": [{"id": "zone", "center": {"x": 2, "y": 1}, "yaw": 0,
                           "length": 0.8, "width": 0.6, "capacity": 3, "occupied": 0,
                           "release_allowed": True, "accessible": True,
                           "approach_pose": {"x": 1.7, "y": 1, "yaw": 0}}],
                "mission": {"remaining_s": self.seconds - self.time},
            }
        return {"world": {"game_time_s": self.time,
                          "robots": [{"id": "blue", "pose": {"x": 2.9, "y": 1.9, "yaw": 0}},
                                     {"id": "red", "pose": {"x": 0, "y": 0, "yaw": 0}}],
                          "secret_truth": "never an observation", "scores": {"blue": 5, "red": 0},
                          "match_finished": self.time >= self.seconds},
                "observations": observations}

    def advance(self, actions: dict, *, max_game_seconds: float) -> dict:
        assert max_game_seconds >= 1
        self.actions.append(copy.deepcopy(actions))
        action = actions["blue"]
        if action["action"].startswith("GO_TO_"):
            self.pose = copy.deepcopy(action["target_pose"])
        elif action["action"] == "TAKE":
            self.loaded = True
        elif action["action"] == "RELEASE":
            self.loaded = False
        self.time += 1
        result = self.state()
        result["responses"] = {robot: {"status": "succeeded"} for robot in actions}
        return result


def make_backend() -> ToyBackend:
    return ToyBackend()


def _action_points(previous, actions, responses, next_world, robot_ids):
    return {robot: 0.1 for robot in robot_ids}


def _terminal_points(winner, robot_ids):
    return {robot: 1.0 if robot == winner else -1.0 for robot in robot_ids}


def _collect(policy: AgentPolicy, destination: Path, *, rewarded: bool = True, plan: bool = False) -> tuple[ToyBackend, dict]:
    backend = ToyBackend()
    env = TwoRobotArenaEnv(backend.advance, controlled_robot_id="blue", opponent_robot_id="red",
                           records_path=destination, episode_seconds=backend.seconds, gamma=0.9,
                           action_rewarder=_action_points if rewarded else None,
                           terminal_rewarder=_terminal_points if rewarded else None)
    env.reset(backend.reset())
    scripted = ("GO_TO_CARTON", "CENTER", "TAKE", "GO_TO_ZONE", "RELEASE", "WAIT")
    for index in range(backend.seconds):
        before = env.agent_observations
        if plan:
            # Bias, then sample the real masked distributions; not a fabricated decision.
            with torch.no_grad():
                head = policy.network.action_head[-1]
                head.weight.zero_()
                head.bias.fill_(-30)
                head.bias[ACTIONS.index(scripted[index])] = 30
        decisions = [policy.decide(before[robot], deterministic=False) for robot in env.robot_ids]
        _, _, _, _, info = env.step(*decisions)
        if index < backend.seconds - 1:
            assert not destination.exists() or not list(destination.glob("*.jsonl"))
    env.close()
    return backend, info


class TrainingFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.policy = AgentPolicy()

    def test_policy_can_attempt_every_action_regardless_of_payload(self) -> None:
        executor = ActionExecutor()
        for payload in (None, "empty", "loaded"):
            world = ToyBackend().reset()["observations"]["blue"]
            world["robot"]["payload_state"] = payload
            world["cartons"][0]["accessible"] = False
            world["zones"][0].update(accessible=False, release_allowed=False, capacity=0)
            world["last_action_result"] = {"status": "failed", "finished_sequence": world["sequence"]}
            for action in ACTIONS:
                with self.subTest(payload=payload, action=action):
                    with torch.no_grad():
                        head = self.policy.network.action_head[-1]
                        head.weight.zero_()
                        head.bias.fill_(-30)
                        head.bias[ACTIONS.index(action)] = 30
                    decision = self.policy.decide(world)
                    self.assertEqual(decision["action"], action)
                    log_prob, _, _ = self.policy.evaluate_action(world, decision)
                    self.assertAlmostEqual(float(log_prob.detach()), decision["behavior"]["log_prob"])
                    if action != "WAIT":
                        argv, _ = executor._command(decision, world)
                        self.assertEqual(argv[0], "ros2")

    def test_manipulation_needs_no_target_but_navigation_does(self) -> None:
        world = ToyBackend().reset()["observations"]["blue"]
        world["robot"]["payload_state"] = None
        world["cartons"] = []
        world["zones"] = []
        output = self.policy.network(world)
        distribution, _ = self.policy._distributions(world, output)
        self.assertEqual(torch.isfinite(distribution.logits).tolist(),
                         [True, False, False, True, True, True])
        world["active_operation"] = {"type": "TAKE", "status": "running"}
        self.assertEqual(self.policy.decide(world)["action"], "WAIT")
        self.assertEqual(self.policy.decide(world)["reason"], "operation in progress")

    def test_six_step_flow_relabels_all_records_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode"
            backend, info = _collect(self.policy, path, plan=True)
            self.assertEqual([step["blue"]["action"] for step in backend.actions],
                             ["GO_TO_CARTON", "CENTER", "TAKE", "GO_TO_ZONE", "RELEASE", "WAIT"])
            self.assertEqual(backend.actions[0]["blue"]["target_pose"], {"x": 0.6, "y": 0.5, "yaw": 0})
            rows = [json.loads(line) for line in next(path.glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(all(row["winner_targets"] == {"blue": 1, "red": 0} for row in rows))
            self.assertTrue(all(row["complete"] and row["rewards_complete"] for row in rows))
            transitions = [row for row in rows if row["kind"] == "transition"]
            self.assertEqual(len(transitions), 6)
            expected = sum(0.1 * 0.9 ** step for step in range(6)) + 0.9 ** 5
            self.assertAlmostEqual(transitions[0]["return_targets"]["blue"], expected)
            self.assertAlmostEqual(transitions[-1]["reward"]["blue"], 1.1)
            for row in transitions:
                self.assertNotIn("secret_truth", row["agent_observations"]["blue"])
                self.assertIn("secret_truth", row["world"])
            self.assertNotEqual(transitions[0]["actions"]["blue"]["target_pose"], transitions[0]["backend_actions"]["blue"]["target_pose"])
            self.assertFalse(info["rewards_pending"])
            episodes, _ = read_episodes([path])
            self.assertEqual(len(next(iter(episodes.values()))), 12)

    def test_pending_reward_functions_do_not_create_fake_training_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes"
            _, info = _collect(self.policy, path, rewarded=False)
            self.assertTrue(info["rewards_pending"])
            rows = [json.loads(line) for line in next(path.glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(all(row["winner_targets"] is not None for row in rows))
            self.assertTrue(all(row.get("return_targets") is None for row in rows))
            with self.assertRaisesRegex(ValueError, "no trainable episodes"):
                read_episodes([path])

    def test_interrupted_episode_is_saved_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ToyBackend()
            env = TwoRobotArenaEnv(backend.advance, controlled_robot_id="blue", opponent_robot_id="red", records_path=directory)
            env.reset(backend.reset())
            env.step("WAIT", "WAIT")
            env.close()
            path = next(Path(directory).glob("*.jsonl"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(all(not row["complete"] and row["winner_targets"] is None for row in rows))
            env.close()
            self.assertEqual(path.read_text().count('"kind":"episode_end"'), 1)
            with self.assertRaisesRegex(ValueError, "no trainable episodes"):
                read_episodes([path])

    def test_publication_error_never_exposes_a_partial_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = EpisodeRecords(directory, "example", ("blue", "red"))
            records.add({"kind": "decision", "actions": {"blue": {"action": "WAIT"}}})
            with patch("agent.records.os.replace", side_effect=OSError("test disk failure")):
                with self.assertRaises(OSError):
                    records.finish(complete=True, winner_id="blue", winner_targets={"blue": 1, "red": 0}, rewards_complete=False)
            self.assertFalse(list(Path(directory).glob("*.jsonl")))
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            self.assertIsNone(records.path)
            records.finish(complete=False, winner_id=None, winner_targets=None, rewards_complete=False)
            self.assertEqual(sum(row["kind"] == "episode_end" for row in records.rows), 1)
            self.assertTrue(all(row["winner_targets"] == {"blue": 1, "red": 0} for row in records.rows))

    def test_draw_and_unknown_winner_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for label, targets in (("draw", {"blue": 0.5, "red": 0.5}), ("unknown", None)):
                records = EpisodeRecords(Path(directory) / label, label, ("blue", "red"))
                records.add({"kind": "transition", "reward": None})
                records.finish(complete=True, winner_id=None, winner_targets=targets, rewards_complete=False)
                row = json.loads(records.path.read_text().splitlines()[0])
                self.assertEqual(row["winner_targets"], targets)
                self.assertEqual(row["winner_known"], targets is not None)

    def test_joint_likelihood_and_actor_gradient(self) -> None:
        backend = ToyBackend()
        world = backend.reset()["observations"]["blue"]
        world["cartons"].append({**copy.deepcopy(world["cartons"][0]), "id": "second"})
        world["cartons"][1]["pose"]["x"] = 1.5
        world["cartons"][1]["approach_pose"]["x"] = 1.1
        with torch.no_grad():
            self.policy.network.action_head[-1].bias[ACTIONS.index("GO_TO_CARTON")] = 5
        decision = None
        for _ in range(100):
            candidate = self.policy.decide(world, deterministic=False)
            if candidate["action"] == "GO_TO_CARTON":
                decision = candidate
                break
        self.assertIsNotNone(decision)
        log_prob, _, value = self.policy.evaluate_action(world, decision)
        self.assertAlmostEqual(float(log_prob.detach()), decision["behavior"]["log_prob"], places=5)
        sample = {"observation": world, "decision": decision, "old_log_prob": float(log_prob.detach()),
                  "old_value": float(value.detach()), "return_target": float(value.detach()) + 1}
        loss, _ = ppo_loss(self.policy, sample)
        loss.backward()
        self.assertGreater(float(self.policy.network.action_head[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.policy.network.carton_head[-1].bias.grad.abs().sum()) +
                           float(self.policy.network.carton_head[-1].weight.grad.abs().sum()), 0)

    def test_episode_split_is_reproducible_and_keeps_both_robots_together(self) -> None:
        episodes = {str(index): [{"robot": "blue"}, {"robot": "red"}] for index in range(10)}
        train_set, val_set = split_episodes(episodes, 0.2, 7)
        self.assertFalse(set(train_set) & set(val_set))
        self.assertEqual(len(val_set), 2)
        self.assertEqual((train_set, val_set), split_episodes(episodes, 0.2, 7))
        self.assertTrue(all(len(samples) == 2 for samples in (*train_set.values(), *val_set.values())))

    def test_recorded_episodes_train_and_reload_with_held_out_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, output = root / "collection.pt", root / "trained.pt"
            self.policy.save_checkpoint(model)
            for index in range(3):
                _collect(self.policy, root / "data" / str(index))
            _collect(self.policy, root / "held-out-test")
            args = argparse.Namespace(datasets=[root / "data"], model=model, output=output,
                                      validation_data=None, test_data=[root / "held-out-test"], validation_fraction=0.34,
                                      robot_id=None, seed=7, device="cpu", clip=0.2,
                                      value_weight=0.5, entropy_weight=0.01,
                                      learning_rate=0.001, epochs=2, batch_size=8)
            report = train(args)
            self.assertEqual(len(report["train_episode_ids"]), 2)
            self.assertEqual(len(report["validation_episode_ids"]), 1)
            self.assertFalse(set(report["train_episode_ids"]) & set(report["validation_episode_ids"]))
            self.assertTrue(output.with_suffix(".metrics.json").exists())
            self.assertIn("value_mse", report["test"])
            self.assertFalse(set(report["test_episode_ids"]) &
                             (set(report["train_episode_ids"]) | set(report["validation_episode_ids"])))
            loaded = AgentPolicy()
            loaded.load_checkpoint(output)
            decision = loaded.decide(ToyBackend().reset()["observations"]["blue"])
            self.assertIn(decision["action"], ACTIONS)
            args.validation_data = [root / "data"]
            with self.assertRaisesRegex(ValueError, "disjoint"):
                train(args)

    def test_deterministic_decisions_and_duplicate_episodes_are_not_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _collect(self.policy, root / "data")
            original = next((root / "data").glob("*.jsonl"))
            rows = [json.loads(line) for line in original.read_text().splitlines()]
            for row in rows:
                if row["kind"] == "transition":
                    for decision in row["actions"].values():
                        decision["behavior"]["sampled"] = False
            deterministic = root / "deterministic.jsonl"
            deterministic.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "no trainable episodes"):
                read_episodes([deterministic])
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(original.read_text())
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                read_episodes([original, duplicate])

    def test_time_limit_finishes_and_unknown_results_do_not_become_draws(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ToyBackend()
            def advance(actions, *, max_game_seconds):
                result = backend.advance(actions, max_game_seconds=max_game_seconds)
                result["world"].pop("scores")
                result["world"]["match_finished"] = False
                return result
            env = TwoRobotArenaEnv(advance, controlled_robot_id="blue", opponent_robot_id="red",
                                   episode_seconds=2, records_path=directory,
                                   action_rewarder=_action_points, terminal_rewarder=_terminal_points)
            env.reset(backend.reset())
            env.step("WAIT", "WAIT")
            _, _, terminated, truncated, info = env.step("WAIT", "WAIT")
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["winner_known"])
            self.assertTrue(info["rewards_pending"])
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(all(row["winner_targets"] is None for row in rows))
            self.assertTrue(all(row.get("return_targets") is None for row in rows))

    def test_backend_failure_closes_episode_and_blocks_retry_on_stale_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ToyBackend()
            def failing_advance(actions, *, max_game_seconds):
                raise ValueError("test backend failed")
            env = TwoRobotArenaEnv(failing_advance, controlled_robot_id="blue", opponent_robot_id="red", records_path=directory)
            env.reset(backend.reset())
            with self.assertRaisesRegex(ValueError, "test backend failed"):
                env.step("WAIT", "WAIT")
            self.assertTrue(env.done)
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(any(row["kind"] == "decision" for row in rows))
            self.assertTrue(all(not row["complete"] for row in rows))
            with self.assertRaises(RuntimeError):
                env.step("WAIT", "WAIT")

    def test_runner_eof_is_not_a_completed_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = ToyBackend().reset()["observations"]["blue"]
            process = subprocess.run([
                sys.executable, "-m", "agent.architecture.run_agent", "--dry-run",
                "--model", str(root / "policy.pt"), "--records", str(root / "records"),
            ], input=json.dumps(world) + "\n", text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            end = json.loads(process.stdout.splitlines()[-1])
            self.assertFalse(end["complete"])
            rows = [json.loads(line) for line in Path(end["records_path"]).read_text().splitlines()]
            self.assertTrue(any(row["kind"] == "decision" for row in rows))
            self.assertTrue(all(not row["complete"] and row["winner_targets"] is None for row in rows))

    def test_runner_persists_decisions_and_does_not_act_on_final_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = ToyBackend().reset()["observations"]["blue"]
            final = copy.deepcopy(world)
            final.update(sequence=2, timestamp=1, match={"finished": True, "winner_id": "blue"})
            process = subprocess.run([
                sys.executable, "-m", "agent.architecture.run_agent", "--dry-run",
                "--model", str(root / "policy.pt"), "--records", str(root / "records"),
            ], input=json.dumps(world) + "\n" + json.dumps(final) + "\n", text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            rows = [json.loads(line) for line in next((root / "records").glob("*.jsonl")).read_text().splitlines()]
            decisions = [row for row in rows if row["kind"] == "decision"]
            self.assertEqual(len(decisions), 1)
            self.assertTrue(all(row["winner_targets"] == {"blue": 1.0} for row in rows))
            self.assertEqual(decisions[0]["responses"]["blue"]["status"], "dry_run")

    def test_collect_cli_records_without_filling_in_reward_stubs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = subprocess.run([
                sys.executable, "-m", "agent.rl.collect", "--backend", "agent.tests.test_training_flow:make_backend",
                "--robot-id", "blue", "--opponent-id", "red", "--episodes", "1", "--episode-seconds", "6",
                "--record-only", "--model", str(root / "policy.pt"), "--records", str(root / "records"),
            ], text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertTrue(result["rewards_pending"])
            self.assertTrue(Path(result["records_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
