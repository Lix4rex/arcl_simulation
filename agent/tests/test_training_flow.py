from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
import subprocess
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from agent.architecture.policy import ACTIONS, AgentPolicy
from agent.architecture.action_executor import ActionExecutor
from agent.architecture.world_encoder import LEGACY_OPERATION_TYPES, WorldEncoder
from agent.architecture.demonstrations import DemonstrationRecorder
from agent.architecture.targets import one_hot_action, selfplay_target_distribution
from agent.architecture.train_targets import (
    augment_length_symmetry,
    balance_action_samples,
    mirror_sample_along_length,
    read_target_datasets,
    split_groups,
    target_loss,
    train,
)
from agent.records import EpisodeRecords
from agent.rl.environment import TwoRobotArenaEnv
from agent.rl.prepare_targets import prepare_targets, read_selfplay_episodes


class ToyBackend:
    """Only test data: no ROS, Gazebo or production scoring rules."""

    def __init__(self, seconds: int = 6) -> None:
        self.seconds = seconds
        self.actions: list[dict] = []

    def reset(self) -> dict:
        self.time = 0
        self.pose = {"x": 0.25, "y": 0.5, "yaw": 0.4}
        self.loaded = False
        self.suction = {"1": "free", "2": "free", "3": "free"}
        self.wrist = 0.0
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
                "schema_version": 2,
                "sequence": self.time + 1, "timestamp": self.time,
                "arena": {"bounds": {"x_min": 0, "x_max": 3, "y_min": 0, "y_max": 2}},
                "robot": {"id": robot, "pose": copy.deepcopy(self.pose) if robot == "blue" else {"x": 2.7, "y": 1.7, "yaw": 0},
                          "pose_age_s": 0, "velocity": {"x": 0, "y": 0, "yaw": 0},
                          "wrist_angle_rad": self.wrist if robot == "blue" else 0.0,
                          "suction_states": copy.deepcopy(self.suction) if robot == "blue" else {"1": "free", "2": "free", "3": "free"},
                          "payload_state": ("loaded" if self.loaded else "empty") if robot == "blue" else None,
                          "held_object": held if robot == "blue" and self.loaded else None},
                "cartons": [copy.deepcopy(carton)], "obstacles": [], "other_robots": [], "aruco_detections": [],
                "active_operation": None, "last_action_result": None,
                "zones": [{"id": "zone", "center": {"x": 2, "y": 1}, "yaw": 0,
                           "length": 0.8, "width": 0.6, "capacity": 3, "occupied": 0,
                           "release_allowed": True, "accessible": True,
                           "approach_pose": {"x": 1.7, "y": 1, "yaw": 0}}],
                "mission": {"remaining_s": self.seconds - self.time, "total_s": self.seconds},
                "sources": {"robot_pose": {"fresh": True, "age_s": 0},
                            "wrist": {"fresh": True, "age_s": 0}},
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
        if action["action"] == "MOVE":
            target = action["target_pose"]
            self.pose = {"x": target["x"], "y": target["y"],
                         "yaw": math.atan2(target["sin"], target["cos"])}
        elif action["action"] == "SUCTION_1_ON":
            self.suction["1"] = "holding"
            self.loaded = True
        elif action["action"] == "SUCTION_1_OFF":
            self.suction["1"] = "free"
            self.loaded = False
        elif action["action"] == "TOGGLE_WRIST":
            self.wrist = 0.0 if self.wrist > 0.0 else 1.5708
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
    scripted = ("MOVE", "TOGGLE_WRIST", "SUCTION_1_ON", "MOVE", "SUCTION_1_OFF", "IDLE")
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

    def test_selfplay_target_distribution_is_bounded_and_preserves_action_mask(self) -> None:
        probabilities = [0.2, 0.0, 0.5, 0.3] + [0.0] * (len(ACTIONS) - 4)
        positive = selfplay_target_distribution(probabilities, "MOVE", 1.0)
        negative = selfplay_target_distribution(probabilities, "MOVE", -1.0)
        neutral = selfplay_target_distribution(probabilities, "MOVE", 0.0)
        self.assertTrue(math.isclose(sum(positive), 1.0))
        self.assertTrue(math.isclose(sum(negative), 1.0))
        self.assertEqual(positive[1], 0.0)
        self.assertEqual(negative[1], 0.0)
        self.assertGreater(positive[ACTIONS.index("MOVE")], probabilities[ACTIONS.index("MOVE")])
        self.assertLess(negative[ACTIONS.index("MOVE")], probabilities[ACTIONS.index("MOVE")])
        self.assertEqual(neutral, probabilities)

    def test_policy_masks_cancel_when_idle_and_can_cancel_only_a_move(self) -> None:
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
                    expected = "IDLE" if action == "CANCEL" else action
                    self.assertEqual(decision["action"], expected)
                    log_prob, _, _ = self.policy.evaluate_action(world, decision)
                    self.assertAlmostEqual(float(log_prob.detach()), decision["behavior"]["log_prob"])
                    if action == "MOVE":
                        self.assertGreaterEqual(decision["target_pose"]["x"], 0.2)
                        self.assertLessEqual(decision["target_pose"]["x"], 2.8)
                        self.assertAlmostEqual(math.hypot(decision["target_pose"]["sin"],
                                                         decision["target_pose"]["cos"]), 1.0)
            world["active_operation"] = {"type": "MOVE", "status": "running", "cancelable": True}
            with torch.no_grad():
                self.policy.network.action_head[-1].bias.fill_(-30)
                self.policy.network.action_head[-1].bias[ACTIONS.index("CANCEL")] = 30
            self.assertEqual(self.policy.decide(world)["action"], "CANCEL")

    def test_policy_fails_closed_when_operation_state_is_unknown(self) -> None:
        world = ToyBackend().reset()["observations"]["blue"]
        world.pop("active_operation")
        output = self.policy.network(world)
        distribution, _ = self.policy._distributions(world, output)
        self.assertEqual(torch.isfinite(distribution.logits).tolist(),
                         [True] + [False] * (len(ACTIONS) - 1))
        world["active_operation"] = {"status": "unknown"}
        self.assertEqual(self.policy.decide(world)["action"], "IDLE")

    def test_manipulation_needs_no_target_but_navigation_does(self) -> None:
        world = ToyBackend().reset()["observations"]["blue"]
        world["robot"]["payload_state"] = None
        world["cartons"] = []
        world["zones"] = []
        output = self.policy.network(world)
        distribution, _ = self.policy._distributions(world, output)
        self.assertEqual(torch.isfinite(distribution.logits).tolist(),
                         [True, False] + [True] * (len(ACTIONS) - 2))
        world["active_operation"] = {"type": "SUCTION_1_ON", "status": "running", "cancelable": False}
        self.assertEqual(self.policy.decide(world)["action"], "IDLE")

    def test_encoder_is_permutation_invariant_with_empty_sets_and_unknown_pose(self) -> None:
        encoder = WorldEncoder()
        world = ToyBackend().reset()["observations"]["blue"]
        world["cartons"].append({**copy.deepcopy(world["cartons"][0]), "id": "second"})
        world["cartons"][1]["pose"]["x"] = 1.5
        world["obstacles"] = [
            {"id": "one", "pose": {"x": 1.0, "y": 1.0, "yaw": 0.2}, "radius": 0.1},
            {"id": "two", "pose": {"x": 2.0, "y": 1.0, "yaw": -0.2}, "radius": 0.2},
        ]
        self.assertAlmostEqual(encoder.feature_dicts(world)["cartons"][0]["pose_x"], 1.0 / 3.0)
        with torch.no_grad():
            expected = encoder(world)["context"]
            permuted = copy.deepcopy(world)
            permuted["cartons"].reverse()
            permuted["obstacles"].reverse()
            actual = encoder(permuted)["context"]
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))
            empty = copy.deepcopy(world)
            empty["cartons"] = []
            empty["obstacles"] = []
            self.assertTrue(torch.isfinite(encoder(empty)["context"]).all())
            unknown_pose = copy.deepcopy(world)
            unknown_pose["robot"]["pose"] = None
            features = encoder.feature_dicts(unknown_pose)
            self.assertEqual(features["robot"]["position_x_known"], 0.0)
            output = self.policy.network(unknown_pose)
            distribution, _ = self.policy._distributions(unknown_pose, output)
            self.assertFalse(torch.isfinite(distribution.logits[ACTIONS.index("MOVE")]))

    def test_training_samples_are_balanced_and_length_mirrored_with_angles(self) -> None:
        observation = ToyBackend().reset()["observations"]["blue"]
        observation["robot"]["pose"] = {"x": 0.25, "y": 0.5, "yaw": 0.4}
        observation["robot"]["velocity"] = {"x": 0.3, "y": -0.2, "yaw": 0.5}
        observation["cartons"][0]["pose"] = {"x": 1.0, "y": 0.5, "yaw": 0.25}
        observation["obstacles"] = [
            {"pose": {"x": 2.0, "y": 1.0, "yaw": -0.2}, "radius": 0.2}
        ]
        observation["active_operation"] = {
            "target_pose": {"x": 2.5, "y": 1.0, "sin": math.sin(0.6), "cos": math.cos(0.6)}
        }
        move_sample = {
            "chosen_action": "MOVE",
            "target_action_probs": one_hot_action("MOVE"),
            "target_move_pose": {
                "x": 2.25, "y": 0.75, "sin": math.sin(0.7), "cos": math.cos(0.7),
            },
            "move_target_valid": True,
            "observation": observation,
        }
        samples = [
            {**copy.deepcopy(move_sample), "chosen_action": "IDLE",
             "target_action_probs": one_hot_action("IDLE"), "move_target_valid": False,
             "target_move_pose": None}
            for _ in range(3)
        ] + [move_sample]

        balanced = balance_action_samples(samples, seed=7)
        self.assertEqual(len(balanced), 6)
        self.assertEqual([sum(sample["chosen_action"] == action for sample in balanced)
                          for action in ("IDLE", "MOVE")], [3, 3])
        augmented = augment_length_symmetry(balanced)
        self.assertEqual(len(augmented), 2 * len(balanced))

        original = copy.deepcopy(move_sample)
        mirrored = mirror_sample_along_length(move_sample)
        mirror_world = mirrored["observation"]
        self.assertAlmostEqual(mirror_world["robot"]["pose"]["x"], 2.75)
        self.assertAlmostEqual(mirror_world["robot"]["pose"]["yaw"], math.pi - 0.4)
        self.assertAlmostEqual(mirror_world["robot"]["velocity"]["x"], -0.3)
        self.assertAlmostEqual(mirror_world["robot"]["velocity"]["yaw"], -0.5)
        self.assertAlmostEqual(mirror_world["cartons"][0]["pose"]["x"], 2.0)
        self.assertAlmostEqual(mirror_world["cartons"][0]["pose"]["yaw"], math.pi - 0.25)
        self.assertAlmostEqual(mirror_world["obstacles"][0]["pose"]["yaw"], math.pi + 0.2)
        active_target = mirror_world["active_operation"]["target_pose"]
        self.assertAlmostEqual(active_target["x"], 0.5)
        self.assertAlmostEqual(active_target["sin"], math.sin(0.6))
        self.assertAlmostEqual(active_target["cos"], -math.cos(0.6))
        self.assertAlmostEqual(mirrored["target_move_pose"]["x"], 0.75)
        self.assertAlmostEqual(mirrored["target_move_pose"]["sin"], math.sin(0.7))
        self.assertAlmostEqual(mirrored["target_move_pose"]["cos"], -math.cos(0.7))
        self.assertEqual(mirrored["chosen_action"], "MOVE")
        self.assertEqual(move_sample, original)

    def test_executor_validates_fresh_pose_and_wrist_without_starting_ros(self) -> None:
        executor = ActionExecutor()
        world = ToyBackend().reset()["observations"]["blue"]
        move = {"action": "MOVE", "target_pose": {"x": 1.0, "y": 1.0, "sin": 0.0, "cos": 2.0}}
        executor._validate_action("MOVE", move, world)
        self.assertEqual(move["target_pose"]["cos"], 2.0)
        stale = copy.deepcopy(world)
        stale["robot"]["pose_age_s"] = 2.0
        with self.assertRaisesRegex(ValueError, "fresh robot pose"):
            executor._validate_action("MOVE", move, stale)
        unknown_wrist = copy.deepcopy(world)
        unknown_wrist["robot"]["wrist_angle_rad"] = None
        with self.assertRaisesRegex(ValueError, "fresh measured wrist angle"):
            executor._validate_action("TOGGLE_WRIST", {"action": "TOGGLE_WRIST"}, unknown_wrist)
        stationary_move = {"x": world["robot"]["pose"]["x"],
                          "y": world["robot"]["pose"]["y"], "sin": 0.0, "cos": 1.0}
        with self.assertRaisesRegex(ValueError, "too close to the current position"):
            executor._validate_action("MOVE", {"target_pose": stationary_move}, world)

    def test_late_remote_result_resolves_unknown_lock(self) -> None:
        class Future:
            @staticmethod
            def done():
                return True

            @staticmethod
            def result():
                return SimpleNamespace(success=True, message="confirmed")

        executor = ActionExecutor()
        executor.active = {
            "id": "operation-1", "type": "SUCTION_1_ON", "status": "unknown",
            "cancelable": False, "target_pose": None, "started_sequence": 1,
            "terminal_kind": "service", "terminal_future": Future(),
        }
        response = asyncio.run(executor.dispatch(
            {"action": "SUCTION_1_OFF", "world_sequence": 2},
            {"sequence": 2, "active_operation": None},
        ))
        self.assertEqual(response["status"], "rejected")
        asyncio.run(executor.update({"sequence": 3}))
        self.assertIsNone(executor.active)
        self.assertEqual(executor.last_result["status"], "succeeded")

    def test_actuator_service_receives_operation_and_tracks_remote_future(self) -> None:
        class FakeSetBool:
            class Request:
                def __init__(self) -> None:
                    self.data = False

        class FakeClient:
            def __init__(self) -> None:
                self.request = None
                self.future = object()

            @staticmethod
            def service_is_ready() -> bool:
                return True

            def call_async(self, request):
                self.request = request
                return self.future

        std_srvs = ModuleType("std_srvs")
        srv = ModuleType("std_srvs.srv")
        srv.SetBool = FakeSetBool
        std_srvs.srv = srv
        client = FakeClient()
        executor = ActionExecutor()
        executor._wrist_client = client
        operation = {"type": "TOGGLE_WRIST"}

        async def wait_for_response(future, timeout_s):
            self.assertIs(future, client.future)
            self.assertGreater(timeout_s, 0.0)
            return SimpleNamespace(success=True, message="wrist command accepted")

        with patch.dict(sys.modules, {"std_srvs": std_srvs, "std_srvs.srv": srv}), patch(
            "agent.architecture.action_executor._wait_ros_future", new=wait_for_response
        ):
            result = asyncio.run(executor._run_operation(
                operation, {"action": "TOGGLE_WRIST"},
                {"robot": {"wrist_angle_rad": 0.0}},
            ))

        self.assertTrue(client.request.data)
        self.assertIs(operation["terminal_future"], client.future)
        self.assertEqual(result["status"], "succeeded")

    def test_group_suction_calls_the_aggregate_service(self) -> None:
        class FakeSetBool:
            class Request:
                def __init__(self) -> None:
                    self.data = False

        class FakeClient:
            request = None
            future = object()

            @staticmethod
            def service_is_ready() -> bool:
                return True

            def call_async(self, request):
                self.request = request
                return self.future

        std_srvs = ModuleType("std_srvs")
        srv = ModuleType("std_srvs.srv")
        srv.SetBool = FakeSetBool
        std_srvs.srv = srv
        client = FakeClient()
        executor = ActionExecutor()
        executor._suction_all_client = client

        async def wait_for_response(future, timeout_s):
            self.assertIs(future, client.future)
            self.assertGreater(timeout_s, 0.0)
            return SimpleNamespace(success=True, message="group suction accepted")

        with patch.dict(sys.modules, {"std_srvs": std_srvs, "std_srvs.srv": srv}), patch(
            "agent.architecture.action_executor._wait_ros_future", new=wait_for_response
        ):
            result = asyncio.run(executor._run_operation(
                {"type": "SUCTION_ALL_ON"}, {"action": "SUCTION_ALL_ON"}, {}
            ))

        self.assertTrue(client.request.data)
        self.assertEqual(result["status"], "succeeded")

    def test_positive_move_log_density_and_permissive_v1_checkpoint_load(self) -> None:
        world = ToyBackend().reset()["observations"]["blue"]
        with torch.no_grad():
            self.policy.network.action_head[-1].bias.fill_(-30)
            self.policy.network.action_head[-1].bias[ACTIONS.index("MOVE")] = 30
            self.policy.network.move_log_std.fill_(-5)
        decision = self.policy.decide(world)
        log_prob, _, _ = self.policy.evaluate_action(world, decision)
        self.assertGreater(float(log_prob), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.pt"
            old_state = {
                name: value.detach().clone()
                for name, value in self.policy.network.state_dict().items()
            }
            old_state["action_head.2.weight"] = old_state["action_head.2.weight"][:10].clone()
            old_state["action_head.2.bias"] = old_state["action_head.2.bias"][:10].clone()
            old_encoder = WorldEncoder(operation_types=LEGACY_OPERATION_TYPES)
            old_names = old_encoder.feature_names["context"]
            new_names = self.policy.network.encoder.feature_names["context"]
            old_context_weights = {}
            for key, prefix_size in (
                ("encoder.attention_query.0.weight", 32),
                ("encoder.context_mlp.0.weight", 96),
            ):
                current = old_state[key]
                old_context_weights[key] = torch.cat((
                    current[:, :prefix_size],
                    current[:, [prefix_size + new_names.index(name) for name in old_names]],
                ), dim=1)
                old_state[key] = old_context_weights[key].clone()
            torch.save({"format": "arcl-agent-policy-v1",
                        "state_dict": old_state}, legacy)
            loaded = AgentPolicy()
            loaded.load_checkpoint(legacy)
            loaded_state = loaded.network.state_dict()
            self.assertTrue(torch.equal(
                old_state["action_head.2.weight"], loaded_state["action_head.2.weight"][:10]
            ))
            self.assertTrue(torch.equal(
                old_state["action_head.2.bias"], loaded_state["action_head.2.bias"][:10]
            ))
            for key, prefix_size in (
                ("encoder.attention_query.0.weight", 32),
                ("encoder.context_mlp.0.weight", 96),
            ):
                for old_index, name in enumerate(old_names):
                    self.assertTrue(torch.equal(
                        old_context_weights[key][:, prefix_size + old_index],
                        loaded_state[key][:, prefix_size + new_names.index(name)],
                    ))

    def test_six_step_flow_relabels_all_records_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode"
            backend, info = _collect(self.policy, path, plan=True)
            self.assertEqual([step["blue"]["action"] for step in backend.actions],
                             ["MOVE", "TOGGLE_WRIST", "SUCTION_1_ON", "MOVE", "SUCTION_1_OFF", "IDLE"])
            self.assertTrue(0.2 <= backend.actions[0]["blue"]["target_pose"]["x"] <= 2.8)
            rows = [json.loads(line) for line in next(path.glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(all(row["winner_targets"] == {"blue": 1, "red": 0} for row in rows))
            self.assertTrue(all(row["complete"] and row["rewards_complete"] for row in rows))
            transitions = [row for row in rows if row["kind"] == "transition"]
            self.assertEqual(len(transitions), 6)
            expected = sum(0.1 * 0.9 ** step for step in range(6)) + 0.9 ** 5
            self.assertAlmostEqual(transitions[-1]["reward"]["blue"], 1.1)
            for row in transitions:
                self.assertNotIn("secret_truth", row["agent_observations"]["blue"])
                self.assertIn("secret_truth", row["world"])
            self.assertEqual(transitions[0]["actions"]["blue"]["target_pose"]["x"],
                             transitions[0]["backend_actions"]["blue"]["target_pose"]["x"])
            self.assertFalse(info["rewards_pending"])
            self.assertEqual(len(transitions[0]["actions"]["blue"]["behavior"]["action_probs"]),
                             len(ACTIONS))
            prepared = Path(directory) / "targets.jsonl"
            self.assertEqual(prepare_targets([path], prepared), 12)
            self.assertEqual(prepare_targets([path], prepared), 0)
            samples = [json.loads(line) for line in prepared.read_text().splitlines()]
            self.assertEqual(len(samples), 12)
            self.assertTrue(all(sample["format"] == "arcl-agent-targets-v1" for sample in samples))
            self.assertTrue(all(not sample["move_target_valid"] for sample in samples))
            self.assertTrue(all(math.isclose(sum(sample["target_action_probs"]), 1.0)
                                for sample in samples))
            first_blue = next(sample for sample in samples
                              if sample["metadata"]["robot_id"] == "blue"
                              and sample["metadata"]["step"] == 1)
            self.assertAlmostEqual(first_blue["metadata"]["discounted_return"], expected)

    def test_pending_reward_functions_do_not_create_fake_training_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes"
            _, info = _collect(self.policy, path, rewarded=False)
            self.assertTrue(info["rewards_pending"])
            rows = [json.loads(line) for line in next(path.glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(all(row["winner_targets"] is not None for row in rows))
            self.assertTrue(all("return_targets" not in row for row in rows))
            with self.assertRaisesRegex(ValueError, "rewards are incomplete"):
                prepare_targets([path], Path(directory) / "targets.jsonl")

    def test_interrupted_episode_is_saved_and_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ToyBackend()
            env = TwoRobotArenaEnv(backend.advance, controlled_robot_id="blue", opponent_robot_id="red", records_path=directory)
            env.reset(backend.reset())
            env.step("IDLE", "IDLE")
            env.close()
            path = next(Path(directory).glob("*.jsonl"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(all(not row["complete"] and row["winner_targets"] is None for row in rows))
            env.close()
            self.assertEqual(path.read_text().count('"kind":"episode_end"'), 1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                prepare_targets([path], Path(directory) / "targets.jsonl")

    def test_publication_error_never_exposes_a_partial_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = EpisodeRecords(directory, "example", ("blue", "red"))
            records.add({"kind": "decision", "actions": {"blue": {"action": "IDLE"}}})
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

    def test_selfplay_soft_target_trains_action_head_without_pose_targets(self) -> None:
        backend = ToyBackend()
        world = backend.reset()["observations"]["blue"]
        decision = self.policy.decide(world, deterministic=False)
        probabilities = decision["behavior"]["action_probs"]
        target = selfplay_target_distribution(probabilities, decision["action"], 1.0)
        sample = {"observation": world, "target_action_probs": target,
                  "target_move_pose": None, "move_target_valid": False}
        loss, _ = target_loss(self.policy, sample, torch.ones(len(ACTIONS)))
        loss.backward()
        self.assertGreater(float(self.policy.network.action_head[-1].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.policy.network.move_mean_head[-1].weight.grad)

    def test_target_group_split_is_reproducible_and_keeps_episode_robots_together(self) -> None:
        episodes = {str(index): [{"robot": "blue"}, {"robot": "red"}] for index in range(10)}
        train_set, val_set = split_groups(episodes, 0.2, 7)
        self.assertFalse(set(train_set) & set(val_set))
        self.assertEqual(len(val_set), 2)
        self.assertEqual((train_set, val_set), split_groups(episodes, 0.2, 7))
        self.assertTrue(all(len(samples) == 2 for samples in (*train_set.values(), *val_set.values())))

    def test_mixed_sources_train_and_reload_with_held_out_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, output = root / "collection.pt", root / "trained.pt"
            self.policy.save_checkpoint(model)
            for index in range(3):
                _collect(self.policy, root / "raw" / str(index))
            _collect(self.policy, root / "held-out-raw")
            combined_targets = root / "targets.jsonl"
            heldout_targets = root / "heldout.jsonl"
            prepare_targets([root / "raw"], combined_targets)
            prepare_targets([root / "held-out-raw"], heldout_targets)
            operator = DemonstrationRecorder(
                combined_targets, session_id="operator-session"
            )
            operator.record_decision(
                ToyBackend().reset()["observations"]["blue"], {"action": "IDLE"},
                command="idle", requires_result=False,
            )
            operator.stop()
            with self.assertRaisesRegex(ValueError, "canonical"):
                read_target_datasets([root / "raw"])
            args = argparse.Namespace(datasets=[combined_targets], model=model,
                                      output=output, validation_data=None,
                                      test_data=[heldout_targets], validation_fraction=0.25,
                                      seed=7, device="cpu", position_weight=1.0,
                                      orientation_weight=1.0, learning_rate=0.001,
                                      epochs=2, batch_size=8)
            report = train(args)
            self.assertEqual(len(report["train_group_ids"]), 3)
            self.assertEqual(len(report["validation_group_ids"]), 1)
            self.assertFalse(set(report["train_group_ids"]) & set(report["validation_group_ids"]))
            self.assertEqual(set(report["source_counts"]), {"operator", "selfplay"})
            self.assertTrue(output.with_suffix(".metrics.json").exists())
            self.assertIn("expected_action_accuracy", report["test"])
            self.assertFalse(set(report["test_group_ids"]) &
                             (set(report["train_group_ids"]) | set(report["validation_group_ids"])))
            loaded = AgentPolicy()
            loaded.load_checkpoint(output)
            decision = loaded.decide(ToyBackend().reset()["observations"]["blue"])
            self.assertIn(decision["action"], ACTIONS)
            args.validation_data = [combined_targets]
            with self.assertRaisesRegex(ValueError, "disjoint"):
                train(args)

    def test_selfplay_preparer_accepts_saved_distributions_and_rejects_duplicate_episodes(self) -> None:
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
            targets = root / "targets.jsonl"
            self.assertEqual(prepare_targets([deterministic], targets), 12)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(original.read_text())
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                read_selfplay_episodes([original, duplicate])

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
            env.step("IDLE", "IDLE")
            _, _, terminated, truncated, info = env.step("IDLE", "IDLE")
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["winner_known"])
            self.assertTrue(info["rewards_pending"])
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(all(row["winner_targets"] is None for row in rows))
            self.assertTrue(all("return_targets" not in row for row in rows))

    def test_backend_failure_closes_episode_and_blocks_retry_on_stale_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ToyBackend()
            def failing_advance(actions, *, max_game_seconds):
                raise ValueError("test backend failed")
            env = TwoRobotArenaEnv(failing_advance, controlled_robot_id="blue", opponent_robot_id="red", records_path=directory)
            env.reset(backend.reset())
            with self.assertRaisesRegex(ValueError, "test backend failed"):
                env.step("IDLE", "IDLE")
            self.assertTrue(env.done)
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            self.assertTrue(any(row["kind"] == "decision" for row in rows))
            self.assertTrue(all(not row["complete"] for row in rows))
            with self.assertRaises(RuntimeError):
                env.step("IDLE", "IDLE")

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
