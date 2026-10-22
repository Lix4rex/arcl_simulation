"""Two-robot RL wrapper with separate simulator truth and agent observations.

The simulator backend advances a joint action and returns ``world`` for scoring
and logs, plus an ``observations`` mapping keyed by robot ID. Policies receive
only their own observation.
"""

from __future__ import annotations

import copy
import math
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ..architecture.policy import ACTIONS
from ..records import EpisodeRecords


class SimulatorStep(Protocol):
    def __call__(
        self,
        actions: dict[str, dict[str, Any]],
        *,
        max_game_seconds: float,
    ) -> dict[str, Any]: ...


RewardFunction = Callable[..., Mapping[str, float]]


def assign_action_points(
    previous_world: dict[str, Any],
    actions: dict[str, dict[str, Any]],
    responses: dict[str, Any],
    next_world: dict[str, Any],
    robot_ids: tuple[str, str],
) -> Mapping[str, float]:
    """TODO: assign per-action points for both robots."""
    raise NotImplementedError("define the per-action reward before RL training")


def assign_winner_points(
    winner_id: str | None,
    robot_ids: tuple[str, str],
) -> Mapping[str, float]:
    """TODO: assign terminal points after comparing the two robots."""
    raise NotImplementedError("define win, loss and draw points before RL training")


def winner_targets(
    winner_id: str | None,
    robot_ids: tuple[str, str],
    *,
    result_known: bool,
) -> dict[str, float] | None:
    """Return terminal value targets: win=1, loss=0, draw=0.5."""
    if not result_known:
        return None
    if winner_id is None:
        return {robot_id: 0.5 for robot_id in robot_ids}
    if winner_id not in robot_ids:
        raise ValueError(f"winner {winner_id!r} is not one of the two robots")
    return {
        robot_id: 1.0 if robot_id == winner_id else 0.0
        for robot_id in robot_ids
    }


def winner_from_world(
    world: dict[str, Any], robot_ids: tuple[str, str]
) -> tuple[str | None, bool]:
    """Read an explicit winner or compare the final scores of both robots."""
    match = world.get("match", {})
    if not isinstance(match, dict):
        match = {}
    explicit = match.get("winner_id", world.get("winner_id"))
    if explicit in robot_ids:
        return str(explicit), True
    if explicit in ("draw", "tie"):
        return None, True

    scores = _score_map(world, robot_ids)
    if scores is None:
        return None, False
    first, second = robot_ids
    if scores[first] == scores[second]:
        return None, True
    return (first if scores[first] > scores[second] else second), True


class TwoRobotArenaEnv:
    """Gym-like joint-action wrapper with a 90-second game-time horizon.

    `advance` receives both decisions and the remaining time budget. Its result
    must contain the simulator-truth `world`, per-robot `observations`, and
    optional per-robot `responses`. Truth is used for match results, rewards
    and records; policy observations remain in the absolute map frame.
    """

    def __init__(
        self,
        advance: SimulatorStep,
        *,
        controlled_robot_id: str,
        opponent_robot_id: str,
        episode_seconds: float = 90.0,
        robot_margin_m: float = 0.20,
        records_path: str | Path | None = "runs/episodes",
        gamma: float = 0.99,
        action_rewarder: RewardFunction | None = None,
        terminal_rewarder: RewardFunction | None = None,
    ) -> None:
        if controlled_robot_id == opponent_robot_id:
            raise ValueError("the two robot IDs must be different")
        if not math.isfinite(episode_seconds) or episode_seconds <= 0.0:
            raise ValueError("episode_seconds must be positive")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if not math.isfinite(robot_margin_m) or robot_margin_m < 0.0:
            raise ValueError("robot_margin_m must be finite and non-negative")
        self.advance = advance
        self.robot_ids = (controlled_robot_id, opponent_robot_id)
        self.episode_seconds = float(episode_seconds)
        self.robot_margin_m = float(robot_margin_m)
        self.records_path = Path(records_path) if records_path is not None else None
        self.gamma = gamma
        self.records: EpisodeRecords | None = None
        self.action_rewarder = action_rewarder
        self.terminal_rewarder = terminal_rewarder
        self.episode_id: str | None = None
        self.world: dict[str, Any] | None = None
        self.observations: dict[str, dict[str, Any]] | None = None
        self.started_at: float | None = None
        self.elapsed_s = 0.0
        self.done = False
        self.steps = 0

    def reset(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        initial_world, observations = _state_pair(initial_state, self.robot_ids)
        available_ids = {str(robot["id"]) for robot in _robot_entries(initial_world)}
        missing_ids = set(self.robot_ids) - available_ids
        if missing_ids:
            raise ValueError(f"world is missing robot IDs: {', '.join(sorted(missing_ids))}")
        self.close(reason="reset before episode completion")
        self.episode_id = str(uuid.uuid4())
        self.records = EpisodeRecords(self.records_path, self.episode_id, self.robot_ids, gamma=self.gamma)
        self.world = copy.deepcopy(initial_world)
        self.observations = copy.deepcopy(observations)
        self.started_at = _game_time(initial_world)
        self.elapsed_s = 0.0
        self.done = False
        self.steps = 0
        policy_observations = self.agent_observations
        observation = policy_observations[self.robot_ids[0]]
        self._record({
            "kind": "reset",
            "episode_id": self.episode_id,
            "robot_ids": self.robot_ids,
            "game_time_s": self.started_at,
            "world": initial_world,
            "observations": observations,
            "agent_observations": policy_observations,
            "observation": observation,
        })
        return observation

    def step(
        self,
        controlled_action: str | dict[str, Any],
        opponent_action: str | dict[str, Any] = "IDLE",
    ) -> tuple[dict[str, Any], dict[str, float] | None, bool, bool, dict[str, Any]]:
        try:
            return self._step(controlled_action, opponent_action)
        except Exception:
            self.close(reason="step failed; episode cannot be used for training")
            raise

    def _step(
        self,
        controlled_action: str | dict[str, Any],
        opponent_action: str | dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float] | None, bool, bool, dict[str, Any]]:
        if self.world is None or self.started_at is None:
            raise RuntimeError("call reset({world, observations}) before step")
        if self.done:
            raise RuntimeError("episode is finished; call reset before another step")

        actions = {
            self.robot_ids[0]: _action(controlled_action),
            self.robot_ids[1]: _action(opponent_action),
        }
        policy_observations_before = self.agent_observations
        backend_actions = {
            robot_id: _backend_action(actions[robot_id], self.observations[robot_id], self.robot_margin_m)
            for robot_id in self.robot_ids
        }
        self._record({"kind": "decision", "step": self.steps + 1,
                      "agent_observations": policy_observations_before,
                      "actions": actions, "backend_actions": backend_actions})
        previous_world = self.world
        remaining = max(0.0, self.episode_seconds - self.elapsed_s)
        result = self.advance(backend_actions, max_game_seconds=remaining)
        if not isinstance(result, dict) or not isinstance(result.get("world"), dict):
            raise ValueError("simulator advance must return a dict containing world")
        next_world = copy.deepcopy(result["world"])
        _, next_observations = _state_pair(
            {"world": next_world, "observations": result.get("observations")}, self.robot_ids
        )
        next_observations = copy.deepcopy(next_observations)
        responses = result.get("responses", {})
        if not isinstance(responses, dict):
            raise ValueError("simulator responses must be keyed by robot ID")

        game_time = _game_time(next_world)
        previous_time = _game_time(previous_world)
        if game_time < previous_time:
            raise ValueError("simulator game time moved backwards")
        if game_time - previous_time > remaining + 1e-6:
            raise ValueError("simulator advance exceeded max_game_seconds")
        self.elapsed_s = min(self.episode_seconds, max(0.0, game_time - self.started_at))
        self.steps += 1
        match = next_world.get("match", {})
        match_finished = isinstance(match, dict) and match.get("finished") is True
        match_finished = match_finished or next_world.get("match_finished") is True
        if game_time == previous_time and not match_finished:
            raise ValueError("simulator advance must advance game time or finish the match")
        time_limit_reached = game_time - self.started_at >= self.episode_seconds
        terminated = match_finished
        truncated = time_limit_reached and not match_finished
        self.done = terminated or truncated

        policy_observations = {
            robot_id: self._policy_observation(next_observations[robot_id], robot_id)
            for robot_id in self.robot_ids
        }
        observation = policy_observations[self.robot_ids[0]]
        winner_id, result_known = winner_from_world(next_world, self.robot_ids) if self.done else (None, False)
        targets = winner_targets(winner_id, self.robot_ids, result_known=result_known)
        rewards: dict[str, float] | None = None
        if self.action_rewarder is not None:
            rewards = _normalize_rewards(
                self.action_rewarder(previous_world, backend_actions, responses, next_world, self.robot_ids),
                self.robot_ids,
            )
        if self.done and result_known and self.terminal_rewarder is not None:
            terminal_rewards = _normalize_rewards(
                self.terminal_rewarder(winner_id, self.robot_ids), self.robot_ids
            )
            rewards = _add_rewards(rewards, terminal_rewards, self.robot_ids)

        rewards_complete = self.action_rewarder is not None and self.terminal_rewarder is not None and result_known
        info = {
            "episode_id": self.episode_id,
            "step": self.steps,
            "elapsed_game_s": self.elapsed_s,
            "responses": copy.deepcopy(responses),
            "agent_observations": policy_observations,
            "winner_id": winner_id,
            "winner_known": result_known,
            "winner_targets": targets,
            "rewards_pending": self.action_rewarder is None or (self.done and not rewards_complete),
        }
        self._record({
            "kind": "transition",
            "episode_id": self.episode_id,
            "step": self.steps,
            "game_time_s": game_time,
            "elapsed_game_s": self.elapsed_s,
            "state": self._policy_observation(
                self.observations[self.robot_ids[0]], self.robot_ids[0]
            ),
            "world": previous_world,
            "observations": self.observations,
            "agent_observations": policy_observations_before,
            "actions": actions,
            "backend_actions": backend_actions,
            "responses": responses,
            "reward": rewards,
            "next_state": observation,
            "next_agent_observations": policy_observations,
            "next_world": next_world,
            "next_observations": next_observations,
            "terminated": terminated,
            "truncated": truncated,
            "winner_id": winner_id,
            "winner_targets": targets,
        })
        if self.done and self.records is not None:
            self.records.finish(complete=True, winner_id=winner_id, winner_targets=targets,
                                rewards_complete=rewards_complete)
            info["records_path"] = str(self.records.path) if self.records.path else None
        self.world = next_world
        self.observations = next_observations
        return observation, rewards, terminated, truncated, info

    @property
    def agent_observations(self) -> dict[str, dict[str, Any]]:
        """Return the versioned map-frame observation for each policy."""
        if self.observations is None:
            raise RuntimeError("call reset({world, observations}) before reading observations")
        return {
            robot_id: self._policy_observation(self.observations[robot_id], robot_id)
            for robot_id in self.robot_ids
        }

    def _policy_observation(self, observation: dict[str, Any], robot_id: str) -> dict[str, Any]:
        result = copy.deepcopy(observation)
        if result.get("schema_version") != 2:
            raise ValueError("backend observations must use schema_version 2 map-frame inputs")
        if str(result.get("robot", {}).get("id")) != robot_id:
            raise ValueError(f"observation robot ID does not match {robot_id!r}")
        if not isinstance(result.get("mission"), dict):
            result["mission"] = {}
        result["mission"]["total_s"] = self.episode_seconds
        return result

    def _record(self, item: dict[str, Any]) -> None:
        if self.records is not None:
            self.records.add(item)

    def close(self, *, reason: str = "episode interrupted") -> None:
        if self.records is not None and not self.records.finished:
            self.records.finish(complete=False, winner_id=None, winner_targets=None,
                                rewards_complete=False, reason=reason)
        self.done = True


def _robot_entries(world: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(world.get("robots"), list):
        robots = world["robots"]
    else:
        robots = []
        if isinstance(world.get("robot"), dict):
            robots.append(world["robot"])
        others = world.get("other_robots", [])
        if isinstance(others, list):
            robots.extend(others)
    if any(not isinstance(robot, dict) or robot.get("id") is None for robot in robots):
        raise ValueError("each robot needs a stable id for a two-robot episode")
    ids = [str(robot["id"]) for robot in robots]
    if len(ids) != len(set(ids)):
        raise ValueError("robot IDs must be unique in a world snapshot")
    return robots


def _state_pair(
    state: Any, robot_ids: tuple[str, str]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate a backend result without deriving observations from truth."""
    if not isinstance(state, dict) or not isinstance(state.get("world"), dict):
        raise ValueError("backend state must contain a simulator-truth world object")
    truth_ids = {str(robot["id"]) for robot in _robot_entries(state["world"])}
    missing_truth = set(robot_ids) - truth_ids
    if missing_truth:
        raise ValueError(
            "simulator truth is missing robot IDs: " + ", ".join(sorted(missing_truth))
        )
    observations = state.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("backend state must provide observations keyed by robot ID")
    missing = [robot_id for robot_id in robot_ids if not isinstance(observations.get(robot_id), dict)]
    if missing:
        raise ValueError(
            "backend observations are missing robot IDs: " + ", ".join(missing)
        )
    for robot_id in robot_ids:
        observation_ids = {str(robot["id"]) for robot in _robot_entries(observations[robot_id])}
        if robot_id not in observation_ids:
            raise ValueError(f"observation for {robot_id!r} does not contain that robot")
    return state["world"], {
        robot_id: observations[robot_id] for robot_id in robot_ids
    }


def _game_time(world: dict[str, Any]) -> float:
    value = _number(world.get("game_time_s", world.get("timestamp")), math.nan)
    if not math.isfinite(value):
        raise ValueError("world needs a finite game_time_s or timestamp")
    return value


def _score_map(world: dict[str, Any], robot_ids: tuple[str, str]) -> dict[str, float] | None:
    match = world.get("match", {})
    if not isinstance(match, dict):
        match = {}
    scoreboard = world.get("scoreboard", {})
    if not isinstance(scoreboard, dict):
        scoreboard = {}
    candidates = (match.get("scores"), scoreboard.get("scores"), world.get("scores"))
    for candidate in candidates:
        if isinstance(candidate, dict) and all(robot_id in candidate for robot_id in robot_ids):
            scores = {robot_id: _number(candidate[robot_id], math.nan) for robot_id in robot_ids}
            if all(math.isfinite(value) for value in scores.values()):
                return scores
    try:
        robots = _robot_entries(world)
    except ValueError:
        return None
    score_by_id = {
        str(robot["id"]): _number(robot.get("score", robot.get("points")), math.nan)
        for robot in robots
    }
    if all(robot_id in score_by_id and math.isfinite(score_by_id[robot_id]) for robot_id in robot_ids):
        return {robot_id: score_by_id[robot_id] for robot_id in robot_ids}
    return None


def _backend_action(action: dict[str, Any], observation: dict[str, Any], robot_margin_m: float = 0.20) -> dict[str, Any]:
    result = copy.deepcopy(action)
    if "world_sequence" in result and result["world_sequence"] != observation.get("sequence"):
        raise ValueError("decision is based on a stale observation sequence")
    active = observation.get("active_operation")
    if "active_operation" not in observation:
        if result["action"] != "IDLE":
            raise ValueError("active-operation state is unknown; only IDLE is valid")
    elif isinstance(active, dict) and active.get("status") in ("issuing", "running", "cancel_requested", "canceling", "unknown"):
        if active.get("type") == "MOVE" and active.get("cancelable") is True and active.get("status") in ("issuing", "running"):
            if result["action"] not in ("IDLE", "CANCEL"):
                raise ValueError("only IDLE or CANCEL is valid during an active move")
        elif result["action"] != "IDLE":
            raise ValueError("only IDLE is valid during a non-cancelable operation")
    elif active is not None and (not isinstance(active, dict) or active.get("status") not in (
        "succeeded", "failed", "canceled", "rejected"
    )):
        if result["action"] != "IDLE":
            raise ValueError("active-operation state is unknown; only IDLE is valid")
    elif result["action"] == "CANCEL":
        raise ValueError("CANCEL requires an active cancelable move")
    if result["action"] == "MOVE":
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        pose = robot.get("pose")
        sources = observation.get("sources") if isinstance(observation.get("sources"), dict) else {}
        pose_source = sources.get("robot_pose") if isinstance(sources.get("robot_pose"), dict) else {}
        if (not isinstance(pose, dict) or pose_source.get("fresh", True) is not True
                or _number(robot.get("pose_age_s"), math.inf) > 0.75
                or not all(math.isfinite(_number(pose.get(key), math.nan)) for key in ("x", "y", "yaw"))):
            raise ValueError("MOVE requires a fresh robot pose")
        target = result.get("target_pose")
        if not isinstance(target, dict):
            raise ValueError("MOVE requires an absolute target_pose")
        x = _number(target.get("x"), math.nan)
        y = _number(target.get("y"), math.nan)
        sine = _number(target.get("sin"), math.nan)
        cosine = _number(target.get("cos"), math.nan)
        if not all(math.isfinite(value) for value in (x, y, sine, cosine)):
            raise ValueError("MOVE target must contain finite x, y, sin and cos")
        arena = observation.get("arena") if isinstance(observation.get("arena"), dict) else {}
        bounds = arena.get("bounds", arena)
        if not isinstance(bounds, dict):
            raise ValueError("arena bounds are unavailable")
        margin = robot_margin_m
        if not (float(bounds.get("x_min", 0.0)) + margin <= x <= float(bounds.get("x_max", 3.0)) - margin
                and float(bounds.get("y_min", 0.0)) + margin <= y <= float(bounds.get("y_max", 2.0)) - margin):
            raise ValueError("MOVE target lies outside the robot-center arena bounds")
        norm = math.hypot(sine, cosine)
        if norm < 1e-6:
            raise ValueError("MOVE target orientation vector has near-zero norm")
        result["target_pose"] = {"x": x, "y": y, "sin": sine / norm, "cos": cosine / norm}
    elif result["action"] == "TOGGLE_WRIST":
        robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
        wrist = _number(robot.get("wrist_angle_rad"), math.nan)
        sources = observation.get("sources") if isinstance(observation.get("sources"), dict) else {}
        wrist_source = sources.get("wrist") if isinstance(sources.get("wrist"), dict) else {}
        if (not math.isfinite(wrist) or wrist_source.get("fresh", True) is not True
                or _number(wrist_source.get("age_s"), math.inf) > 1.0):
            raise ValueError("TOGGLE_WRIST requires a fresh measured wrist angle")
    elif result["action"] not in ("IDLE", "CANCEL") and not result["action"].startswith("SUCTION_"):
        raise ValueError(f"unsupported action: {result['action']!r}")
    if result["action"] != "MOVE":
        result.pop("target_pose", None)
    return result


def _action(value: str | dict[str, Any]) -> dict[str, Any]:
    result = {"action": value} if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(result, dict) or result.get("action") not in ACTIONS:
        raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
    return result


def _normalize_rewards(values: Mapping[str, float], robot_ids: tuple[str, str]) -> dict[str, float]:
    result = {robot_id: float(values[robot_id]) for robot_id in robot_ids}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("reward values must be finite")
    return result


def _add_rewards(
    first: dict[str, float] | None,
    second: dict[str, float],
    robot_ids: tuple[str, str],
) -> dict[str, float]:
    first = first or {robot_id: 0.0 for robot_id in robot_ids}
    return {robot_id: first[robot_id] + second[robot_id] for robot_id in robot_ids}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


