"""Masked hybrid actor for single-operation robot control."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .world_encoder import CONTEXT_SIZE, WorldEncoder


CHECKPOINT_FORMAT = "arcl-agent-policy-v1"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "checkpoints" / "policy.pt"
ACTIONS = (
    "IDLE", "CANCEL", "MOVE", "SUCTION_1_ON", "SUCTION_2_ON", "SUCTION_3_ON",
    "SUCTION_1_OFF", "SUCTION_2_OFF", "SUCTION_3_OFF", "TOGGLE_WRIST",
)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
DEFAULT_ROBOT_MARGIN_M = 0.20


class PolicyNetwork(nn.Module):
    """Shared observation encoder with categorical, continuous, and value heads."""

    def __init__(self, *, velocity_scale_m_s: float = 1.0,
                 arena_bounds: tuple[float, float, float, float] = (0.0, 3.0, 0.0, 2.0)) -> None:
        super().__init__()
        self.encoder = WorldEncoder(velocity_scale_m_s=velocity_scale_m_s, arena_bounds=arena_bounds)
        self.action_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 64), nn.Tanh(), nn.Linear(64, len(ACTIONS))
        )
        self.move_mean_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 64), nn.Tanh(), nn.Linear(64, 4)
        )
        self.move_log_std = nn.Parameter(torch.full((4,), -0.5))
        self.value_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, world: dict[str, Any]) -> dict[str, Any]:
        context = self.encoder(world)["context"]
        move_mean = self.move_mean_head(context)
        return {
            "action_logits": self.action_head(context),
            "move_mean": move_mean,
            "move_log_std": self.move_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(
                move_mean
            ),
            "value": self.value_head(context).squeeze(-1),
        }


class AgentPolicy:
    """Sample valid intents and absolute map-frame move targets."""

    def __init__(self, *, device: str = "cpu", robot_margin_m: float = DEFAULT_ROBOT_MARGIN_M,
                 velocity_scale_m_s: float = 1.0,
                 arena_bounds: tuple[float, float, float, float] = (0.0, 3.0, 0.0, 2.0)) -> None:
        self.device = torch.device(device)
        self.robot_margin_m = float(robot_margin_m)
        if not math.isfinite(self.robot_margin_m) or self.robot_margin_m < 0:
            raise ValueError("robot_margin_m must be finite and non-negative")
        self.velocity_scale_m_s = float(velocity_scale_m_s)
        self.arena_bounds = tuple(float(value) for value in arena_bounds)
        self.network = PolicyNetwork(velocity_scale_m_s=self.velocity_scale_m_s,
                                     arena_bounds=self.arena_bounds).to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint must contain a policy state dictionary")
        configuration = checkpoint.get("observation_config") if isinstance(checkpoint, dict) else None
        if isinstance(configuration, dict):
            velocity_scale = _finite(configuration.get("velocity_scale_m_s"), math.nan)
            bounds = configuration.get("arena_bounds")
            if (math.isfinite(velocity_scale) and velocity_scale > 0 and isinstance(bounds, list)
                    and len(bounds) == 4
                    and all(math.isfinite(_finite(value, math.nan)) for value in bounds)):
                normalized_bounds = tuple(float(value) for value in bounds)
                if normalized_bounds[0] < normalized_bounds[1] and normalized_bounds[2] < normalized_bounds[3]:
                    self.velocity_scale_m_s = velocity_scale
                    self.arena_bounds = normalized_bounds
                    self.network.encoder.velocity_scale_m_s = velocity_scale
                    self.network.encoder.arena_bounds = normalized_bounds
        self.network.load_state_dict(state_dict, strict=True)
        self.network.eval()

    def load_or_create_checkpoint(self, path: str | Path) -> bool:
        checkpoint_path = Path(path)
        if checkpoint_path.is_file():
            self.load_checkpoint(checkpoint_path)
            return False
        self.save_checkpoint(checkpoint_path)
        return True

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "observation_config": {
                "velocity_scale_m_s": self.velocity_scale_m_s,
                "arena_bounds": list(self.arena_bounds),
            },
            "state_dict": self.network.state_dict(),
        }, checkpoint_path)

    @staticmethod
    def _pose_available(world: dict[str, Any]) -> bool:
        robot = world.get("robot")
        pose = robot.get("pose") if isinstance(robot, dict) else None
        sources = world.get("sources") if isinstance(world.get("sources"), dict) else {}
        source = sources.get("robot_pose") if isinstance(sources.get("robot_pose"), dict) else {}
        age = _finite(robot.get("pose_age_s"), math.inf) if isinstance(robot, dict) else math.inf
        if not isinstance(pose, dict) or source.get("fresh", True) is not True or age > 0.75:
            return False
        return all(math.isfinite(_finite(pose.get(key), math.nan)) for key in ("x", "y", "yaw"))

    @staticmethod
    def _wrist_available(world: dict[str, Any]) -> bool:
        robot = world.get("robot")
        if not isinstance(robot, dict) or not math.isfinite(_finite(robot.get("wrist_angle_rad"), math.nan)):
            return False
        sources = world.get("sources") if isinstance(world.get("sources"), dict) else {}
        source = sources.get("wrist") if isinstance(sources.get("wrist"), dict) else {}
        return source.get("fresh", True) is True and _finite(source.get("age_s"), math.inf) <= 1.0

    @staticmethod
    def _mask(world: dict[str, Any], device: torch.device) -> torch.Tensor:
        mask = torch.zeros(len(ACTIONS), dtype=torch.bool, device=device)
        operation = world.get("active_operation")
        if "active_operation" not in world:
            mask[ACTIONS.index("IDLE")] = True
        elif operation is None:
            mask[:] = True
            mask[ACTIONS.index("CANCEL")] = False
        elif isinstance(operation, dict) and operation.get("status") in ("issuing", "running"):
            mask[ACTIONS.index("IDLE")] = True
            if operation.get("cancelable") is True and operation.get("type") == "MOVE":
                mask[ACTIONS.index("CANCEL")] = True
        elif isinstance(operation, dict) and operation.get("status") in (
            "succeeded", "failed", "canceled", "rejected"
        ):
            mask[:] = True
            mask[ACTIONS.index("CANCEL")] = False
        else:
            mask[ACTIONS.index("IDLE")] = True
        return mask

    def _distributions(self, world: dict[str, Any], output: dict[str, Any]) -> tuple[Categorical, Normal]:
        mask = self._mask(world, self.device)
        if not self._pose_available(world):
            mask[ACTIONS.index("MOVE")] = False
        if not self._wrist_available(world):
            mask[ACTIONS.index("TOGGLE_WRIST")] = False
        if not bool(mask.any()):
            mask[ACTIONS.index("IDLE")] = True
        action = Categorical(logits=output["action_logits"][0].masked_fill(~mask, -torch.inf))
        move = Normal(output["move_mean"][0], output["move_log_std"][0].exp())
        return action, move

    def _move_bounds(self, world: dict[str, Any]) -> tuple[float, float, float, float]:
        arena = world.get("arena") if isinstance(world.get("arena"), dict) else {}
        bounds = arena.get("bounds", arena) if isinstance(arena, dict) else {}
        x_min = _finite(bounds.get("x_min"), 0.0) + self.robot_margin_m
        x_max = _finite(bounds.get("x_max"), 3.0) - self.robot_margin_m
        y_min = _finite(bounds.get("y_min"), 0.0) + self.robot_margin_m
        y_max = _finite(bounds.get("y_max"), 2.0) - self.robot_margin_m
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("arena is too small for the configured robot margin")
        return x_min, x_max, y_min, y_max

    def _decode_move(self, world: dict[str, Any], latent: torch.Tensor) -> dict[str, float]:
        x_min, x_max, y_min, y_max = self._move_bounds(world)
        x = x_min + (x_max - x_min) * torch.sigmoid(latent[0])
        y = y_min + (y_max - y_min) * torch.sigmoid(latent[1])
        direction = latent[2:4]
        norm = torch.linalg.vector_norm(direction)
        if float(norm) < 1e-6:
            robot = world.get("robot", {})
            yaw = _finite(_object(robot.get("pose")).get("yaw"), 0.0)
            sine, cosine = math.sin(yaw), math.cos(yaw)
        else:
            sine, cosine = (direction / norm).tolist()
        yaw = math.atan2(sine, cosine)
        return {"x": float(x), "y": float(y), "sin": float(sine), "cos": float(cosine)}

    def _evaluate_output(self, world: dict[str, Any], decision: dict[str, Any], output: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dist, move_dist = self._distributions(world, output)
        action = decision.get("action")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        action_index = ACTIONS.index(action)
        if not torch.isfinite(action_dist.logits[action_index]).item():
            raise ValueError(f"recorded action {action} is incompatible with its observation")
        log_prob = action_dist.log_prob(torch.tensor(action_index, device=self.device))
        entropy = action_dist.entropy()
        if action == "MOVE":
            latent_values = decision.get("behavior", {}).get("move_latent", decision.get("move_latent"))
            if not isinstance(latent_values, list) or len(latent_values) != 4:
                raise ValueError("MOVE decision must record its four latent Gaussian samples")
            latent = torch.tensor(latent_values, dtype=torch.float32, device=self.device)
            log_prob = log_prob + move_dist.log_prob(latent).sum()
            entropy = entropy + move_dist.entropy().sum()
        return log_prob, entropy, output["value"][0]

    def evaluate_action(self, world: dict[str, Any], decision: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_output(world, decision, self.network(world))

    def decide(self, world: dict[str, Any], *, deterministic: bool = True) -> dict[str, Any]:
        with torch.no_grad():
            output = self.network(world)
            action_dist, move_dist = self._distributions(world, output)
            action_index = int(torch.argmax(action_dist.logits).item()) if deterministic else int(action_dist.sample().item())
            action = ACTIONS[action_index]
            decision: dict[str, Any] = {
                "action": action,
                "world_sequence": int(world.get("sequence", -1)),
                "world_timestamp": _finite(world.get("timestamp")),
            }
            if action == "MOVE":
                latent = move_dist.mean if deterministic else move_dist.sample()
                latent_list = latent.tolist()
                target = self._decode_move(world, latent)
                decision.update({"target_pose": target, "move_latent": latent_list})
            log_prob, _, value = self._evaluate_output(world, decision, output)
            decision["behavior"] = {
                "sampled": not deterministic,
                "log_prob": float(log_prob),
                "value": float(value),
                **({"move_latent": decision.pop("move_latent")} if action == "MOVE" else {}),
            }
        return decision


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
