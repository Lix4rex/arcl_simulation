"""High-level action policy over encoded world snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .world_encoder import CONTEXT_SIZE, EMBEDDING_SIZE, WorldEncoder


CHECKPOINT_FORMAT = "arcl-agent-policy-v1"
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "policy.pt"
)
ACTIONS = (
    "WAIT",
    "GO_TO_CARTON",
    "GO_TO_ZONE",
    "CENTER",
    "TAKE",
    "RELEASE",
)


class PolicyNetwork(nn.Module):
    """Compact actor with shared per-entity target scoring heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = WorldEncoder()
        self.action_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, 48), nn.Tanh(), nn.Linear(48, len(ACTIONS))
        )
        self.carton_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE + EMBEDDING_SIZE, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.zone_head = nn.Sequential(
            nn.Linear(CONTEXT_SIZE + EMBEDDING_SIZE, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.value_head = nn.Sequential(nn.Linear(CONTEXT_SIZE, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, world: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(world)
        context = encoded["context"]
        carton_embeddings = encoded["entities"]["cartons"]
        zone_embeddings = encoded["entities"]["zones"]
        if carton_embeddings.shape[1]:
            carton_context = context.unsqueeze(1).expand(-1, carton_embeddings.shape[1], -1)
            carton_scores = self.carton_head(
                torch.cat([carton_context, carton_embeddings], dim=-1)
            ).squeeze(-1)
        else:
            carton_scores = torch.empty((1, 0), device=context.device)
        if zone_embeddings.shape[1]:
            zone_context = context.unsqueeze(1).expand(-1, zone_embeddings.shape[1], -1)
            zone_scores = self.zone_head(
                torch.cat([zone_context, zone_embeddings], dim=-1)
            ).squeeze(-1)
        else:
            zone_scores = torch.empty((1, 0), device=context.device)
        return {
            "action_logits": self.action_head(context),
            "carton_scores": carton_scores,
            "zone_scores": zone_scores,
            "context": context,
            "value": self.value_head(context).squeeze(-1),
        }


class AgentPolicy:
    """Mask impossible decisions and decode network outputs to stable targets."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.network = PolicyNetwork().to(self.device)

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported agent checkpoint format")
        self.network.load_state_dict(checkpoint["state_dict"], strict=True)
        self.network.eval()

    def load_or_create_checkpoint(self, path: str | Path) -> bool:
        """Load an existing checkpoint, or save the newly initialized network.

        Returns True when a new checkpoint was created.
        """
        checkpoint_path = Path(path)
        if checkpoint_path.is_file():
            self.load_checkpoint(checkpoint_path)
            return False
        self.save_checkpoint(checkpoint_path)
        return True

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the current network parameters."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "state_dict": self.network.state_dict(),
        }, checkpoint_path)

    @staticmethod
    def _wait(world: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "action": "WAIT",
            "world_sequence": int(world.get("sequence", 0)),
            "world_timestamp": _finite(world.get("timestamp")),
            "reason": reason,
        }

    @staticmethod
    def _has_active_operation(world: dict[str, Any]) -> bool:
        operation = world.get("active_operation")
        return isinstance(operation, dict) and operation.get("status") in (
            "issuing", "running", "unknown"
        )

    @staticmethod
    def _eligible_cartons(world: dict[str, Any]) -> list[int]:
        return [
            i for i, item in enumerate(world["cartons"])
            if item.get("id") is not None
            and isinstance(item.get("approach_pose"), dict)
        ]

    @staticmethod
    def _eligible_zones(
        world: dict[str, Any]
    ) -> list[int]:
        return [
            i for i, item in enumerate(world["zones"])
            if item.get("id") is not None
            and isinstance(item.get("approach_pose"), dict)
        ]

    def _distributions(
        self, world: dict[str, Any], output: dict[str, torch.Tensor]
    ) -> tuple[Categorical, dict[str, list[int]]]:
        mask = torch.zeros(len(ACTIONS), dtype=torch.bool, device=self.device)
        mask[0] = True
        candidates = {"GO_TO_CARTON": [], "GO_TO_ZONE": []}
        if not self._has_active_operation(world):
            candidates["GO_TO_CARTON"] = self._eligible_cartons(world)
            candidates["GO_TO_ZONE"] = self._eligible_zones(world)
            mask[1] = bool(candidates["GO_TO_CARTON"])
            mask[2] = bool(candidates["GO_TO_ZONE"])
            mask[3:] = True
        return Categorical(logits=output["action_logits"][0].masked_fill(~mask, -torch.inf)), candidates

    def _evaluate_output(
        self, world: dict[str, Any], decision: dict[str, Any], output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, candidates = self._distributions(world, output)
        action = decision["action"]
        action_index = ACTIONS.index(action)
        if not torch.isfinite(distribution.logits[action_index]).item():
            raise ValueError(f"recorded action {action} is incompatible with its observation")
        log_prob = distribution.log_prob(torch.tensor(action_index, device=self.device))
        entropy = distribution.entropy()
        if action in candidates:
            group, key = ("cartons", "target_id") if action == "GO_TO_CARTON" else ("zones", "zone_id")
            eligible = candidates[action]
            matches = [local for local, index in enumerate(eligible)
                       if world[group][index].get("id") == decision.get(key)]
            if decision.get(key) is None or len(matches) != 1:
                raise ValueError(f"recorded {key} must identify one eligible entity")
            scores = output["carton_scores" if group == "cartons" else "zone_scores"][0, eligible]
            target_distribution = Categorical(logits=scores)
            log_prob = log_prob + target_distribution.log_prob(torch.tensor(matches[0], device=self.device))
            entropy = entropy + target_distribution.entropy()
        return log_prob, entropy, output["value"][0]

    def evaluate_action(
        self, world: dict[str, Any], decision: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable joint action/target likelihood, using inference's masks."""
        reason = self._pose_unavailable(world)
        if reason is not None:
            raise ValueError(f"recorded stochastic decision has no usable pose: {reason}")
        return self._evaluate_output(world, decision, self.network(world))

    @staticmethod
    def _pose_unavailable(world: dict[str, Any]) -> str | None:
        if world.get("ego_frame_status") in ("unknown", "stale"):
            return "robot pose estimate cannot define a fresh ego frame"
        robot = world.get("robot")
        if not isinstance(robot, dict) or not isinstance(robot.get("pose"), dict):
            return "robot pose estimate is unknown"
        pose_age = robot.get("pose_age_s")
        if pose_age is not None and _finite(pose_age, math.inf) > 0.75:
            return "robot pose estimate is stale"
        return None

    def decide(self, world: dict[str, Any], *, deterministic: bool = True) -> dict[str, Any]:
        reason = self._pose_unavailable(world)
        if reason is not None:
            return self._wait(world, reason)
        with torch.no_grad():
            output = self.network(world)
            distribution, candidates = self._distributions(world, output)
            action_index = int(torch.argmax(distribution.logits).item()) if deterministic else int(distribution.sample().item())
            action = ACTIONS[action_index]
            decision: dict[str, Any] = {
                "action": action, "world_sequence": int(world["sequence"]),
                "world_timestamp": _finite(world.get("timestamp")),
                "reason": "operation in progress" if self._has_active_operation(world) else "policy output",
            }
            if action in candidates:
                group, key = ("cartons", "target_id") if action == "GO_TO_CARTON" else ("zones", "zone_id")
                eligible = candidates[action]
                scores = output["carton_scores" if group == "cartons" else "zone_scores"][0, eligible]
                local = int(torch.argmax(scores).item()) if deterministic else int(Categorical(logits=scores).sample().item())
                entity = world[group][eligible[local]]
                decision[key] = entity["id"]
                decision["target_pose"] = entity["approach_pose"]
            log_prob, _, value = self._evaluate_output(world, decision, output)
            decision["behavior"] = {"sampled": not deterministic, "log_prob": float(log_prob), "value": float(value)}
        return decision


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default
