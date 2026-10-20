"""Named, normalized observations; separate encoders for each entity family."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


EMBEDDING_SIZE = 32
CONTEXT_SIZE = 64
ENTITY_GROUPS = ("cartons", "obstacles", "zones", "aruco_detections")
ACTION_TYPES = ("WAIT", "GO_TO_CARTON", "GO_TO_ZONE", "CENTER", "TAKE", "RELEASE")


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _pose(item: dict[str, Any]) -> dict[str, Any]:
    if "pose" in item:
        return _object(item["pose"])
    return _object(item.get("center", item))


def _coordinate(pose: dict[str, Any], key: str) -> Any:
    return pose.get(key, _object(pose.get("position")).get(key))


def _yaw(item: dict[str, Any]) -> Any:
    pose = _pose(item)
    return item.get("yaw", item.get("theta", pose.get("yaw", pose.get("theta"))))


def _value(row: dict[str, float], name: str, value: Any, scale: float = 1.0) -> None:
    number = _number(value)
    row[name] = 0.0 if number is None else number / scale
    row[name + "_known"] = float(number is not None)


def _flag(row: dict[str, float], name: str, value: Any) -> None:
    row[name] = float(value is True)
    row[name + "_known"] = float(isinstance(value, bool))


def _category(row: dict[str, float], name: str, value: Any, choices: tuple[str, ...]) -> None:
    for choice in (*choices, "unknown"):
        row[name + "_" + choice] = float(value == choice if choice != "unknown" else value not in choices)


def _bounded(value: Any) -> float | None:
    """Invertible normalization for counts/scores with no configured maximum."""
    number = _number(value)
    return None if number is None else 0.5 + number / (2.0 * (1.0 + abs(number)))


def _count(value: Any) -> float | None:
    number = _number(value)
    return None if number is None or number < 0 else number / (1.0 + number)


def _posture(item: dict[str, Any]) -> str | None:
    if "posture" in item:
        return item["posture"] if item["posture"] in ("vertical", "horizontal") else None
    size = _object(item.get("size"))
    x, y, z = (_number(size.get(key, size.get(alias))) for key, alias in
               (("x", "length"), ("y", "width"), ("z", "height")))
    if x is not None and y is not None:
        if z is not None and z > max(x, y):
            return "vertical"
        if not math.isclose(x, y) and (z is None or max(x, y) > z):
            return "horizontal"
    return None


def opponent_obstacle(robot: dict[str, Any]) -> dict[str, Any]:
    """Only an observed opponent position is part of the policy input."""
    pose = _pose(robot)
    return {"id": robot.get("id"), "pose": {
        "x": _coordinate(pose, "x"), "y": _coordinate(pose, "y"),
    }}


class _Frame:
    def __init__(self, world: dict[str, Any]) -> None:
        arena = _object(world.get("arena"))
        corners = arena.get("corners")
        if corners is None:
            bounds = _object(arena.get("bounds", arena))
            x0, x1 = bounds.get("x_min", 0.0), bounds.get("x_max", 3.0)
            y0, y1 = bounds.get("y_min", 0.0), bounds.get("y_max", 2.0)
            corners = [{"x": x0, "y": y0}, {"x": x1, "y": y0},
                       {"x": x1, "y": y1}, {"x": x0, "y": y1}]
        points = np.asarray([[point["x"], point["y"]] for point in corners], dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("arena must define four finite, ordered rectangular corners")
        self.origin = points[0]
        u, v = points[1] - points[0], points[3] - points[0]
        self.width, self.height = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if min(self.width, self.height) <= 0 or not np.isclose(u @ v, 0, atol=1e-6):
            raise ValueError("arena must be a non-degenerate rectangle")
        if not np.allclose(points[2], points[0] + u + v, atol=1e-6):
            raise ValueError("arena corners must follow the rectangle perimeter")
        self.u, self.v = u / self.width, v / self.height
        self.heading = math.atan2(float(self.u[1]), float(self.u[0]))
        self.length_scale = max(self.width, self.height)
        total = _number(_object(world.get("mission")).get("total_s"))
        self.total_s = total if total is not None and total > 0 else 90.0

    def position(self, row: dict[str, float], name: str, item: dict[str, Any], *, z: bool = False) -> None:
        pose = _pose(item)
        x, y = (_number(_coordinate(pose, key)) for key in ("x", "y"))
        for key, axis, scale in (("x", self.u, self.width), ("y", self.v, self.height)):
            components = [(value, float(weight), float(origin))
                          for value, weight, origin in zip((x, y), axis, self.origin)
                          if abs(weight) > 1e-12]
            projected = None if any(value is None for value, _, _ in components) else sum(
                (value - origin) * weight for value, weight, origin in components
            ) / scale
            _value(row, name + "_" + key, projected)
        if z:
            _value(row, name + "_z", _coordinate(pose, "z"), self.length_scale)

    def angle(self, row: dict[str, float], name: str, value: Any, symmetry: int = 1) -> None:
        angle = _number(value)
        row[name + "_sin"] = 0.0 if angle is None else (math.sin(symmetry * (angle - self.heading)) + 1.0) / 2.0
        row[name + "_cos"] = 0.0 if angle is None else (math.cos(symmetry * (angle - self.heading)) + 1.0) / 2.0
        row[name + "_known"] = float(angle is not None)


def _quality(row: dict[str, float], item: dict[str, Any], frame: _Frame) -> None:
    _value(row, "age", item.get("age_s"), frame.total_s)
    _value(row, "confidence", item.get("confidence"))


def _carton_geometry(row: dict[str, float], item: dict[str, Any], frame: _Frame, prefix: str = "") -> None:
    frame.position(row, prefix + "position", item, z=True)
    posture = _posture(item)
    _category(row, prefix + "posture", posture, ("vertical", "horizontal"))
    # A square footprint has period pi/2; a rectangular footprint has period pi.
    frame.angle(row, prefix + "angle", _yaw(item) if posture else None, 4 if posture == "vertical" else 2)
    size = _object(item.get("size"))
    for key, alias in (("x", "length"), ("y", "width"), ("z", "height")):
        _value(row, prefix + "size_" + key, size.get(key, size.get(alias)), frame.length_scale)


def _approach(row: dict[str, float], item: dict[str, Any], frame: _Frame) -> None:
    approach = _object(item.get("approach_pose"))
    frame.position(row, "approach", approach, z=True)
    frame.angle(row, "approach_angle", _yaw(approach))


def _target_matches(item: dict[str, Any], operation: dict[str, Any], key: str) -> bool | None:
    target = operation.get(key)
    return None if target is None or item.get("id") is None else item["id"] == target


class WorldEncoder(nn.Module):
    """No truncation, velocities, arbitrary numeric IDs, or invented measurements.

    `feature_dicts` exposes every named input before learned compression. Positions
    are normalized in the arena axes even when the snapshot uses an ego frame.
    Values outside physical bounds are not clipped, so normalization loses no data.
    """

    def __init__(self) -> None:
        super().__init__()
        sample = {"robot": {"pose": {"x": 0, "y": 0, "yaw": 0}}}
        frame = _Frame(sample)
        templates = {"robot": self._robot_features(sample, frame), "context": self._context_features(sample, frame)}
        templates.update({group: self._entity_features({}, group, sample, frame) for group in ENTITY_GROUPS})
        self.feature_names = {key: tuple(row) for key, row in templates.items()}
        self.robot_mlp = self._mlp(len(templates["robot"]))
        self.entity_mlps = nn.ModuleDict({group: self._mlp(len(templates[group])) for group in ENTITY_GROUPS})
        self.context_mlp = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE * (1 + len(ENTITY_GROUPS)) + len(templates["context"]), 96),
            nn.Tanh(), nn.Linear(96, CONTEXT_SIZE), nn.Tanh(),
        )

    @staticmethod
    def _mlp(inputs: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, EMBEDDING_SIZE), nn.Tanh())

    @staticmethod
    def _robot_features(world: dict[str, Any], frame: _Frame) -> dict[str, float]:
        robot = _object(world.get("robot"))
        row: dict[str, float] = {}
        frame.position(row, "position", robot)
        frame.angle(row, "angle", _yaw(robot))
        _value(row, "pose_age", robot.get("pose_age_s"), frame.total_s)
        _category(row, "payload", robot.get("payload_state"), ("empty", "loaded"))
        _category(row, "suction", robot.get("suction_state"), ("free", "aspirating", "holding", "lowering"))
        held = robot.get("held_object")
        _flag(row, "held", True if isinstance(held, dict) else False if robot.get("payload_state") == "empty" else None)
        _carton_geometry(row, _object(held), frame, "held_")
        return row

    @staticmethod
    def _context_features(world: dict[str, Any], frame: _Frame) -> dict[str, float]:
        row: dict[str, float] = {}
        mission = _object(world.get("mission"))
        _value(row, "remaining", mission.get("remaining_s"), frame.total_s)
        row["total_seconds"] = _count(frame.total_s)
        row["arena_width"] = _count(frame.width)
        row["arena_height"] = _count(frame.height)
        _value(row, "score", _bounded(mission.get("score")))
        for key in ("cartons_delivered", "cartons_remaining"):
            _value(row, key, _count(mission.get(key)))
        operation = _object(world.get("active_operation"))
        _category(row, "operation_type", operation.get("type", "idle"), (*ACTION_TYPES, "idle"))
        _category(row, "operation_status", operation.get("status", "idle"), ("idle", "issuing", "running"))
        timeout = _number(operation.get("timeout_s"))
        _value(row, "operation_elapsed", operation.get("elapsed_s"), timeout if timeout is not None and timeout > 0 else frame.total_s)
        _value(row, "operation_timeout", operation.get("timeout_s"), frame.total_s)
        goal = _object(operation.get("target_pose"))
        if not goal:
            for group, key in (("cartons", "target_id"), ("zones", "zone_id")):
                matches = [item for item in world.get(group, []) if operation.get(key) is not None and item.get("id") == operation[key]]
                if len(matches) == 1:
                    goal = _object(matches[0].get("approach_pose"))
        frame.position(row, "operation_goal", goal, z=True)
        frame.angle(row, "operation_goal_angle", _yaw(goal))
        last = _object(world.get("last_action_result"))
        _category(row, "last_type", last.get("type", "idle"), (*ACTION_TYPES, "idle"))
        _category(row, "last_status", last.get("status", "idle"), ("idle", "succeeded", "failed", "rejected", "canceled"))
        for group in ENTITY_GROUPS:
            row[group + "_count"] = _count(len(world.get(group, [])))
            source = _object(_object(world.get("sources")).get({"aruco_detections": "aruco"}.get(group, group)))
            _category(row, group + "_source", source.get("status"), ("fresh", "stale", "error", "invalid", "wrong_frame"))
        aruco = _object(_object(world.get("sources")).get("aruco"))
        _category(row, "aruco_transform", aruco.get("map_transform_status"), ("aligned", "time_skew"))
        return row

    @staticmethod
    def _entity_features(item: dict[str, Any], group: str, world: dict[str, Any], frame: _Frame) -> dict[str, float]:
        row: dict[str, float] = {}
        if group == "cartons":
            _carton_geometry(row, item, frame)
            _quality(row, item, frame)
            _flag(row, "visible", item.get("visible"))
            _flag(row, "accessible", item.get("accessible"))
            _approach(row, item, frame)
            robot = _object(world.get("robot"))
            held = _object(robot.get("held_object"))
            _flag(row, "is_held", _target_matches(item, {"target_id": held.get("id")}, "target_id") if robot.get("payload_state") != "empty" else False)
        elif group == "zones":
            frame.position(row, "position", item)
            frame.angle(row, "angle", _yaw(item), 2)
            for key in ("length", "width", "margin"):
                _value(row, key, item.get(key), frame.length_scale)
            for key in ("capacity", "occupied"):
                _value(row, key, _count(item.get(key)))
            for key in ("release_allowed", "accessible"):
                _flag(row, key, item.get(key))
            _quality(row, item, frame)
            _approach(row, item, frame)
        elif group == "obstacles":
            frame.position(row, "position", item)
            _value(row, "radius", item.get("radius"), frame.length_scale)
            size = _object(item.get("size"))
            for key, axis in (("length", "x"), ("width", "y")):
                _value(row, key, item.get(key, size.get(axis, size.get(key))), frame.length_scale)
            frame.angle(row, "angle", _yaw(item), 2)
            _quality(row, item, frame)
        elif group == "aruco_detections":
            frame.position(row, "position", item, z=True)
            marker_id = item.get("marker_id")
            _flag(row, "centering_candidate", None if marker_id is None else marker_id == 13)
            normal = _object(item.get("normal"))
            nx, ny = _number(normal.get("x")), _number(normal.get("y"))
            for key, value in (("x", None if nx is None or ny is None else nx * frame.u[0] + ny * frame.u[1]),
                               ("y", None if nx is None or ny is None else nx * frame.v[0] + ny * frame.v[1]),
                               ("z", _number(normal.get("z")))):
                _value(row, "normal_" + key, None if value is None else (value + 1) / 2)
            _value(row, "camera_depth", item.get("camera_depth"), frame.length_scale)
            _quality(row, item, frame)
            _flag(row, "associated", item.get("associated_carton_id") is not None)
        if group in ("cartons", "zones"):
            key = "target_id" if group == "cartons" else "zone_id"
            for name, operation in (("operation_target", world.get("active_operation")), ("last_target", world.get("last_action_result"))):
                _flag(row, name, _target_matches(item, _object(operation), key))
        return row

    def feature_dicts(self, world: dict[str, Any]) -> dict[str, Any]:
        """Return named scalar rows and ID associations, without mutating input."""
        robot = _object(world.get("robot"))
        if any(_number(value) is None for value in (_coordinate(_pose(robot), "x"), _coordinate(_pose(robot), "y"), _yaw(robot))):
            raise ValueError("world.robot.pose must contain finite x, y and yaw")
        groups = {}
        for group in (*ENTITY_GROUPS, "other_robots"):
            items = world.get(group, [])
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise ValueError(f"world.{group} must be a list of objects")
            groups[group] = list(items)
        groups["obstacles"].extend(opponent_obstacle(robot) for robot in groups.pop("other_robots"))
        view = {**world, **groups}
        frame = _Frame(view)
        rows: dict[str, Any] = {"robot": self._robot_features(view, frame), "context": self._context_features(view, frame)}
        rows.update({group: [self._entity_features(item, group, view, frame) for item in groups[group]] for group in ENTITY_GROUPS})
        links = np.zeros((len(groups["cartons"]), len(groups["aruco_detections"])), dtype=np.float32)
        ids = [item.get("id") for item in groups["cartons"]]
        for index, detection in enumerate(groups["aruco_detections"]):
            identifier = detection.get("associated_carton_id")
            if identifier is not None and ids.count(identifier) == 1:
                links[ids.index(identifier), index] = 1.0
        rows["aruco_associations"] = links
        return rows

    def forward(self, world: dict[str, Any]) -> dict[str, Any]:
        rows = self.feature_dicts(world)
        device = next(self.parameters()).device

        def tensor(values: list[dict[str, float]], group: str) -> torch.Tensor:
            matrix = [[row[key] for key in self.feature_names[group]] for row in values]
            return torch.as_tensor(matrix, dtype=torch.float32, device=device).reshape(1, len(values), len(self.feature_names[group]))

        robot = self.robot_mlp(tensor([rows["robot"]], "robot").squeeze(1))
        entities = {group: self.entity_mlps[group](tensor(rows[group], group)) for group in ENTITY_GROUPS}
        links = torch.as_tensor(rows["aruco_associations"], device=device)
        if links.numel():
            # Preserve tag/carton associations without assigning numeric meaning to IDs.
            associated = links @ entities["aruco_detections"][0]
            associated = associated / links.sum(dim=1, keepdim=True).clamp_min(1)
            entities["cartons"] = entities["cartons"] + associated.unsqueeze(0)
        pooled = [entities[group].mean(dim=1) if entities[group].shape[1] else torch.zeros_like(robot) for group in ENTITY_GROUPS]
        context = tensor([rows["context"]], "context").squeeze(1)
        return {"context": self.context_mlp(torch.cat([robot, *pooled, context], dim=-1)),
                "entities": entities}
