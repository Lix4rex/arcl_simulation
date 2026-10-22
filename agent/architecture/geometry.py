"""Deterministic geometry checks shared by policy masking and execution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def payload_fits_zone(
    world: dict[str, Any],
    zone: dict[str, Any],
    *,
    object_pose: dict[str, float] | None = None,
) -> bool:
    """Check capacity and all four oriented carton corners against a drop zone."""
    if zone.get("release_allowed") is not True:
        return False
    capacity = _finite(zone.get("capacity"), math.nan)
    occupied = _finite(zone.get("occupied"), math.nan)
    if not math.isfinite(capacity) or not math.isfinite(occupied) or occupied >= capacity:
        return False

    robot = world.get("robot", {})
    held = robot.get("held_object") if isinstance(robot, dict) else None
    if not isinstance(held, dict):
        return False
    size = held.get("size", {})
    if not isinstance(size, dict):
        return False

    try:
        if object_pose is None:
            object_pose = _pose(held.get("pose"), "held_object.pose")
        raw_zone_pose = zone.get("pose")
        if not isinstance(raw_zone_pose, dict):
            raw_zone_pose = zone.get("center", zone)
        if isinstance(raw_zone_pose, dict) and "yaw" not in raw_zone_pose and "theta" not in raw_zone_pose:
            raw_zone_pose = {**raw_zone_pose, "yaw": zone.get("yaw")}
        zone_pose = _pose(raw_zone_pose, "zone pose")
    except ValueError:
        return False
    if "yaw" in zone:
        zone_pose["yaw"] = _finite(zone.get("yaw"), zone_pose["yaw"])

    carton_length = _finite(size.get("x", size.get("length")), math.nan)
    carton_width = _finite(size.get("y", size.get("width")), math.nan)
    zone_length = _finite(zone.get("length"), math.nan)
    zone_width = _finite(zone.get("width"), math.nan)
    margin = _finite(zone.get("margin"), 0.0)
    if not math.isfinite(margin) or margin < 0.0:
        return False
    if not all(math.isfinite(value) and value > 0.0 for value in (
        carton_length, carton_width, zone_length, zone_width
    )):
        return False

    local_corners = np.asarray([
        [-carton_length / 2.0, -carton_width / 2.0],
        [-carton_length / 2.0, carton_width / 2.0],
        [carton_length / 2.0, -carton_width / 2.0],
        [carton_length / 2.0, carton_width / 2.0],
    ], dtype=np.float64)
    object_rotation = np.asarray([
        [math.cos(object_pose["yaw"]), -math.sin(object_pose["yaw"])],
        [math.sin(object_pose["yaw"]), math.cos(object_pose["yaw"])],
    ])
    zone_rotation_inverse = np.asarray([
        [math.cos(zone_pose["yaw"]), math.sin(zone_pose["yaw"])],
        [-math.sin(zone_pose["yaw"]), math.cos(zone_pose["yaw"])],
    ])
    object_center = np.asarray([object_pose["x"], object_pose["y"]])
    zone_center = np.asarray([zone_pose["x"], zone_pose["y"]])
    world_corners = local_corners @ object_rotation.T + object_center
    zone_corners = (world_corners - zone_center) @ zone_rotation_inverse.T
    limits = np.asarray([zone_length / 2.0 - margin, zone_width / 2.0 - margin])
    return bool(np.all(np.abs(zone_corners) <= limits))


def payload_fits_zone_at_approach(world: dict[str, Any], zone: dict[str, Any]) -> bool:
    """Predict the held carton pose when the robot reaches this zone approach."""
    robot = world.get("robot", {})
    held = robot.get("held_object") if isinstance(robot, dict) else None
    approach = zone.get("approach_pose")
    if not isinstance(held, dict) or not isinstance(approach, dict):
        return False
    try:
        robot_pose = _pose(robot.get("pose"), "robot.pose")
        held_pose = _pose(held.get("pose"), "held_object.pose")
        goal_pose = _pose(approach, "zone approach_pose")
    except ValueError:
        return False

    relative_x = held_pose["x"] - robot_pose["x"]
    relative_y = held_pose["y"] - robot_pose["y"]
    current_cosine = math.cos(robot_pose["yaw"])
    current_sine = math.sin(robot_pose["yaw"])
    local_x = current_cosine * relative_x + current_sine * relative_y
    local_y = -current_sine * relative_x + current_cosine * relative_y
    goal_cosine = math.cos(goal_pose["yaw"])
    goal_sine = math.sin(goal_pose["yaw"])
    predicted_pose = {
        "x": goal_pose["x"] + goal_cosine * local_x - goal_sine * local_y,
        "y": goal_pose["y"] + goal_sine * local_x + goal_cosine * local_y,
        "yaw": goal_pose["yaw"] + held_pose["yaw"] - robot_pose["yaw"],
    }
    return payload_fits_zone(world, zone, object_pose=predicted_pose)


def _pose(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    position = value.get("position", {})
    if not isinstance(position, dict):
        position = {}
    result = {
        "x": _finite(value.get("x", position.get("x")), math.nan),
        "y": _finite(value.get("y", position.get("y")), math.nan),
        "yaw": _finite(value.get("yaw", value.get("theta")), math.nan),
    }
    if not all(math.isfinite(number) for number in result.values()):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return result


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default
