"""Allow-listed ROS command executor backed by asynchronous subprocesses."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any



class ActionExecutor:
    """Translate agent intents into allow-listed ROS CLI calls.

    Each ROS subprocess sources the workspace overlay before executing its
    command. Long commands stay active while new world snapshots arrive;
    incompatible motion/manipulation commands are rejected until the active
    operation has a terminal result.
    """

    def __init__(
        self,
        *,
        ros_setup: str | Path | None = None,
        command_timeout_s: float = 35.0,
        navigation_timeout_s: float = 90.0,
        position_tolerance_m: float = 0.04,
        heading_tolerance_rad: float = 0.10,
        pose_freshness_s: float = 0.75,
    ) -> None:
        setup_path = Path(ros_setup) if ros_setup is not None else (
            Path(__file__).resolve().parents[2] / "install" / "setup.bash"
        )
        self.ros_setup = setup_path.expanduser().resolve()
        self.command_timeout_s = command_timeout_s
        self.navigation_timeout_s = navigation_timeout_s
        self.position_tolerance_m = position_tolerance_m
        self.heading_tolerance_rad = heading_tolerance_rad
        self.pose_freshness_s = pose_freshness_s
        self.active: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None

    def observation(self) -> dict[str, Any] | None:
        if self.active is None:
            return None
        observation = {
            key: self.active.get(key)
            for key in ("id", "type", "target_id", "zone_id", "status", "target_pose", "reason")
        }
        observation["elapsed_s"] = max(0.0, time.monotonic() - self.active["started_at"])
        observation["timeout_s"] = (
            self.navigation_timeout_s
            if self.active["type"].startswith("GO_TO_") and self.active["status"] != "issuing"
            else self.command_timeout_s
        )
        return observation

    async def update(self, world: dict[str, Any]) -> None:
        """Advance process/navigation state using the newest world snapshot."""
        operation = self.active
        if operation is None:
            return

        if operation["type"] in ("GO_TO_CARTON", "GO_TO_ZONE"):
            if operation["status"] == "issuing":
                task = operation["task"]
                if not task.done():
                    return
                result = task.result()
                if result["status"] == "succeeded":
                    operation["status"] = "running"
                    operation["started_at"] = time.monotonic()
                    operation["reason"] = "goal published; monitoring estimated pose"
                elif result["status"] == "unknown":
                    operation["status"] = "unknown"
                    operation["reason"] = result["reason"]
                    return
                else:
                    self._finish(result["status"], result["reason"], world)
                    return

            if operation["status"] in ("running", "unknown"):
                pose_age = _finite(world["robot"].get("pose_age_s"), math.inf)
                if pose_age <= self.pose_freshness_s and self._at_goal(world, operation["target_pose"]):
                    self._finish("succeeded", "goal reached in a fresh pose estimate", world)
                    return
                elapsed = time.monotonic() - operation["started_at"]
                if elapsed > self.navigation_timeout_s:
                    # There is no navigation cancel API yet; retain the lock.
                    operation["status"] = "unknown"
                    operation["reason"] = "navigation timeout; goal may still be active"
                elif operation["status"] == "unknown" and pose_age <= self.pose_freshness_s:
                    operation["reason"] = "navigation outcome unknown; waiting for a safe terminal condition"
            return

        task = operation["task"]
        if not task.done():
            return
        result = task.result()
        if result["status"] == "unknown":
            # A timeout can leave a service/action running at the ROS server.
            # Keep the actuator locked until an operator or a ROS status source
            # resolves the uncertainty.
            operation["status"] = "unknown"
            operation["reason"] = result["reason"]
            return
        self._finish(result["status"], result["reason"], world, result.get("output", ""))

    async def dispatch(self, decision: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        action = decision.get("action")
        if action == "WAIT":
            return {"status": "waiting", "active_operation": self.observation()}
        if action not in (
            "GO_TO_CARTON", "GO_TO_ZONE", "CENTER", "TAKE", "RELEASE"
        ):
            return {"status": "rejected", "reason": f"unsupported action: {action!r}"}
        if self.active is not None:
            return {
                "status": "rejected",
                "reason": f"{self.active['type']} is still {self.active['status']}",
                "active_operation": self.observation(),
            }
        external_operation = world.get("active_operation")
        if isinstance(external_operation, dict) and external_operation.get("status") in (
            "issuing", "running", "unknown"
        ):
            return {
                "status": "rejected",
                "reason": "another controller reports an active operation",
                "active_operation": external_operation,
            }
        if int(decision.get("world_sequence", -1)) != int(world.get("sequence", -2)):
            return {"status": "rejected", "reason": "decision is based on a stale world sequence"}

        try:
            argv, target_pose = self._command(decision, world)
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "rejected", "reason": str(error)}

        operation_id = str(uuid.uuid4())
        ros_selects_target = action in ("CENTER", "TAKE")
        operation = {
            "id": operation_id,
            "type": action,
            "target_id": None if ros_selects_target else decision.get("target_id"),
            "zone_id": None if ros_selects_target else decision.get("zone_id"),
            "status": "issuing" if action.startswith("GO_TO_") else "running",
            "started_at": time.monotonic(),
            "started_sequence": int(world["sequence"]),
            "target_pose": target_pose,
            "reason": "ROS command is being executed",
        }
        operation["task"] = asyncio.create_task(
            self._run_command(action, argv, self.command_timeout_s)
        )
        self.active = operation
        return {"status": operation["status"], "operation": self.observation(), "argv": argv}

    def _command(
        self, decision: dict[str, Any], world: dict[str, Any]
    ) -> tuple[list[str], dict[str, float] | None]:
        action = decision["action"]
        if action == "GO_TO_CARTON":
            target = _find_by_id(world["cartons"], decision.get("target_id"), "carton")
            target_pose = _pose(target.get("approach_pose"), "carton approach_pose")
            self._check_map_bounds(target_pose, world)
            message = _goal_message(target_pose)
            return [
                "ros2", "topic", "pub", "--once", "/goal_pose",
                "geometry_msgs/msg/PoseStamped", json.dumps(message, separators=(",", ":")),
            ], target_pose

        if action == "GO_TO_ZONE":
            zone = _find_by_id(world["zones"], decision.get("zone_id"), "zone")
            target_pose = _pose(zone.get("approach_pose"), "zone approach_pose")
            self._check_map_bounds(target_pose, world)
            message = _goal_message(target_pose)
            return [
                "ros2", "topic", "pub", "--once", "/goal_pose",
                "geometry_msgs/msg/PoseStamped", json.dumps(message, separators=(",", ":")),
            ], target_pose

        if action == "CENTER":
            return [
                "ros2", "action", "send_goal", "/rock_centering",
                "robot_msgs/action/RockCentering", "{}", "--feedback",
            ], None

        if action == "TAKE":
            return [
                "ros2", "service", "call", "/robot_omni/ventouse/set",
                "std_srvs/srv/SetBool", "{data: true}",
            ], None

        if action == "RELEASE":
            return [
                "ros2", "service", "call", "/robot_omni/ventouse/set",
                "std_srvs/srv/SetBool", "{data: false}",
            ], None
        raise ValueError(f"unsupported action: {action!r}")

    async def _run_command(
        self, action: str, argv: list[str], timeout_s: float
    ) -> dict[str, Any]:
        if not self.ros_setup.is_file():
            return {
                "status": "failed",
                "reason": f"ROS setup script not found: {self.ros_setup}",
            }

        # `source` is a shell builtin. Keep the setup path and the allow-listed
        # ROS argv as separate positional arguments so snapshot data is never
        # interpolated into shell code.
        shell_script = 'source "$1" || exit 125; shift; exec "$@"'
        command = [
            "/bin/bash", "--noprofile", "--norc", "-c", shell_script,
            "action_executor", str(self.ros_setup), *argv,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as error:
            return {"status": "failed", "reason": f"could not start ROS CLI: {error}"}

        if self.active is not None:
            self.active["process"] = process
        try:
            output_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            return {
                "status": "unknown",
                "reason": "ROS CLI timed out; server-side action may still be active",
            }
        except asyncio.CancelledError:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            raise
        except Exception as error:
            return {
                "status": "unknown",
                "reason": f"ROS CLI communication failed; remote outcome is uncertain: {error}",
            }

        output = output_bytes.decode("utf-8", errors="replace")
        if process.returncode == 125:
            return {
                "status": "failed",
                "reason": f"could not source ROS setup script: {self.ros_setup}",
                "output": output,
            }
        if action.startswith("GO_TO_"):
            if process.returncode == 0:
                return {"status": "succeeded", "reason": "goal published", "output": output}
            return {
                "status": "failed",
                "reason": f"goal publication failed with exit code {process.returncode}",
                "output": output,
            }
        if process.returncode != 0:
            return {
                "status": "unknown",
                "reason": f"ROS CLI exited with code {process.returncode}; remote result is uncertain",
                "output": output,
            }
        match = re.search(r"\bsuccess\s*[:=]\s*(true|false)\b", output, re.IGNORECASE)
        if match is None:
            return {
                "status": "unknown",
                "reason": "could not parse a terminal success field from ROS CLI output",
                "output": output,
            }
        success = match.group(1).lower() == "true"
        return {
            "status": "succeeded" if success else "failed",
            "reason": "ROS operation succeeded" if success else "ROS operation reported failure",
            "output": output,
        }

    def _finish(
        self, status: str, reason: str, world: dict[str, Any], output: str = ""
    ) -> None:
        if self.active is None:
            return
        self.last_result = {
            "id": self.active["id"],
            "type": self.active["type"],
            "target_id": self.active.get("target_id"),
            "zone_id": self.active.get("zone_id"),
            "status": status,
            "reason": reason,
            "finished_sequence": int(world.get("sequence", 0)),
            "output": output[-4000:],
        }
        self.active = None

    def _at_goal(self, world: dict[str, Any], goal: dict[str, float]) -> bool:
        robot_pose = _pose(world["robot"].get("pose"), "robot.pose")
        distance = math.hypot(robot_pose["x"] - goal["x"], robot_pose["y"] - goal["y"])
        angle_error = abs(_wrap(robot_pose["yaw"] - goal["yaw"]))
        return (
            distance <= self.position_tolerance_m
            and angle_error <= self.heading_tolerance_rad
        )

    @staticmethod
    def _check_map_bounds(pose: dict[str, float], world: dict[str, Any]) -> None:
        arena = world.get("arena", {})
        bounds = arena.get("bounds", arena) if isinstance(arena, dict) else {}
        x_min = _finite(bounds.get("x_min"), 0.0)
        x_max = _finite(bounds.get("x_max"), 3.0)
        y_min = _finite(bounds.get("y_min"), 0.0)
        y_max = _finite(bounds.get("y_max"), 2.0)
        if not (x_min <= pose["x"] <= x_max and y_min <= pose["y"] <= y_max):
            raise ValueError("target pose lies outside configured map bounds")

    async def close(self) -> None:
        """Stop the local CLI client; this does not cancel remote ROS goals."""
        if self.active is None:
            return
        process = self.active.get("process")
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        task = self.active.get("task")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.active["status"] = "unknown"
        self.active["reason"] = "local runner stopped; remote ROS goal may still be active"


def _find_by_id(items: list[dict[str, Any]], identifier: Any, label: str) -> dict[str, Any]:
    for item in items:
        if item.get("id") == identifier:
            return item
    raise ValueError(f"selected {label} {identifier!r} is absent from the current world")


def _pose(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    position = value.get("position", {})
    if not isinstance(position, dict):
        position = {}
    result = {
        "x": _finite(value.get("x", position.get("x")), math.nan),
        "y": _finite(value.get("y", position.get("y")), math.nan),
        "z": _finite(value.get("z", position.get("z")), 0.0),
        "yaw": _finite(value.get("yaw", value.get("theta")), math.nan),
    }
    if not all(math.isfinite(number) for number in result.values()):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return result


def _goal_message(pose: dict[str, float]) -> dict[str, Any]:
    half = pose["yaw"] / 2.0
    return {
        "header": {"frame_id": "map"},
        "pose": {
            "position": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
            "orientation": {"z": math.sin(half), "w": math.cos(half)},
        },
    }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
