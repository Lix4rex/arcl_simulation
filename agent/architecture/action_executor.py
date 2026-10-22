"""Asynchronous ROS topic/service client for the hybrid robot policy."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from typing import Any

from .policy import ACTIONS


class ActionExecutor:
    """Run one move action or actuator service at a time using typed ROS APIs."""

    def __init__(self, *, navigation_timeout_s: float = 90.0,
                 command_timeout_s: float = 35.0, robot_margin_m: float = 0.20,
                 pose_freshness_s: float = 0.75) -> None:
        self.navigation_timeout_s = float(navigation_timeout_s)
        self.command_timeout_s = float(command_timeout_s)
        self.robot_margin_m = float(robot_margin_m)
        self.pose_freshness_s = float(pose_freshness_s)
        if (not math.isfinite(self.navigation_timeout_s) or self.navigation_timeout_s <= 0
                or not math.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0
                or not math.isfinite(self.robot_margin_m) or self.robot_margin_m < 0
                or not math.isfinite(self.pose_freshness_s) or self.pose_freshness_s <= 0):
            raise ValueError("timeouts and pose freshness must be positive; robot margin must be non-negative")
        self.active: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self._rclpy: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._owns_rclpy = False
        self._goal_publisher: Any = None
        self._suction_clients: dict[int, Any] = {}
        self._wrist_client: Any = None
        self._last_world: dict[str, Any] | None = None

    def start(self) -> None:
        if self._node is not None:
            return
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from std_srvs.srv import SetBool

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        self._node = Node("arcl_agent_executor")
        self._goal_publisher = self._node.create_publisher(PoseStamped, "/goal_pose", 10)
        self._suction_clients = {
            index: self._node.create_client(SetBool, f"/robot_omni/ventouse_{index}/set")
            for index in range(1, 4)
        }
        self._wrist_client = self._node.create_client(SetBool, "/robot_omni/poignet/set")
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def observation(self) -> dict[str, Any] | None:
        if self.active is None:
            return None
        return {
            key: self.active.get(key) for key in (
                "id", "type", "target_pose", "cancel_target_pose", "status", "cancelable", "reason",
                "started_sequence", "failed",
            )
        } | {
            "elapsed_s": max(0.0, time.monotonic() - self.active["started_at"]),
            "timeout_s": self.navigation_timeout_s if self.active["type"] == "MOVE" else self.command_timeout_s,
        }

    async def update(self, world: dict[str, Any]) -> None:
        self._last_world = world
        operation = self.active
        if operation is None:
            return
        if operation["type"] == "MOVE":
            self._update_move(operation, world)
            if self.active is None:
                return
        terminal_future = operation.get("terminal_future")
        if operation["status"] == "unknown" and terminal_future is not None and terminal_future.done():
            try:
                response = terminal_future.result()
            except Exception as error:
                operation["reason"] = f"late remote result remains uncertain: {error}"
                return
            status = "succeeded" if response.success else "failed"
            reason = response.message or ("service succeeded" if response.success else "service reported failure")
            message = response.message
            self._finish(status, reason, world, message)
            return
        task = operation.get("task")
        if task is None or not task.done():
            return
        try:
            result = task.result()
        except Exception as error:
            result = {"status": "unknown", "reason": f"ROS operation outcome uncertain: {error}"}
        if result["status"] == "unknown":
            operation["status"] = "unknown"
            operation["cancelable"] = False
            operation["reason"] = result["reason"]
            return
        if result["status"] == "running":
            operation["status"] = "running"
            operation["reason"] = result["reason"]
            return
        self._finish(result["status"], result["reason"], world, result.get("message", ""))

    async def dispatch(self, decision: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        action = decision.get("action")
        if action == "IDLE":
            return {"status": "waiting", "active_operation": self.observation()}
        if action not in ACTIONS:
            return {"status": "rejected", "reason": f"unsupported action: {action!r}"}
        if int(decision.get("world_sequence", -1)) != int(world.get("sequence", -2)):
            return {"status": "rejected", "reason": "decision is based on a stale world sequence"}
        if action == "CANCEL":
            if self.active is None or self.active.get("type") != "MOVE" or not self.active.get("cancelable"):
                return {"status": "rejected", "reason": "there is no cancelable move"}
            if self.active["status"] in ("cancel_requested", "canceling", "unknown"):
                return {"status": "waiting", "active_operation": self.observation()}
            try:
                cancel_target = self._publish_cancel_goal(world)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                return {"status": "rejected", "reason": str(error)}
            self.active["status"] = "canceling"
            self.active["cancelable"] = False
            self.active["cancel_target_pose"] = cancel_target
            self.active["settled_since"] = None
            self.active["settled_updates"] = 0
            self.active["reason"] = "move goal replaced with a short stop target; awaiting pose and velocity confirmation"
            return {"status": "canceling", "operation": self.observation()}
        if self.active is not None:
            return {"status": "rejected", "reason": f"{self.active['type']} is still {self.active['status']}",
                    "active_operation": self.observation()}
        if "active_operation" not in world:
            return {"status": "rejected", "reason": "active-operation state is unknown"}
        external = world.get("active_operation")
        if external is not None:
            if not isinstance(external, dict) or external.get("status") not in (
                "issuing", "running", "cancel_requested", "canceling", "unknown",
                "succeeded", "failed", "canceled", "rejected",
            ):
                return {"status": "rejected", "reason": "active-operation state is malformed"}
            if external.get("status") not in ("succeeded", "failed", "canceled", "rejected"):
                return {"status": "rejected", "reason": "another controller reports an active operation",
                        "active_operation": external}
        try:
            self._validate_action(action, decision, world)
        except (TypeError, ValueError, KeyError) as error:
            return {"status": "rejected", "reason": str(error)}
        if self._node is None:
            try:
                self.start()
            except (ImportError, RuntimeError) as error:
                return {"status": "failed", "reason": f"ROS client unavailable: {error}"}
        operation_id = str(uuid.uuid4())
        is_move = action == "MOVE"
        operation = {
            "id": operation_id, "type": action,
            "status": "issuing", "cancelable": is_move, "failed": False,
            "started_at": time.monotonic(), "started_sequence": int(world["sequence"]),
            "target_pose": decision.get("target_pose") if is_move else None,
            "reason": "ROS request is being sent",
        }
        self.active = operation
        if is_move:
            try:
                self._publish_goal(decision["target_pose"])
            except (ImportError, RuntimeError, TypeError, ValueError) as error:
                self._finish("failed", f"could not publish MOVE goal: {error}", world)
                return {"status": "failed", "reason": str(error)}
            operation["status"] = "running"
            operation["reason"] = "goal published; waiting for fresh pose and stopped-velocity confirmation"
            return {"status": "running", "operation": self.observation()}
        operation["task"] = asyncio.create_task(self._run_operation(operation, decision, world))
        return {"status": "issuing", "operation": self.observation()}

    def _validate_action(self, action: str, decision: dict[str, Any], world: dict[str, Any]) -> None:
        if action == "MOVE":
            robot = world.get("robot") if isinstance(world.get("robot"), dict) else {}
            pose = robot.get("pose")
            sources = world.get("sources") if isinstance(world.get("sources"), dict) else {}
            pose_source = sources.get("robot_pose") if isinstance(sources.get("robot_pose"), dict) else {}
            pose_age = _finite(robot.get("pose_age_s"), math.inf)
            if (not isinstance(pose, dict) or pose_source.get("fresh", True) is not True
                    or pose_age > self.pose_freshness_s
                    or not all(math.isfinite(_finite(pose.get(key), math.nan)) for key in ("x", "y", "yaw"))):
                raise ValueError("MOVE requires a fresh robot pose")
            target = decision.get("target_pose")
            if not isinstance(target, dict):
                raise ValueError("MOVE requires target_pose")
            x, y = _finite(target.get("x"), math.nan), _finite(target.get("y"), math.nan)
            sine, cosine = _finite(target.get("sin"), math.nan), _finite(target.get("cos"), math.nan)
            if not all(math.isfinite(value) for value in (x, y, sine, cosine)):
                raise ValueError("MOVE target requires finite x, y, sin and cos")
            arena = world.get("arena") if isinstance(world.get("arena"), dict) else {}
            bounds = arena.get("bounds", arena)
            if not isinstance(bounds, dict):
                raise ValueError("arena bounds are unavailable")
            if not (_finite(bounds.get("x_min"), 0.0) + self.robot_margin_m <= x <= _finite(bounds.get("x_max"), 3.0) - self.robot_margin_m
                    and _finite(bounds.get("y_min"), 0.0) + self.robot_margin_m <= y <= _finite(bounds.get("y_max"), 2.0) - self.robot_margin_m):
                raise ValueError("MOVE target lies outside the robot-center arena bounds")
            if math.hypot(x - pose["x"], y - pose["y"]) < 1e-4:
                raise ValueError("MOVE target is too close to the current position for the existing trajectory controller")
            norm = math.hypot(sine, cosine)
            if norm < 1e-6:
                raise ValueError("MOVE target orientation vector has near-zero norm")
        elif action.startswith("SUCTION_"):
            try:
                index, state = action.removeprefix("SUCTION_").split("_")
                if index not in ("1", "2", "3") or state not in ("ON", "OFF"):
                    raise ValueError
            except ValueError as error:
                raise ValueError("invalid suction action") from error
        elif action == "TOGGLE_WRIST":
            robot = world.get("robot") if isinstance(world.get("robot"), dict) else {}
            wrist = _finite(robot.get("wrist_angle_rad"), math.nan)
            sources = world.get("sources") if isinstance(world.get("sources"), dict) else {}
            wrist_source = sources.get("wrist") if isinstance(sources.get("wrist"), dict) else {}
            if (not math.isfinite(wrist) or wrist_source.get("fresh", True) is not True
                    or _finite(wrist_source.get("age_s"), math.inf) > 1.0):
                raise ValueError("TOGGLE_WRIST requires a fresh measured wrist angle")
        else:
            raise ValueError(f"action {action} cannot start an operation")

    async def _run_operation(self, operation: dict[str, Any], decision: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        return await self._run_service(operation, decision, world)

    def _publish_goal(self, target: dict[str, Any]) -> None:
        from geometry_msgs.msg import PoseStamped

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.position.x = float(target["x"])
        goal.pose.position.y = float(target["y"])
        direction_norm = math.hypot(target["sin"], target["cos"])
        yaw = math.atan2(target["sin"] / direction_norm, target["cos"] / direction_norm)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        if self._goal_publisher is None:
            raise RuntimeError("/goal_pose publisher is unavailable")
        self._goal_publisher.publish(goal)

    def _publish_cancel_goal(self, world: dict[str, Any]) -> dict[str, float]:
        robot = world.get("robot") if isinstance(world.get("robot"), dict) else {}
        pose = robot.get("pose")
        sources = world.get("sources") if isinstance(world.get("sources"), dict) else {}
        pose_source = sources.get("robot_pose") if isinstance(sources.get("robot_pose"), dict) else {}
        age = _finite(robot.get("pose_age_s"), math.inf)
        if (not isinstance(pose, dict) or pose_source.get("fresh", True) is not True
                or age > self.pose_freshness_s
                or not all(math.isfinite(_finite(pose.get(key), math.nan)) for key in ("x", "y", "yaw"))):
            raise ValueError("CANCEL needs a fresh robot pose to replace the active /goal_pose")
        arena = world.get("arena") if isinstance(world.get("arena"), dict) else {}
        bounds = arena.get("bounds", arena)
        if not isinstance(bounds, dict):
            raise ValueError("CANCEL requires known arena bounds")
        x_min = _finite(bounds.get("x_min"), 0.0) + self.robot_margin_m
        x_max = _finite(bounds.get("x_max"), 3.0) - self.robot_margin_m
        y_min = _finite(bounds.get("y_min"), 0.0) + self.robot_margin_m
        y_max = _finite(bounds.get("y_max"), 2.0) - self.robot_margin_m
        yaw = float(pose["yaw"])
        distance = 0.02
        candidates = (1.0, -1.0)
        cancel_target = None
        for direction in candidates:
            x = float(pose["x"]) + direction * distance * math.cos(yaw)
            y = float(pose["y"]) + direction * distance * math.sin(yaw)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                cancel_target = {"x": x, "y": y, "sin": math.sin(yaw), "cos": math.cos(yaw)}
                break
        if cancel_target is None:
            raise ValueError("CANCEL cannot find a safe short replacement target inside the map")
        self._publish_goal(cancel_target)
        return cancel_target

    def _update_move(self, operation: dict[str, Any], world: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - operation["started_at"] > self.navigation_timeout_s and operation["status"] != "unknown":
            operation["status"] = "unknown"
            operation["cancelable"] = False
            operation["reason"] = "move timed out; /goal_pose has no terminal result interface"
        robot = world.get("robot") if isinstance(world.get("robot"), dict) else {}
        pose = robot.get("pose")
        sources = world.get("sources") if isinstance(world.get("sources"), dict) else {}
        pose_source = sources.get("robot_pose") if isinstance(sources.get("robot_pose"), dict) else {}
        velocity_source = sources.get("velocity") if isinstance(sources.get("velocity"), dict) else {}
        velocity = robot.get("velocity")
        sequence = int(world.get("sequence", -1))
        pose_age = _finite(robot.get("pose_age_s"), math.inf)
        velocity_age = _finite(velocity_source.get("age_s"), math.inf)
        if (sequence <= operation["started_sequence"] or not isinstance(pose, dict)
                or pose_source.get("fresh", True) is not True or pose_age > self.pose_freshness_s
                or not isinstance(velocity, dict) or velocity_source.get("fresh") is not True
                or velocity_age > 0.5):
            operation["settled_since"] = None
            operation["settled_updates"] = 0
            return
        target = operation.get("cancel_target_pose") or operation["target_pose"]
        try:
            position_error = math.hypot(float(target["x"]) - float(pose["x"]),
                                        float(target["y"]) - float(pose["y"]))
            target_yaw = math.atan2(float(target["sin"]), float(target["cos"]))
            heading_error = abs(_wrap_angle(target_yaw - float(pose["yaw"])))
            vx = _finite(velocity.get("x"), math.inf)
            vy = _finite(velocity.get("y"), math.inf)
            omega = _finite(velocity.get("yaw"), math.inf)
        except (KeyError, TypeError, ValueError):
            operation["settled_since"] = None
            operation["settled_updates"] = 0
            return
        stopped = math.hypot(vx, vy) <= 0.02 and abs(omega) <= 0.05
        if position_error <= 0.04 and heading_error <= 0.10 and stopped:
            if operation.get("settled_since") is None:
                operation["settled_since"] = now
                operation["settled_updates"] = 0
            operation["settled_updates"] += 1
            if operation["settled_updates"] >= 5 and now - operation["settled_since"] >= 0.4:
                canceled = operation.get("cancel_target_pose") is not None
                self._finish("canceled" if canceled else "succeeded",
                             "replacement goal reached and robot stopped" if canceled else
                             "fresh pose reached target and robot stopped", world)
        else:
            operation["settled_since"] = None
            operation["settled_updates"] = 0

    async def _run_service(self, operation: dict[str, Any], decision: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        from std_srvs.srv import SetBool

        action = operation["type"]
        if action.startswith("SUCTION_"):
            index, state = action.removeprefix("SUCTION_").split("_")
            client = self._suction_clients[int(index)]
            request = SetBool.Request()
            request.data = state == "ON"
        else:
            client = self._wrist_client
            current = _finite(world.get("robot", {}).get("wrist_angle_rad"), math.nan)
            if not math.isfinite(current):
                return {"status": "failed", "reason": "wrist angle is unknown"}
            request = SetBool.Request()
            request.data = current < math.pi / 4.0
        if not client.service_is_ready():
            return {"status": "failed", "reason": f"service for {action} is unavailable"}
        try:
            operation["terminal_kind"] = "service"
            operation["terminal_future"] = client.call_async(request)
            response = await _wait_ros_future(operation["terminal_future"], self.command_timeout_s)
        except asyncio.TimeoutError:
            return {"status": "unknown", "reason": f"{action} service timed out; physical outcome is uncertain"}
        return {"status": "succeeded" if response.success else "failed",
                "reason": response.message or ("service succeeded" if response.success else "service reported failure"),
                "message": response.message}

    def _finish(self, status: str, reason: str, world: dict[str, Any], message: str = "") -> None:
        operation = self.active
        if operation is None:
            return
        self.last_result = {
            "id": operation["id"], "type": operation["type"], "status": status,
            "failed": status in ("failed", "rejected"), "reason": reason,
            "cancelable": operation["cancelable"], "target_pose": operation.get("target_pose"),
            "message": message, "finished_sequence": int(world.get("sequence", 0)),
        }
        self.active = None

    async def close(self) -> None:
        operation = self.active
        if operation is not None and operation.get("type") == "MOVE":
            try:
                if self._last_world is not None:
                    self._publish_cancel_goal(self._last_world)
            except Exception:
                pass
        if self.active is not None:
            operation = self.active
            task = operation.get("task")
            if task is not None and task.done():
                try:
                    result = task.result()
                except Exception as error:
                    result = {"status": "unknown", "reason": f"ROS operation outcome uncertain: {error}"}
                if result["status"] != "unknown":
                    self._finish(result["status"], result["reason"], {"sequence": operation["started_sequence"]}, result.get("message", ""))
            if self.active is not None:
                self.active["status"] = "unknown"
                self.active["reason"] = "runner stopped; remote operation may still be active"
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
            self._executor = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


async def _wait_ros_future(future: Any, timeout_s: float) -> Any:
    started = time.monotonic()
    while not future.done():
        if time.monotonic() - started > timeout_s:
            raise asyncio.TimeoutError
        await asyncio.sleep(0.01)
    return future.result()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
