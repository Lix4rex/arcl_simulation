"""Collect the scripted strategy as canonical supervised demonstrations."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .actions import ACTIONS
from .demonstrations import DemonstrationRecorder, operator_choice_to_decision
from .targets import DEFAULT_TARGET_DATASET_PATH


class StrategyRecordingSession:
    """Translate strategy lifecycle events and observations into target rows."""

    def __init__(self, destination: str | Path, *, session_id: str | None = None) -> None:
        self.recorder = DemonstrationRecorder(
            destination, session_id=session_id, source="strategy"
        )
        self._lock = threading.RLock()
        self._latest_observation: dict[str, Any] | None = None
        self._active_operation: dict[str, Any] | None = None
        self._last_action_result: dict[str, Any] | None = None
        self._decision_ids: dict[str, str] = {}
        self._last_idle_at: float | None = None
        self._closed = False

    @property
    def path(self) -> Path:
        return self.recorder.path

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._latest_observation is not None

    def update_observation(self, observation: dict[str, Any]) -> None:
        if not isinstance(observation, dict) or observation.get("schema_version") != 2:
            return
        with self._lock:
            self._latest_observation = copy.deepcopy(observation)

    def handle_event(self, event: dict[str, Any], *, now: float | None = None) -> bool:
        if not isinstance(event, dict):
            return False
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._closed:
                return False
            kind = event.get("event")
            if kind == "start":
                return self._start(event, now)
            if kind == "result":
                self._result(event)
                return True
            if kind == "finish":
                self.close(reason="strategy finished")
                return True
        return False

    def _observation(self) -> dict[str, Any] | None:
        if self._latest_observation is None:
            return None
        result = copy.deepcopy(self._latest_observation)
        result["active_operation"] = copy.deepcopy(self._active_operation)
        result["last_action_result"] = copy.deepcopy(self._last_action_result)
        return result

    def _start(self, event: dict[str, Any], now: float) -> bool:
        operation_id = str(event.get("id", ""))
        command = event.get("command")
        parameters = event.get("parameters", {})
        observation = self._observation()
        recorded = False
        decision: dict[str, Any] | None = None
        if (operation_id and isinstance(command, str) and isinstance(parameters, dict)
                and observation is not None and command != "rock_centering"):
            decision = operator_choice_to_decision(command, parameters, observation)
            if decision.get("action") in ACTIONS:
                decision_id = self.recorder.record_decision(
                    observation,
                    decision,
                    command=command,
                    parameters=parameters,
                )
                self._decision_ids[operation_id] = decision_id
                recorded = True

        operation_type = event.get("operation_type")
        if not isinstance(operation_type, str):
            operation_type = "UNKNOWN"
        self._active_operation = {
            "id": operation_id or None,
            "type": operation_type,
            "status": "running",
            "cancelable": event.get("cancelable") is True,
            "failed": False,
            "target_pose": copy.deepcopy(decision.get("target_pose"))
                if observation is not None and command != "rock_centering"
                and isinstance(decision, dict) else None,
        }
        self._last_idle_at = now
        return recorded or command == "rock_centering"

    def _result(self, event: dict[str, Any]) -> None:
        operation_id = str(event.get("id", ""))
        status = event.get("status")
        if not isinstance(status, str):
            status = "unknown"
        active = self._active_operation
        operation_type = active.get("type") if active and active.get("id") == operation_id else "UNKNOWN"
        target_pose = active.get("target_pose") if active and active.get("id") == operation_id else None
        decision_id = self._decision_ids.pop(operation_id, None)
        if decision_id is not None:
            self.recorder.record_result(decision_id, {
                "status": status,
                "failed": status in ("failed", "rejected", "canceled", "unknown"),
            })
        self._last_action_result = {
            "id": operation_id or None,
            "type": operation_type,
            "status": status,
            "failed": status in ("failed", "rejected", "canceled", "unknown"),
            "cancelable": False,
            "target_pose": copy.deepcopy(target_pose),
        }
        if active and active.get("id") == operation_id:
            self._active_operation = None
        self._last_idle_at = None

    def sample_idle(self, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            operation = self._active_operation
            if (self._closed or operation is None or operation.get("type") == "CENTRAGE"
                    or self._last_idle_at is None or now - self._last_idle_at < 0.1 - 1e-9):
                return False
            observation = self._observation()
            if observation is None:
                return False
            self.recorder.record_decision(
                observation,
                {"action": "IDLE"},
                command="idle",
                requires_result=False,
            )
            self._last_idle_at = now
            return True

    def close(self, *, reason: str = "strategy recorder stopped") -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.recorder.abort_pending(reason=reason)
            self.recorder.stop(reason=reason)


class _WorldObserverProcess:
    def __init__(self, session: StrategyRecordingSession, workspace: Path, carton_source: str) -> None:
        self.session = session
        self.workspace = workspace
        self.carton_source = carton_source
        self.process: subprocess.Popen[str] | None = None
        self.threads: list[threading.Thread] = []
        self.stopping = threading.Event()

    def start(self) -> None:
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(self.workspace) + (
            os.pathsep + python_path if python_path else ""
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "agent.architecture.world_observer",
             "--carton-source", self.carton_source, "--publish-hz", "10"],
            cwd=self.workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.threads = [
            threading.Thread(target=self._read_snapshots, daemon=True),
            threading.Thread(target=self._read_errors, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _read_snapshots(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            try:
                observation = json.loads(line)
            except json.JSONDecodeError:
                print("strategy recorder: invalid observer JSON", file=sys.stderr, flush=True)
                continue
            if isinstance(observation, dict):
                was_ready = self.session.ready
                self.session.update_observation(observation)
                if not was_ready and self.session.ready:
                    print("strategy recorder: first observation received", flush=True)

    def _read_errors(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        for line in self.process.stderr:
            message = line.strip()
            if message:
                print(message, file=sys.stderr, flush=True)

    def stop(self) -> None:
        if self.process is None:
            return
        self.stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_TARGET_DATASET_PATH)
    parser.add_argument("--carton-source", default="gazebo_sim")
    parser.add_argument("--event-topic", default="/agent/strategy_events")
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as error:
        parser.error(f"ROS 2 Python libraries are unavailable: {error}")

    workspace = Path(__file__).resolve().parents[2]
    session = StrategyRecordingSession(args.records)
    observer = _WorldObserverProcess(session, workspace, args.carton_source)

    class RecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("strategy_demonstration_recorder")
            self.create_subscription(String, args.event_topic, self.on_event, 10)
            self.create_timer(0.1, session.sample_idle)

        def on_event(self, message: Any) -> None:
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError) as error:
                self.get_logger().warning(f"invalid strategy event: {error}")
                return
            if not session.handle_event(event):
                self.get_logger().warning(f"ignored strategy event: {event!r}")

    rclpy.init()
    node = RecorderNode()
    try:
        observer.start()
        node.get_logger().info(
            f"recording strategy events on {args.event_topic}; waiting for strategy completion"
        )
        while rclpy.ok() and not session._closed:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        observer.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if session.recorder.published_samples:
            print(
                f"strategy recorder: appended {session.recorder.published_samples} targets to "
                f"{session.path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
