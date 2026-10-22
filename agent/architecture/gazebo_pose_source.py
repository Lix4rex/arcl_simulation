"""Carton-only reader for Gazebo Harmonic's SceneBroadcaster Pose_V topic.

This module deliberately extracts only model poses whose names start with
``pierre_``. It never exposes robot or other simulator poses to observations.
"""

from __future__ import annotations

import copy
import math
import re
import subprocess
import threading
import time
from typing import Any, Protocol


_POSE_START = re.compile(r"\bpose\s*\{")
_FIELD = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([-+0-9.eE]+)")
_NAME = re.compile(r'\bname\s*:\s*"((?:\\.|[^"\\])*)"')


class CartonPoseProvider(Protocol):
    """Internal provider contract shared by simulator and future camera input."""

    provider: str

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _balanced_end(text: str, opening_brace: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening_brace, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _fields(block: str, field_name: str, names: tuple[str, ...]) -> dict[str, float]:
    match = re.search(rf"\b{re.escape(field_name)}\s*\{{([^{{}}]*)\}}", block, re.DOTALL)
    if match is None:
        raise ValueError(f"Pose_V record is missing {field_name}")
    values = {name: float(value) for name, value in _FIELD.findall(match.group(1))}
    # Gazebo's protobuf text output omits scalar fields that equal their
    # default (zero), so omitted coordinates/quaternion components are zero.
    result = {name: values.get(name, 0.0) for name in names}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"Pose_V {field_name} contains a non-finite value")
    if field_name == "orientation" and not any(result.values()):
        raise ValueError("Pose_V orientation is a zero quaternion")
    return result


def parse_pose_v_text(text: str) -> dict[str, dict[str, Any]]:
    """Parse Gazebo protobuf text and return only named carton model poses.

    Model IDs are the stable SDF instance names (for example
    ``pierre_centre_sud_1``). Scoped link names are accepted but deduplicated in
    favor of an exact model-name record when both are present.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty Pose_V text")

    records: dict[str, tuple[int, dict[str, Any]]] = {}
    cursor = 0
    found_pose = False
    while True:
        match = _POSE_START.search(text, cursor)
        if match is None:
            break
        found_pose = True
        opening = text.find("{", match.start(), match.end())
        end = _balanced_end(text, opening)
        if end is None:
            raise ValueError("incomplete Pose_V pose block")
        block = text[match.start():end]
        name_match = _NAME.search(block)
        if name_match is None:
            raise ValueError("Pose_V pose block is missing name")
        scoped_name = name_match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        model_name = scoped_name.split("::", 1)[0]
        if model_name.startswith("pierre_") and model_name != "pierre_camelot":
            position = _fields(block, "position", ("x", "y", "z"))
            orientation = _fields(block, "orientation", ("x", "y", "z", "w"))
            rank = 0 if scoped_name == model_name else 1
            record = {
                "id": model_name,
                "model_name": model_name,
                "position": position,
                "orientation": orientation,
            }
            previous = records.get(model_name)
            if previous is None or rank < previous[0]:
                records[model_name] = (rank, record)
        cursor = end
    if not found_pose:
        raise ValueError("no Pose_V pose blocks found")
    return {key: value[1] for key, value in records.items()}


class PoseVTextParser:
    """Incremental parser for the line-oriented output of ``gz topic -e``."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> list[dict[str, Any]]:
        self._buffer += chunk
        parsed: list[dict[str, Any]] = []
        while True:
            match = _POSE_START.search(self._buffer)
            if match is None:
                self._buffer = self._buffer[-16:]
                break
            opening = self._buffer.find("{", match.start(), match.end())
            end = _balanced_end(self._buffer, opening)
            if end is None:
                self._buffer = self._buffer[match.start():]
                break
            block_text = self._buffer[match.start():end]
            parsed.extend(parse_pose_v_text(block_text).values())
            self._buffer = self._buffer[end:]
        return parsed


class GazeboCliPoseSource:
    """Keep the newest carton poses from one explicit ``gz topic`` process."""

    provider = "gazebo_sim"

    def __init__(
        self,
        *,
        topic: str = "/world/arene_omni/pose/info",
        stale_after_s: float = 1.0,
    ) -> None:
        self.topic = topic
        self.stale_after_s = float(stale_after_s)
        self._lock = threading.Lock()
        self._poses: dict[str, dict[str, Any]] = {}
        self._last_update: dict[str, float] = {}
        self._last_stream_update: float | None = None
        self._last_error: str | None = None
        self._stderr_tail: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                ["gz", "topic", "-e", "-t", self.topic],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            with self._lock:
                self._last_error = f"could not start gz topic: {error}"
            raise RuntimeError(self._last_error) from error

        stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._threads = [stdout_thread, stderr_thread]
        stdout_thread.start()
        stderr_thread.start()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        parser = PoseVTextParser()
        try:
            for line in self._process.stdout:
                if self._stop.is_set():
                    return
                for pose in parser.feed(line):
                    now = time.monotonic()
                    with self._lock:
                        self._poses[pose["id"]] = pose
                        self._last_update[pose["id"]] = now
                        self._last_stream_update = now
        except (OSError, ValueError) as error:
            with self._lock:
                self._last_error = f"invalid Gazebo Pose_V output: {error}"
        finally:
            if not self._stop.is_set():
                return_code = self._process.poll()
                with self._lock:
                    if self._last_error is None:
                        suffix = "" if return_code is None else f" (exit code {return_code})"
                        detail = f": {self._stderr_tail}" if self._stderr_tail else ""
                        self._last_error = f"Gazebo Pose_V stream ended{suffix}{detail}"

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            if line.strip():
                with self._lock:
                    self._stderr_tail = line.strip()[-500:]

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            poses = copy.deepcopy(self._poses)
            ages = {
                carton_id: max(0.0, current - received_at)
                for carton_id, received_at in self._last_update.items()
            }
            stream_age = (
                None if self._last_stream_update is None
                else max(0.0, current - self._last_stream_update)
            )
            error = self._last_error
        fresh = error is None and stream_age is not None and stream_age <= self.stale_after_s
        return {
            "provider": self.provider,
            "status": "fresh" if fresh else ("error" if error else "stale"),
            "fresh": fresh,
            "age_s": stream_age,
            "error": error,
            "poses": poses if fresh else {},
            "carton_ages_s": ages,
        }

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for thread in self._threads:
            thread.join(timeout=0.2)
        self._process = None

