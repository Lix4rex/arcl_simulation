"""NDJSON bridge between world snapshots, policy inference and ROS subprocesses."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

from .action_executor import ActionExecutor
from .policy import DEFAULT_CHECKPOINT_PATH, AgentPolicy
from ..records import EpisodeRecords
from ..rl.environment import winner_from_world, winner_targets


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _match_finished(world: dict[str, Any]) -> bool:
    match = world.get("match") if isinstance(world.get("match"), dict) else {}
    mission = world.get("mission") if isinstance(world.get("mission"), dict) else {}
    remaining = mission.get("remaining_s")
    return match.get("finished") is True or world.get("match_finished") is True or (
        isinstance(remaining, (int, float)) and not isinstance(remaining, bool)
        and math.isfinite(remaining) and remaining <= 0
    )


def _outcome(world: dict[str, Any], robot_id: str, opponent_id: str | None) -> tuple[str | None, dict | None]:
    match = world.get("match") if isinstance(world.get("match"), dict) else {}
    scoreboard = world.get("scoreboard") if isinstance(world.get("scoreboard"), dict) else {}
    if opponent_id == robot_id:
        raise ValueError("opponent ID must differ from the controlled robot ID")
    if opponent_id is None:
        explicit = match.get("winner_id", world.get("winner_id"))
        opponent_id = explicit if isinstance(explicit, str) and explicit not in (robot_id, "draw", "tie") else "opponent"
        for scores in (match.get("scores"), scoreboard.get("scores"), world.get("scores")):
            if isinstance(scores, dict) and robot_id in scores and len(scores) == 2:
                opponent_id = next(key for key in scores if key != robot_id)
                break
    winner, known = winner_from_world(world, (robot_id, opponent_id))
    targets = winner_targets(winner, (robot_id, opponent_id), result_known=known)
    return winner, None if targets is None else {robot_id: targets[robot_id]}


async def _run(args: argparse.Namespace) -> int:
    policy = AgentPolicy(device=args.device)
    policy.load_or_create_checkpoint(args.model)

    executor = ActionExecutor(
        ros_setup=args.ros_setup,
        command_timeout_s=args.command_timeout,
        navigation_timeout_s=args.navigation_timeout,
        pose_freshness_s=args.pose_freshness,
    )
    last_sequence: int | None = None
    last_timestamp: float | None = None
    records: EpisodeRecords | None = None
    robot_id = "robot_omni"
    last_world: dict[str, Any] = {}
    complete = False

    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if line == "":
                break
            if not line.strip():
                continue
            try:
                world = json.loads(line)
                if not isinstance(world, dict):
                    raise ValueError("each input line must be a JSON object")
                sequence = int(world.get("sequence", -1))
                if last_sequence is not None and sequence <= last_sequence:
                    raise ValueError("world sequence must increase for every snapshot")
                timestamp = float(world.get("timestamp", math.nan))
                if not math.isfinite(timestamp):
                    raise ValueError("world timestamp must be finite seconds")
                if last_timestamp is not None and timestamp < last_timestamp:
                    raise ValueError("world timestamp must not move backwards")
                if records is None:
                    robot_id = str((world.get("robot") or {}).get("id", "robot_omni"))
                    records = EpisodeRecords(None if args.no_records else args.records, str(uuid.uuid4()), (robot_id,))
                    records.add({"kind": "reset", "observation": world})
                last_sequence = sequence
                last_timestamp = timestamp
                last_world = copy.deepcopy(world)
                await executor.update(world)
                if _match_finished(world):
                    complete = True
                    break
                world = copy.deepcopy(world)
                local_operation = executor.observation()
                if local_operation is not None:
                    world["active_operation"] = local_operation
                if executor.last_result is not None:
                    world["last_action_result"] = executor.last_result

                decision = policy.decide(world, deterministic=not args.sample_actions)
                execution = {"status": "dry_run"} if args.dry_run else await executor.dispatch(decision, world)
                records.add({"kind": "decision", "agent_observations": {robot_id: world},
                             "actions": {robot_id: decision}, "responses": {robot_id: execution},
                             "active_operation": executor.observation(), "last_action_result": executor.last_result})
                _emit({
                    "world_sequence": world.get("sequence"),
                    "decision": decision,
                    "execution": execution,
                    "active_operation": executor.observation(),
                    "last_action_result": executor.last_result,
                })
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                _emit({"error": str(error)})
    finally:
        await executor.close()
        if records is not None:
            winner, targets = _outcome(last_world, robot_id, args.opponent_id) if complete else (None, None)
            records.add({"kind": "runner_end", "final_observation": last_world,
                         "active_operation": executor.observation(), "last_action_result": executor.last_result})
            records.finish(complete=complete, winner_id=winner, winner_targets=targets,
                           rewards_complete=False, reason="match finished" if complete else "input interrupted or ended before match result")
            _emit({"kind": "episode_end", "episode_id": records.episode_id, "complete": complete,
                   "winner_targets": targets, "records_path": str(records.path) if records.path else None})
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the world policy over newline-delimited JSON snapshots."
    )
    parser.add_argument(
        "--model", "--weights", dest="model", type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="checkpoint to load or create (default: %(default)s)",
    )
    parser.add_argument("--device", default="cpu", help="torch device, default: cpu")
    parser.add_argument(
        "--ros-setup",
        help="ROS workspace setup.bash sourced inside each command subprocess "
        "(default: <repository>/install/setup.bash)",
    )
    parser.add_argument("--command-timeout", type=float, default=35.0)
    parser.add_argument("--navigation-timeout", type=float, default=90.0)
    parser.add_argument("--pose-freshness", type=float, default=0.75)
    parser.add_argument("--records", type=Path, default=Path("runs/agent"), help="episode directory or JSONL file")
    parser.add_argument("--no-records", action="store_true")
    parser.add_argument("--opponent-id", help="opponent ID for final score comparison, otherwise inferred")
    parser.add_argument("--sample-actions", action="store_true", help="sample actions/targets instead of argmax")
    parser.add_argument("--dry-run", action="store_true", help="record decisions without issuing ROS commands")
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"agent: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
