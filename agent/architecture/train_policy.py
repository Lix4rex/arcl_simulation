"""PPO updates from finalized, rewarded episodes; split by entire episode."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .policy import DEFAULT_CHECKPOINT_PATH, AgentPolicy


def read_episodes(paths: list[Path], *, robot_id: str | None = None) -> tuple[dict[str, list[dict]], dict[str, int]]:
    files: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise ValueError(f"dataset path does not exist: {path}")
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.jsonl"))
        else:
            files.add(path.resolve())
    groups: dict[str, list[dict]] = {}
    locations: dict[str, Path] = {}
    for path in sorted(files):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(row, dict) or not isinstance(row.get("episode_id"), str):
                    raise ValueError(f"{path}:{line_number}: missing episode_id")
                identifier = row["episode_id"]
                if identifier in locations and locations[identifier] != path:
                    raise ValueError(f"duplicate episode {identifier!r} in multiple files")
                locations[identifier] = path
                groups.setdefault(identifier, []).append(row)
    episodes: dict[str, list[dict]] = {}
    skipped = {"incomplete": 0, "pending_rewards": 0, "no_sampled_decisions": 0}
    for identifier, rows in groups.items():
        ends = [row for row in rows if row.get("kind") == "episode_end"]
        if len(ends) > 1:
            raise ValueError(f"duplicate finalization for episode {identifier}")
        if not ends or ends[0].get("complete") is not True:
            skipped["incomplete"] += 1
            continue
        if ends[0].get("rewards_complete") is not True:
            skipped["pending_rewards"] += 1
            continue
        samples = []
        for row in rows:
            if row.get("kind") != "transition":
                continue
            observations = row.get("agent_observations")
            targets = row.get("return_targets")
            if not isinstance(observations, dict) or not isinstance(targets, dict):
                raise ValueError(f"episode {identifier}: missing policy observations or finalized returns")
            for robot, decision in row.get("actions", {}).items():
                if robot_id is not None and robot != robot_id:
                    continue
                behavior = decision.get("behavior") if isinstance(decision, dict) else None
                if not isinstance(behavior, dict) or behavior.get("sampled") is not True:
                    continue
                observation = observations.get(robot)
                values = (behavior.get("log_prob"), behavior.get("value"), targets.get(robot))
                if not isinstance(observation, dict) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in values
                ) or values[0] > 1e-6:
                    raise ValueError(f"episode {identifier}: invalid behavior/return for {robot}")
                samples.append({"observation": observation, "decision": decision,
                                "old_log_prob": float(values[0]), "old_value": float(values[1]),
                                "return_target": float(values[2]), "robot_id": robot})
        if samples:
            episodes[identifier] = samples
        else:
            skipped["no_sampled_decisions"] += 1
    if not episodes:
        raise ValueError(f"no trainable episodes: {skipped}; finish games, define both reward functions and sample actions")
    return episodes, skipped


def split_episodes(episodes: dict[str, list[dict]], fraction: float, seed: int) -> tuple[dict, dict]:
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be strictly between zero and one")
    if len(episodes) < 2:
        raise ValueError("at least two complete episodes are needed for train/validation")
    identifiers = sorted(episodes)
    random.Random(seed).shuffle(identifiers)
    count = min(len(identifiers) - 1, max(1, round(len(identifiers) * fraction)))
    validation = {identifier: episodes[identifier] for identifier in identifiers[:count]}
    training = {identifier: episodes[identifier] for identifier in identifiers[count:]}
    return training, validation


def ppo_loss(policy: AgentPolicy, sample: dict, *, advantage_mean: float = 0.0,
             advantage_scale: float = 1.0, clip: float = 0.2,
             value_weight: float = 0.5, entropy_weight: float = 0.01) -> tuple[torch.Tensor, dict[str, float]]:
    log_prob, entropy, value = policy.evaluate_action(sample["observation"], sample["decision"])
    advantage = (sample["return_target"] - sample["old_value"] - advantage_mean) / advantage_scale
    ratio = (log_prob - sample["old_log_prob"]).exp()
    policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    value_loss = (value - sample["return_target"]).square()
    loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
    if not torch.isfinite(loss):
        raise ValueError("non-finite PPO loss")
    return loss, {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                  "value_mse": float(value_loss.detach()), "entropy": float(entropy.detach())}


def _flatten(episodes: dict[str, list[dict]]) -> list[dict]:
    return [sample for samples in episodes.values() for sample in samples]


def _evaluate(policy: AgentPolicy, samples: list[dict], options: dict) -> dict[str, float]:
    policy.network.eval()
    totals: dict[str, float] = {}
    with torch.no_grad():
        for sample in samples:
            _, metrics = ppo_loss(policy, sample, **options)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
    return {key: value / len(samples) for key, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    episodes, skipped = read_episodes(args.datasets, robot_id=args.robot_id)
    if args.validation_data:
        validation, _ = read_episodes(args.validation_data, robot_id=args.robot_id)
        training = episodes
    else:
        training, validation = split_episodes(episodes, args.validation_fraction, args.seed)
    test, _ = read_episodes(args.test_data, robot_id=args.robot_id) if args.test_data else ({}, {})
    if set(training) & set(validation) or set(test) & (set(training) | set(validation)):
        raise ValueError("train, validation and test must contain disjoint episode IDs")
    if not args.model.is_file():
        raise ValueError("the collection checkpoint is required (--model)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = AgentPolicy(device=args.device)
    policy.load_checkpoint(args.model)
    train_samples, validation_samples = _flatten(training), _flatten(validation)
    advantages = np.asarray([sample["return_target"] - sample["old_value"] for sample in train_samples])
    deviation = float(advantages.std())
    options = {"advantage_mean": float(advantages.mean()) if deviation > 1e-8 else 0.0,
               "advantage_scale": deviation if deviation > 1e-8 else 1.0,
               "clip": args.clip, "value_weight": args.value_weight, "entropy_weight": args.entropy_weight}
    # Validate all samples/masks before writing a checkpoint or taking a gradient step.
    baseline = _evaluate(policy, validation_samples, options)
    _evaluate(policy, train_samples, options)
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.learning_rate)
    report: dict[str, Any] = {
        "train_episode_ids": sorted(training), "validation_episode_ids": sorted(validation),
        "test_episode_ids": sorted(test), "train_samples": len(train_samples),
        "validation_samples": len(validation_samples), "skipped": skipped,
        "seed": args.seed, "baseline_validation": baseline, "epochs": [],
        "configuration": {"collection_model": str(args.model), "learning_rate": args.learning_rate,
                          "batch_size": args.batch_size, "epochs": args.epochs, "device": args.device, **options},
    }
    best = baseline["loss"]
    report["best_epoch"] = 0
    policy.save_checkpoint(args.output)
    for epoch in range(1, args.epochs + 1):
        policy.network.train()
        random.shuffle(train_samples)
        for start in range(0, len(train_samples), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = train_samples[start:start + args.batch_size]
            loss = torch.stack([ppo_loss(policy, sample, **options)[0] for sample in batch]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.network.parameters(), 0.5)
            optimizer.step()
        metrics = {"epoch": epoch, "train": _evaluate(policy, train_samples, options),
                   "validation": _evaluate(policy, validation_samples, options)}
        report["epochs"].append(metrics)
        print(json.dumps(metrics, separators=(",", ":")), flush=True)
        if metrics["validation"]["loss"] < best:
            best = metrics["validation"]["loss"]
            report["best_epoch"] = epoch
            policy.save_checkpoint(args.output)
    policy.load_checkpoint(args.output)
    if test:
        report["test"] = _evaluate(policy, _flatten(test), options)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="JSONL files or episode directories")
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="collection checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--validation-data", nargs="+", type=Path)
    parser.add_argument("--test-data", nargs="+", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--robot-id", help="train only this robot's decisions (default: both)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    if not 0 < args.clip < 1 or any(not math.isfinite(value) or value < 0 for value in (args.value_weight, args.entropy_weight)):
        parser.error("invalid PPO loss coefficients")
    try:
        report = train(args)
        print(f"saved best checkpoint: {args.output} (epoch {report['best_epoch']})")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"train_policy: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
