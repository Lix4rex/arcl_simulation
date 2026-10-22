"""Compatibility entry point for the shared prepared-target trainer."""

from .train_targets import (
    _class_weights,
    evaluate,
    main,
    read_target_datasets,
    split_groups,
    target_loss,
    train,
)

imitation_loss = target_loss
read_demonstrations = read_target_datasets
split_sessions = split_groups


if __name__ == "__main__":
    main()
