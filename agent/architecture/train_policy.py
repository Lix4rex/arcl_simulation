"""Compatibility entry point for the shared prepared-target trainer."""

from .train_targets import main, read_target_datasets, split_groups, target_loss, train


if __name__ == "__main__":
    main()
