"""Compatibility launcher for the renamed model-agnostic MuJoCo player."""

try:
    from scripts.mujoco_player import main
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from mujoco_player import main


if __name__ == "__main__":
    main()
