"""Backward-compatible thin entrypoint."""

from radio_cm_cutter.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
