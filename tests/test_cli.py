"""Command-line interface tests."""

from __future__ import annotations

from pathlib import Path

from lightning_frame_stack.cli import run

from conftest import make_test_gif


def test_cli_end_to_end(tmp_path: Path) -> None:
    input_path = tmp_path / "storm.gif"
    output_path = tmp_path / "stacked.png"
    make_test_gif(input_path)

    exit_code = run(
        [
            str(input_path),
            "-o",
            str(output_path),
            "--base-frame",
            "0",
            "--align",
            "none",
            "--no-exposure-match",
            "--threshold",
            "2",
            "--min-difference",
            "1",
            "--min-brightness",
            "35",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()


def test_cli_returns_error_for_missing_input(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.mp4"

    exit_code = run([str(missing), "--quiet"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Input file does not exist" in captured.err
