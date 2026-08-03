"""Command-line interface for lightning-frame-stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .core import LightningStackError, StackConfig, stack_media


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="lightning-stack",
        description="Align and stack lightning from a GIF or video into one image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input GIF, MP4, MOV, AVI, MKV, etc.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("lightning_stacked.png"),
        help="Output image path. PNG is recommended.",
    )
    parser.add_argument(
        "--mode",
        choices=("lightning", "max"),
        default="lightning",
        help=(
            "'lightning' preserves the base exposure and adds transient bright "
            "detail. 'max' performs an exposure-normalized lighten/max stack."
        ),
    )
    parser.add_argument(
        "--base-frame",
        type=int,
        default=None,
        help="Use this zero-based source-frame index as the background/base frame.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every Nth source frame.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds. Omit to process through the end.",
    )

    alignment = parser.add_argument_group("alignment")
    alignment.add_argument(
        "--align",
        choices=("none", "translation", "euclidean", "affine"),
        default="euclidean",
        help="Frame-alignment model. Euclidean allows translation and rotation.",
    )
    alignment.add_argument(
        "--align-scale",
        type=float,
        default=0.35,
        help="Resolution scale used while estimating alignment.",
    )
    alignment.add_argument(
        "--align-from",
        type=float,
        default=0.48,
        help=(
            "Fraction of image height above which alignment is ignored. 0.48 "
            "estimates motion from the lower 52%%, where static foreground objects "
            "usually are."
        ),
    )
    alignment.add_argument(
        "--min-alignment-score",
        type=float,
        default=0.50,
        help="Use identity alignment when the ECC score is lower.",
    )
    alignment.add_argument(
        "--no-exposure-match",
        action="store_true",
        help="Disable robust exposure matching.",
    )

    extraction = parser.add_argument_group("lightning extraction")
    extraction.add_argument(
        "--detail-sigma",
        type=float,
        default=4.0,
        help="Blur radius used to separate thin lightning from broad illumination.",
    )
    extraction.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Lower values retain dimmer branches but may retain more noise.",
    )
    extraction.add_argument(
        "--min-difference",
        type=float,
        default=2.5,
        help="Minimum positive brightness change from the base frame.",
    )
    extraction.add_argument(
        "--min-brightness",
        type=float,
        default=78.0,
        help="Minimum normalized grayscale value for a lightning seed pixel.",
    )
    extraction.add_argument(
        "--dilation",
        type=int,
        default=5,
        help="Mask expansion diameter in pixels. Even values are rounded up.",
    )
    extraction.add_argument(
        "--softness",
        type=float,
        default=8.0,
        help="Soft transition width above --threshold.",
    )
    extraction.add_argument(
        "--lightning-gain",
        type=float,
        default=1.0,
        help="Multiplier applied only to extracted positive lightning detail.",
    )
    extraction.add_argument(
        "--keep-straight-artifacts",
        action="store_true",
        help=(
            "Do not reject very narrow, nearly straight, long components. The "
            "default helps remove rolling-shutter and sensor-flare bars."
        ),
    )
    extraction.add_argument(
        "--mask-output",
        type=Path,
        default=None,
        help="Optional path for a grayscale image of the accumulated mask.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-style exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    config = StackConfig(
        mode=args.mode,
        base_frame=args.base_frame,
        every=args.every,
        start=args.start,
        end=args.end,
        align=args.align,
        align_scale=args.align_scale,
        align_from=args.align_from,
        min_alignment_score=args.min_alignment_score,
        exposure_match=not args.no_exposure_match,
        detail_sigma=args.detail_sigma,
        threshold=args.threshold,
        min_difference=args.min_difference,
        min_brightness=args.min_brightness,
        dilation=args.dilation,
        softness=args.softness,
        lightning_gain=args.lightning_gain,
        reject_straight_artifacts=not args.keep_straight_artifacts,
    )

    progress = None
    if not args.quiet:
        def report_progress(message: str) -> None:
            print(message, flush=True)

        progress = report_progress

    try:
        result = stack_media(
            args.input,
            args.output,
            config=config,
            mask_output_path=args.mask_output,
            progress=progress,
        )
    except LightningStackError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Saved: {result.output_path}")
        if result.mask_output_path is not None:
            print(f"Saved mask: {result.mask_output_path}")
        if result.failed_alignments:
            print(
                f"Note: {result.failed_alignments} frame(s) used identity alignment "
                "because ECC did not meet the minimum score."
            )
    return 0


def entrypoint() -> None:
    """Console-script entry point."""

    raise SystemExit(run())
