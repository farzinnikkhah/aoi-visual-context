"""Locations for code and local study data.

This repository contains the computer-vision code only. Frames, gaze tables,
annotations, and model weights stay outside git.

AOI_DATA_ROOT
    Directory that holds the full-frame pipeline data (frames, detection
    CSVs, annotation folders, fine-tune runs). Defaults to this repository.

AOI_STUDY_ROOT
    Directory that holds the original Tobii exports, used when merging
    detections back onto those tables. Defaults to the parent of AOI_DATA_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AOI_DATA_ROOT", CODE_ROOT)).expanduser().resolve()
STUDY_ROOT = Path(os.environ.get("AOI_STUDY_ROOT", DATA_ROOT.parent)).expanduser().resolve()


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def study_path(*parts: str) -> Path:
    return STUDY_ROOT.joinpath(*parts)
