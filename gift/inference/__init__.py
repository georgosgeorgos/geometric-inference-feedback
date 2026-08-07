import re
import subprocess
import os
import asyncio
import shutil
from typing import List, Tuple

PATH_TO_CAD_PYTHON = os.environ.get(
    "CAD_PYTHON_PATH",
    shutil.which("python") or "/workspace/home/lab/gcg/dev/.cad/bin/python"
)

TEST_TEMPLATE = """
{code}
cq.exporters.export(solid, {file_name})
"""

OCC_IOU_TEMPLATE = """
from gift.inference.geom import align_shapes
import cadquery as cq

solid_gt = cq.importers.importStep("{ground_truth_step_path}")
solid_gen = cq.importers.importStep("{generated_step_path}")

IOU = align_shapes(solid_gen, solid_gt)[1]
"""


from gift.data.utils import extract_code  # noqa: F401


def run_code(code, verbose=False):
    # Given CadQuery code, will return whether it ran successfully or not
    try:
        result = subprocess.run(
            [
                PATH_TO_CAD_PYTHON,
                "-c",
                code,
            ],  # TODO: pass conda environment in as argument?
            capture_output=True,
            text=True,
            timeout=60,  # if code takes more than 1 minute to run, log as failure
        )

        # Print stderr if there are errors and verbose mode is enabled
        if result.returncode != 0 and verbose:
            print(f"[ERROR] Code execution failed with return code {result.returncode}")
            if result.stderr:
                print(f"[STDERR] {result.stderr}")
            if result.stdout:
                print(f"[STDOUT] {result.stdout}")

        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Code execution exceeded for {code[:100]}...")
        return 1


def compute_iou(ground_truth_step_path, mg_step_path, timeout=60):
    # Computes IoU given two steps, returns 0 if error thrown or timeout reached
    code = f"""
import sys
import os
sys.path.insert(0, r"{os.getcwd()}")

import cadquery as cq
from gift.inference.geom import align_shapes
try:
    gt = cq.importers.importStep(r"{ground_truth_step_path}")
    mg = cq.importers.importStep(r"{mg_step_path}")
    _, iou = align_shapes(mg, gt)
    print(iou)
except Exception as e:
    print("[ERROR]", e, file=sys.stderr)
    print(0.0)
            """

    try:
        result = subprocess.run(
            [PATH_TO_CAD_PYTHON, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Debug: Print stderr if there are errors
        if result.stderr:
            print(f"[STDERR] {result.stderr}")

        # Parse last printed float as IoU from stdout
        stdout_lines = result.stdout.strip().splitlines()
        if not stdout_lines:
            print(
                f"[WARNING] No output from IoU computation for {ground_truth_step_path} and {mg_step_path}"
            )
            return 0.0

        for line in reversed(stdout_lines):
            try:
                iou_value = float(line)
                return iou_value
            except ValueError:
                continue

        print(f"[WARNING] Could not parse IoU from output: {result.stdout}")
        return 0.0
    except subprocess.TimeoutExpired:
        print(
            f"[TIMEOUT] IoU computation exceeded {timeout}s for {ground_truth_step_path} and {mg_step_path}"
        )
        return 0.0
