import os
import pickle
import tempfile
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import tabulate
from contextlib import redirect_stdout, redirect_stderr
import sys
from contextlib import contextmanager
import json
import shutil

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

@contextmanager
def suppress_output_os():

    # Ensure this runs only on non-Windows platforms
    if sys.platform == "win32":
        yield
        return

    stdout_fd = 1
    stderr_fd = 2


    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    devnull_fd = os.open(os.devnull, os.O_RDWR)

    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

@contextmanager
def suppress_output():
    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        yield

def _run_gt_code(gt_code, gt_file_path, timeout=30):
    """Run ground truth CadQuery code to produce a STEP file. Returns status code (0=ok, 1=fail)."""
    try:
        gt_full_code = f"""
import sys
import os
sys.path.insert(0, r"{os.getcwd()}")
import cadquery as cq
{gt_code}
cq.exporters.export(solid, r"{gt_file_path}")
"""
        result = subprocess.run(
            [PATH_TO_CAD_PYTHON, "-c", gt_full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return 0 if result.returncode == 0 else 1
    except Exception:
        return 1


def _run_gen_code(gen_code, gen_file_path, timeout=30):
    """Run generated CadQuery code to produce a STEP file. Returns status code (0=ok, 2=fail)."""
    try:
        gen_full_code = f"""
import sys
import os
sys.path.insert(0, r"{os.getcwd()}")
import cadquery as cq
{gen_code}
cq.exporters.export(solid, r"{gen_file_path}")
"""
        result = subprocess.run(
            [PATH_TO_CAD_PYTHON, "-c", gen_full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return 0 if result.returncode == 0 else 2
    except Exception:
        return 2


def _compute_iou(gt_file_path, gen_file_path, timeout=30, iou_mode="align"):
    """Compute IOU between two STEP files. Returns (iou, status_code).

    iou_mode: 'align' (default), 'centering', or 'centering_normalize'.
    """
    _iou_imports = {
        "align": "from gift.inference.geom import align_shapes",
        "centering": "from gift.inference.geom import _compute_iou_centering",
        "centering_normalize": "from gift.inference.geom import _compute_iou_centering_normalize",
    }
    _iou_calls = {
        "align": "align_shapes(solid_gen, solid_gt)[1]",
        "centering": "_compute_iou_centering(solid_gen, solid_gt)",
        "centering_normalize": "_compute_iou_centering_normalize(solid_gen, solid_gt)",
    }
    try:
        iou_code = f"""
import sys
import os
sys.path.insert(0, r"{os.getcwd()}")
{_iou_imports[iou_mode]}
import cadquery as cq

solid_gt = cq.importers.importStep(r"{gt_file_path}")
solid_gen = cq.importers.importStep(r"{gen_file_path}")

IOU = {_iou_calls[iou_mode]}
print(IOU)
"""
        result = subprocess.run(
            [PATH_TO_CAD_PYTHON, "-c", iou_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return (-1, 3)
        try:
            iou = float(result.stdout.strip().split()[-1])
            return (iou, 0)
        except (ValueError, IndexError):
            return (-1, 3)
    except Exception:
        return (-1, 3)


class IOUProcessor:
    def __init__(self, num_workers=64, timeout=15, heal_failed=True, quiet=False, gt_is_step=False, checkpoint_path=None, checkpoint_interval=500, iou_mode="align"):
        self.num_workers = num_workers
        self.timeout = timeout
        self.heal_failed = heal_failed
        self.quiet = quiet
        self.gt_is_step = gt_is_step
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self.iou_mode = iou_mode

    def __call__(self, response):
        if isinstance(response, dict):
            return self._run(response, timeout=self.timeout, heal_failed=self.heal_failed, gt_is_step=self.gt_is_step, iou_mode=self.iou_mode)
        elif isinstance(response, list):
            return self._run_parallel(response)

    @staticmethod
    def _run(response, timeout=30, heal_failed=True, gt_is_step=False, iou_mode="align"):
        gt_data = response['ground_truth']
        gen_code = response['generated']
        gt_file = tempfile.NamedTemporaryFile(suffix=".step", delete=True)
        gen_file = tempfile.NamedTemporaryFile(suffix=".step", delete=True)

        # Per-subprocess timeout: split total budget across steps
        step_timeout = max(timeout // 2, 15)

        try:
            # Step 1: Produce ground truth STEP file
            if gt_is_step:
                try:
                    with open(gt_file.name, "wb") as f:
                        f.write(gt_data)
                except Exception:
                    return (-1, 1)
            else:
                status = _run_gt_code(gt_data, gt_file.name, timeout=step_timeout)
                if status != 0:
                    return (-1, 1)

            # Step 2: Produce generated STEP file
            status = _run_gen_code(gen_code, gen_file.name, timeout=step_timeout)
            if status != 0:
                return (-1, 2)

            # Step 3: Compute IOU
            result = _compute_iou(gt_file.name, gen_file.name, timeout=step_timeout, iou_mode=iou_mode)
        except subprocess.TimeoutExpired:
            result = (-1, 4)
        except Exception:
            result = (-1, 6)
        finally:
            gt_file.close()
            gen_file.close()

        # Allow small floating-point errors (e.g., 1.0000000000000002)
        eps = 1e-6
        if result[0] > 1.0 + eps and heal_failed:
            result = (-1, 5)

        return result

    def _save_checkpoint(self, ordered_results):
        if self.checkpoint_path:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(ordered_results, f)
            os.replace(tmp, self.checkpoint_path)

    def _run_parallel(self, completions):
        iou_func = partial(self._run, timeout=self.timeout, heal_failed=self.heal_failed, gt_is_step=self.gt_is_step, iou_mode=self.iou_mode)

        ordered_results = [None] * len(completions)

        # Load existing checkpoint if available
        pending_indices = set(range(len(completions)))
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as f:
                saved = pickle.load(f)
            if len(saved) == len(completions):
                for i, r in enumerate(saved):
                    if r is not None:
                        ordered_results[i] = r
                        pending_indices.discard(i)
                print(f"Resumed IoU checkpoint: {len(completions) - len(pending_indices)}/{len(completions)} already done")

        if not pending_indices:
            return self._finalize(ordered_results)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx) as executor:
            future_to_index = {
                executor.submit(iou_func, completions[i]): i
                for i in pending_indices
            }

            future_timeout = self.timeout + 10
            completed_count = 0
            since_last_save = 0

            if not self.quiet:
                pbar = tqdm(total=len(completions), initial=len(completions) - len(pending_indices))
                try:
                    for future in as_completed(future_to_index):
                        try:
                            index = future_to_index[future]
                            result = future.result(timeout=future_timeout)
                            ordered_results[index] = result
                        except Exception as e:
                            index = future_to_index[future]
                            ordered_results[index] = (-1, 6)
                            print(f"Error processing item {index}: {e}")
                        completed_count += 1
                        since_last_save += 1
                        pbar.update(1)
                        if since_last_save >= self.checkpoint_interval:
                            self._save_checkpoint(ordered_results)
                            since_last_save = 0
                finally:
                    pbar.close()

            else:
                try:
                    for future in as_completed(future_to_index):
                        try:
                            index = future_to_index[future]
                            result = future.result(timeout=future_timeout)
                            ordered_results[index] = result
                        except Exception as e:
                            index = future_to_index[future]
                            ordered_results[index] = (-1, 6)
                        completed_count += 1
                        since_last_save += 1
                        if since_last_save >= self.checkpoint_interval:
                            self._save_checkpoint(ordered_results)
                            since_last_save = 0
                except KeyboardInterrupt:
                    print(f"\nInterrupted! Processed {completed_count}/{len(pending_indices)} pending items")
                    self._save_checkpoint(ordered_results)
                    for future in future_to_index:
                        future.cancel()
                    raise

        self._save_checkpoint(ordered_results)
        return self._finalize(ordered_results)

    def _finalize(self, ordered_results):
        for i, result in enumerate(ordered_results):
            if result is None:
                ordered_results[i] = (-1, 6)
                print(f"Warning: Item {i} did not complete")

        ious = []
        statuses = []
        for iou, status in ordered_results:
            ious.append(iou)
            statuses.append(status)

        ious = np.array(ious)
        statuses = np.array(statuses, dtype=int)

        return ious, statuses

    def parse_results(self, ious, status_codes, verbose=True):
        summary = {
            'Valid IOU Mean': np.mean(ious[status_codes == 0]),
            'Valid IOU Median': np.median(ious[status_codes == 0]),
            'Valid IOU STD': np.std(ious[status_codes == 0]),
            'VSR': np.mean(status_codes == 0),
            'Adjusted IOU Mean': np.mean(np.where(status_codes == 0, ious, 0)),
            'Adjusted IOU Median': np.median(np.where(status_codes == 0, ious, 0)),
            'Adjusted IOU STD': np.std(np.where(status_codes == 0, ious, 0)),
            'Num Failed GT': np.sum(status_codes == 1),
            'Num Failed Gen': np.sum(status_codes == 2),
            'Num Failed OCC': np.sum(status_codes == 3),
            'Num Timeouts': np.sum(status_codes == 4),
            'Num None Solids': np.sum(status_codes == 5),
            'Num Failed Processing': np.sum(status_codes == 6)
        }

        if verbose:
            print("IOU Summary:")
            print(tabulate.tabulate(
                summary.items(),
                headers=['Metric', 'Value'],
                tablefmt='grid'
            ))

        return summary

    def save_results(self, ious, status_codes, output_file='experiment_results.json'):

        out = {"Summary": self.parse_results(ious, status_codes, verbose=False),
               "results": []}

        Errors = {
            1: "Ground Truth Reconstruction Failed",
            2: "Generated Code Failed",
            3: "OCC Computation Failed",
            4: "OCC Timeout",
            5: "None Solid Geometry Present",
            6: "Multiprocessing Error"
        }

        for i, (iou, status) in enumerate(zip(ious, status_codes)):
            out["results"].append({
                "Index": i,
                "IoU": iou if iou >= 0 else "N/A",
                "Status": Errors.get(status, "Unknown Error")
            })

        with open(output_file, 'w') as f:
            json.dump(out, f, indent=4)

        print(f"Results saved to {output_file}")
