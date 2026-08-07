"""3D-to-2D rendering via CadQuery + PyVista + Xvfb."""
import os
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple
from PIL import Image

CAD_PYTHON = os.environ.get(
    "CAD_PYTHON_PATH",
    "/workspace/home/lab/gcg/dev/.cad/bin/python",
)


def render_code_to_image(
    code: str,
    size: Tuple[int, int] = (448, 448),
    timeout: int = 30,
) -> Optional[Image.Image]:
    """Execute CadQuery code, render the result to a 2D image.

    Returns PIL Image or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_path = os.path.join(tmpdir, "output.stl")
        png_path = os.path.join(tmpdir, "output.png")

        if not _execute_to_stl(code, stl_path, timeout):
            return None

        if not _render_stl_to_png(stl_path, png_path, size, timeout):
            return None

        return Image.open(png_path).copy()


def _execute_to_stl(code: str, stl_path: str, timeout: int) -> bool:
    """Run CadQuery code in subprocess, export to STL."""
    script = f"""
import cadquery as cq
{code}
cq.exporters.export(solid, r"{stl_path}")
"""
    try:
        result = subprocess.run(
            [CAD_PYTHON, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and os.path.exists(stl_path)
    except (subprocess.TimeoutExpired, Exception):
        return False


def _render_stl_to_png(
    stl_path: str, png_path: str, size: Tuple[int, int], timeout: int
) -> bool:
    """Render STL file to PNG via PyVista offscreen under Xvfb."""
    script = f"""
import pyvista as pv
pv.OFF_SCREEN = True
mesh = pv.read(r"{stl_path}")
plotter = pv.Plotter(off_screen=True, window_size={list(size)})
plotter.set_background("white")
plotter.add_mesh(mesh, color="steelblue", smooth_shading=True)
plotter.view_isometric()
plotter.screenshot(r"{png_path}")
plotter.close()
"""
    try:
        result = subprocess.run(
            ["xvfb-run", "-a", CAD_PYTHON, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0 and os.path.exists(png_path)
    except (subprocess.TimeoutExpired, Exception):
        return False


def render_stl_to_image(
    stl_path: str,
    size: Tuple[int, int] = (448, 448),
    timeout: int = 30,
) -> Optional[Image.Image]:
    """Render an existing STL file to image. Convenience wrapper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        png_path = os.path.join(tmpdir, "output.png")
        if not _render_stl_to_png(stl_path, png_path, size, timeout):
            return None
        return Image.open(png_path).copy()
