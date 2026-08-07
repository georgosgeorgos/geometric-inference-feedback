import os
import tempfile
import asyncio
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import tabulate
from contextlib import redirect_stdout, redirect_stderr
import sys
from contextlib import contextmanager

# Ensure gift/inference/ is on sys.path so child processes can find geom module
_INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)
import json
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvloop
import uvicorn
from typing import Optional


TEST_TEMPLATE = """
{code}
cq.exporters.export(solid, {file_name})
"""

OCC_IOU_TEMPLATE = """
import cadquery as cq
from gift.inference.geom import align_shapes

solid_gt = cq.importers.importStep("{ground_truth_step_path}")
solid_gen = cq.importers.importStep("{generated_step_path}")

IOU = align_shapes(solid_gen, solid_gt)[1]
"""

@contextmanager
def suppress_output_os():
    if sys.platform == "win32":
        yield
        return
    stdout_fd, stderr_fd = 1, 2
    saved_stdout_fd, saved_stderr_fd = os.dup(stdout_fd), os.dup(stderr_fd)
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
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

def _get_iou(gt_code, gen_code, gt_file, gen_file, queue):
    with suppress_output_os(), suppress_output():
        try:
            import cadquery as cq
            gt_globals = {'__builtins__': __builtins__, 'cq': cq}
            gt_locals = {}
            exec(TEST_TEMPLATE.format(code=gt_code, file_name=f'"{gt_file}"'), gt_globals, gt_locals)
        except Exception as e:
            queue.put((-1, 1))
            return
        try:
            import cadquery as cq
            gen_globals = {'__builtins__': __builtins__, 'cq': cq}
            gen_locals = {}
            exec(TEST_TEMPLATE.format(code=gen_code, file_name=f'"{gen_file}"'), gen_globals, gen_locals)
        except Exception:
            queue.put((-1, 2))
            return
        try:
            import cadquery as cq
            from gift.inference.geom import align_shapes
            local_vars = {}
            exec_globals = {'__builtins__': __builtins__, 'cq': cq, 'align_shapes': align_shapes}
            exec(OCC_IOU_TEMPLATE.format(
                ground_truth_step_path=gt_file,
                generated_step_path=gen_file
            ), exec_globals, local_vars)
            iou = local_vars.get('IOU', -1.0)
            queue.put((iou, 0))
        except Exception:
            queue.put((-1, 3))

class IOUProcessor:
    def __init__(self, num_workers=8, timeout=10):
        self.num_workers = num_workers
        self.timeout = timeout
        self.process_pool = ProcessPoolExecutor(max_workers=num_workers)

    def __del__(self):
        """Clean up process pool on deletion"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)

    async def _run_async(self, response, timeout=30):
        """Async wrapper for the IOU calculation process using process pool"""
        loop = asyncio.get_event_loop()

        try:
            # Submit to process pool and wait for result
            future = self.process_pool.submit(self._run_in_process, response, timeout)
            result = await loop.run_in_executor(None, future.result)
            return result
        except Exception as e:
            return (-1, 6)  # Unknown error

    @staticmethod
    def _run_in_process(response, timeout=30):
        """Run IOU calculation in a separate process"""
        gt_code = response['ground_truth']
        gen_code = response['generated']
        gt_file = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
        gen_file = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
        gt_file.close()
        gen_file.close()
        queue = Queue()

        process = Process(target=_get_iou, args=(gt_code, gen_code, gt_file.name, gen_file.name, queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.kill()
            process.join()
            result = (-1, 4)  # Timeout
        else:
            if not queue.empty():
                result = queue.get()
            else:
                result = (-1, 6)

        # Clean up files
        try:
            os.remove(gt_file.name)
            os.remove(gen_file.name)
        except OSError:
            pass  # Files might already be deleted

        if result[0] > 1.0:
            result = (-1, 5)

        return result

    @staticmethod
    def _run(response, timeout=30):
        """Keep the original synchronous method for compatibility"""
        gt_code = response['ground_truth']
        gen_code = response['generated']
        gt_file = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
        gen_file = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
        gt_file.close()
        gen_file.close()
        queue = Queue()

        process = Process(target=_get_iou, args=(gt_code, gen_code, gt_file.name, gen_file.name, queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.kill()
            process.join()
            result = (-1, 4)  # Timeout
        else:
            if not queue.empty():
                result = queue.get()
            else:
                result = (-1, 6)

        os.remove(gt_file.name)
        os.remove(gen_file.name)

        if result[0] > 1.0:
            result = (-1, 5)

        return result

class IOURequest(BaseModel):
    ground_truth: str
    generated: str
    timeout: Optional[int] = Field(default=30, ge=1, le=300, description="Timeout in seconds (1-300)")

class IOUResponse(BaseModel):
    status: str
    iou: Optional[float]
    status_code: int
    error: Optional[str] = None
    processing_time: Optional[float] = None

processor = None

def create_app(num_workers: int = 1024) -> FastAPI:
    global processor
    processor = IOUProcessor(num_workers=num_workers)

    app = FastAPI(
        title="IOU Calculator API",
        description="Asynchronous API for calculating Intersection over Union (IOU) between 3D geometries",
        version="1.0.0"
    )

    # Setup all routes
    setup_routes(app)

    return app

app = None

ERROR_MESSAGES = {
    1: "Ground Truth code execution failed.",
    2: "Generated code execution failed.",
    3: "IOU calculation (OCC) failed.",
    4: "Processing timed out.",
    5: "Invalid solid geometry produced.",
    6: "An unknown multiprocessing error occurred."
}

def setup_routes(app):
    """Setup all routes for the FastAPI app"""

    @app.post("/iou", response_model=IOUResponse)
    async def calculate_iou(request: IOURequest):
        """
        Calculate IOU between ground truth and generated 3D geometry code.

        This endpoint accepts two pieces of code (ground_truth and generated)
        and calculates the Intersection over Union (IOU) between the resulting 3D geometries.
        You can also specify a custom timeout (1-300 seconds, default 30).
        """
        if not request.ground_truth or not request.generated:
            raise HTTPException(
                status_code=400,
                detail="Both 'ground_truth' and 'generated' code must be provided"
            )

        response_data = {
            'ground_truth': request.ground_truth,
            'generated': request.generated
        }

        start_time = asyncio.get_event_loop().time()

        try:
            iou, status_code = await processor._run_async(response_data, timeout=request.timeout)

            processing_time = asyncio.get_event_loop().time() - start_time

            if status_code == 0:
                return IOUResponse(
                    status="completed",
                    iou=iou,
                    status_code=status_code,
                    processing_time=processing_time
                )
            else:
                return IOUResponse(
                    status="failed",
                    iou=-1.0,
                    error=ERROR_MESSAGES.get(status_code, "An unknown error occurred."),
                    status_code=status_code,
                    processing_time=processing_time
                )
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "message": "IOU Calculator API is running"}

    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "IOU Calculator API",
            "version": "1.0.0",
            "process_workers": processor.num_workers,
            "endpoints": {
                "calculate_iou": "/iou",
                "health_check": "/health",
                "docs": "/docs"
            }
        }

def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI IOU Calculator Server")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1024,
        help="Number of worker processes for IOU calculations (default: 1024)"
    )
    parser.add_argument(
        "--server-workers",
        type=int,
        default=1,
        help="Number of uvicorn server workers (default: 1)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind the server to (default: 5000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    app = create_app(num_workers=args.num_workers)

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    print(f"Starting IOU Calculator API server...")
    print(f"Process workers: {args.num_workers}")
    print(f"Server workers: {args.server_workers}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")

    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.server_workers,
        reload=args.reload,
        loop="uvloop"
    )
