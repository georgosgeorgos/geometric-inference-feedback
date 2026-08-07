import asyncio
import httpx
from typing import List, Dict
import numpy as np
from tqdm.auto import tqdm


class IOUClient:
    """
    An asynchronous client to send concurrent IOU requests to the Flask server.
    """

    def __init__(
        self,
        server_url: str,
        concurrent_requests: int = 128,
        timeout: int = 30,
        verbose: bool = False,
    ):
        self.iou_endpoint = f"{server_url.rstrip('/')}/iou"
        self.timeout = timeout
        self.verbose = verbose
        self.concurrent_requests = concurrent_requests

    async def _send_request(
        self, job: Dict[str, str], index: int, retry: int = 3
    ) -> Dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.iou_endpoint, json=job, timeout=self.timeout * 2
                )
        except Exception as e:
            if retry > 0:
                return await self._send_request(job, index, retry - 1)
            else:
                return {
                    "status": "failed",
                    "error": str(e),
                    "iou": -1.0,
                    "status_code": 6,
                }

        if response.status_code == 200:
            return response.json()
        else:
            if retry > 0:
                return await self._send_request(job, index, retry - 1)
            else:
                return {
                    "status": "failed",
                    "error": f"HTTP error: {response.status_code}",
                    "iou": -1.0,
                    "status_code": response.status_code,
                }

    async def process_jobs(self, jobs: List[Dict[str, str]]) -> List[Dict]:
        for i in range(len(jobs)):
            if "timeout" not in jobs[i]:
                jobs[i]["timeout"] = self.timeout

        semaphore = asyncio.Semaphore(self.concurrent_requests)

        async def limited_request(job, index: int):
            async with semaphore:
                return await self._send_request(job, index)

        tasks = [limited_request(job, i) for i, job in enumerate(jobs)]

        if self.verbose:
            results = await tqdm.gather(
                *tasks, desc="Processing IOU requests", total=len(tasks)
            )
        else:
            results = await asyncio.gather(*tasks)

        return results

    async def main(self, jobs: List[Dict[str, str]]) -> tuple[List, List]:
        ordered_results = await self.process_jobs(jobs)

        ious = []
        statuses = []
        for result in ordered_results:
            iou = result.get("iou", -1.0)
            status = result.get("status_code", 6)
            ious.append(iou)
            statuses.append(status)

        ious = np.array(ious)
        statuses = np.array(statuses, dtype=int)

        return ious, statuses
