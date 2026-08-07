from typing import Optional
import torch
import requests
import time
import atexit
from torch import nn
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
import socket
from urllib.parse import urlparse
from openai import AsyncOpenAI, OpenAI
import asyncio
from tqdm.asyncio import tqdm

class VLLMClient():
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):

        self.session = requests.Session()

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout
        
        openai_api_key = "EMPTY"
        self.oai_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=self.base_url+ "/v1",
        )
        
        self.oai_simple_client = OpenAI(
            api_key=openai_api_key,
            base_url=self.base_url + "/v1")
        
        if model_name is None:
            # If no model name is provided, use the default model from the server
            self.model_name = self.oai_simple_client.models.list().data[0].id
        else:
            # If a model name is provided, use it
            self.model_name = model_name
            
        # default generation parameters
        self.temperature = 1.0
        self.top_p = 1.0
        self.max_tokens = 4096
        self.n = 1
    
    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.base_url}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    
    def sleep(self):
        
        url = f"{self.base_url}/sleep"
        
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        
        # wait till is_sleeping is True
        url = f"{self.base_url}/is_sleeping"
        while True:
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                if data["is_sleeping"]:
                    break
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
            time.sleep(0.1)
    
    def wake_up(self):
        """
        Wakes up the model from sleep mode.
        """
        url = f"{self.base_url}/wake_up/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
    
    def set_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        n: int = 1,
    ):
        """
        Set the sampling parameters for the model.

        Args:
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.
            max_tokens (`int`, *optional*, defaults to `4096`):
                Maximum number of tokens to generate.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate.
        """
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n

    async def run_chat_completion(self, prompts):
        # upto 5 retries for chat completion
        for i in range(5):
            try:
                results = await self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=prompts,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n = self.n
                )
                if results.choices:
                    return results
            except Exception as e:
                if i == 4:
                    raise e
                continue
    
    async def chat(self, prompts: list[list[dict]], verbose: bool = False, max_concurrent: int = 50):
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_chat_completion(prompt):
            async with semaphore:
                return await self.run_chat_completion(prompt)
        
        results = [limited_chat_completion(prompt) for prompt in prompts]
        
        if verbose:
            completions = await tqdm.gather(*results, desc="Generating completions")
        else:
            completions = await asyncio.gather(*results)
        
        final_completions = []
        for completion in completions:
            final_completions.append([choice.message.content for choice in completion.choices])
            
        return final_completions

    def EZ_chat(self, prompts: list[list[dict]]):
        url = f"{self.base_url}/batch_chat/"

        response = self.session.post(url, json={
            "prompts": prompts,
            "n": self.n,
            "repetition_penalty": 1.0,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": -1,
            "min_p": 0.0,
            "max_tokens": self.max_tokens
        })
        
        if response.status_code == 200:
            data = response.json()
            completions = data["completions"]
            return completions
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"{self.base_url}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        "sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    return None

            # Retry logic: wait before trying again
            time.sleep(retry_interval)
        
    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the world size from the server
        url = f"{self.base_url}/get_world_size/"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, params={"host": self.host, "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=0)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype).replace('torch.',''), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # self.pynccl_comm.group.barrier()
        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        # self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module, exclude: Optional[list[str]] = None):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        self.sleep()
        for name, param in model.named_parameters():
            if exclude is not None:
                for e in exclude:
                    if e in name:
                        # Skip this parameter if it matches any of the excluded keywords
                        continue
            # Update each parameter individually
            self.update_named_param(name, param.data)
        self.wake_up()
            
    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"{self.base_url}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
    
    def __getstate__(self):
        """
        Returns the state of the VLLMClient instance.
        """
        dict_state = self.__dict__.copy()
        
        dict_state["session"] = None  # Don't pickle the session, as it cannot be pickled
        dict_state["oai_client"] = None  # Don't pickle the OpenAI client, as it cannot be pickled
        dict_state["oai_simple_client"] = None  # Don't pickle the simple OpenAI client, as it cannot be pickled
        
        dict_state["pynccl_comm"] = None  # Don't pickle the communicator, as it cannot be pickled
        
        return dict_state
    
    def __setstate__(self, state):
        """
        Sets the state of the VLLMClient instance.
        """
        self.__dict__.update(state)
        
        # Reinitialize the session and clients
        self.session = requests.Session()
        self.oai_client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=self.base_url + "/v1"
        )
        
        self.oai_simple_client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url + "/v1"
        )
        
    
class DebugClient():
    """
    A debug client that inherits from VLLMClient and overrides the `EZ_chat` method to return a fixed response.
    This is useful for testing purposes.
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def EZ_chat(self, *args, **kwargs):
        """
        Returns a fixed response for debugging purposes.
        """
        pass
    
    def run_chat_completion(self, *args, **kwargs):
        """
        Returns a fixed response for debugging purposes.
        """
        pass
    
    def init_communicator(self):
        """
        Initializes the communicator for the debug client.
        """
        pass

    def update_named_param(self, *args, **kwargs):
        """
        Updates a named parameter in the debug client.
        """
        pass

    def update_model_params(self, *args, **kwargs):
        """
        Updates the model parameters in the debug client.
        """
        pass
    
    def close_communicator(self):
        """
        Closes the communicator for the debug client.
        """
        pass
    
    def sleep(self):
        """
        Sleeps the debug client.
        """
        pass

    def wake_up(self):
        """
        Wakes up the debug client.
        """
        pass