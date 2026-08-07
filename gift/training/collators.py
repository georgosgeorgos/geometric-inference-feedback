import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

from PIL import Image
import json
import numpy as np
from tqdm.auto import trange
from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLProcessor
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import re
import multiprocessing as mp
import asyncio
from gift.data.utils import extract_code

class QwenVLCollator:
    def __init__(self, processor: Qwen2VLProcessor, assistant_only: bool = True, max_len: Optional[int] = 4096):
        self.processor = processor
        self.assistant_only = assistant_only
        self.max_len = max_len
        self.special_tokens = processor.tokenizer("<|im_start|>assistant<|im_end|>")['input_ids']
        # Cache image token IDs for faster masking
        if isinstance(processor, Qwen2VLProcessor):
            self.image_tokens = [151652, 151653, 151655]
        else:
            self.image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    def __call__(self, samples):
        texts = self.processor.apply_chat_template(samples, tokenize=False)
        image_inputs = process_vision_info(samples)[0]

        # Tokenize the texts and process the images
        if self.max_len is None:
            batch = self.processor(
                text=texts, images=image_inputs, return_tensors="pt", padding=True
            )
        else:
            batch = self.processor(
                text=texts, images=image_inputs, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_len
            )

        labels = batch["input_ids"].clone()

        # Vectorized masking for better performance
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask image token IDs in the labels (use cached tokens)
        for image_token_id in self.image_tokens:
            labels[labels == image_token_id] = -100

        if self.assistant_only:
            # Only keep the assistant's response in the labels only
            # since all samples have the same length user prompt we just find the start of the assistant response
            for i in range(len(labels)):
                starts = torch.where(labels[i] == self.special_tokens[0])[0]
                ends = torch.where(labels[i] == self.special_tokens[2])[0]

                if len(ends)< len(starts):
                    ends_ = torch.zeros_like(starts)
                    ends_[:len(ends)] = ends
                    ends_[len(ends):] = len(labels[i]) - 1
                    ends = ends_

                for j in range(len(starts)):
                    if labels[i][starts[j]+1] != self.special_tokens[1]:
                        labels[i][starts[j]:ends[j]+2] = -100

        batch["labels"] = labels  # Add labels to the batch
        return batch

class QwenVLRLCollator:
    def __init__(self,
                 processor: Qwen2VLProcessor,
                 vllm_client: Any,
                 iou_client: Any,
                 max_batch_size: int = 8,
                 n : int = 8,
                 max_tokens: int = 4096,
                 temperature: float = 0.5,
                 top_p: float = 0.9,
                 assistant_only: bool = True,
                 system_prompt: Optional[str] = "You are a helpful assistant.",
                 user_prompt: Optional[str] = "Generate the CADQuery code needed to create the CAD for the provided image.",
                 max_len: Optional[int] = 4096,
                 reward_type: Optional[str] = "iou"
                 ):

        self.processor = processor
        self.max_batch_size = max_batch_size
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.client = vllm_client
        self.client.set_sampling_params(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        self.n = n
        self.assistant_only = assistant_only
        self.max_len = max_len
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.special_tokens = processor.tokenizer("<|im_start|>assistant<|im_end|>")['input_ids']
        self.iouclient = iou_client
        self.reward_type = reward_type

    def __call__(self, _batch):

        prompts = [item["vllm_prompt"] for item in _batch]
        completions = asyncio.run(self.client.chat(prompts))

        responses = []

        for i in range(len(completions)):
            code = _batch[i]['code']
            for completion in completions[i]:
                responses.append({
                    "ground_truth": code,
                    "generated": extract_code(completion)
                })


        samples = []
        for i in range(len(_batch)):
            code = _batch[i]['code']
            image = _batch[i]['image']

            for completion in completions[i]:
                prompt = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image},
                                    {"type": "text", "text": self.user_prompt}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": completion}]
                    }
                ]
                samples.append(prompt)

        batch_size = len(_batch)
        iou, status_codes = asyncio.run(self.iouclient.main(responses))

        if self.reward_type == "iou":
            rewards = np.array(iou)
        elif self.reward_type == "syntax":
            rewards = np.where(status_codes == 0 , 1.0, -1.0)

        advantage = rewards.reshape(batch_size, -1)
        advantage = (advantage - advantage.mean(axis=1, keepdims=True)) / (advantage.std(axis=1, keepdims=True) + 1e-6)
        advantage = advantage.reshape(-1)

        n_samples = len(samples)
        n_batches = (n_samples + self.max_batch_size - 1) // self.max_batch_size

        batches = []
        for i in range(n_batches):
            texts = self.processor.apply_chat_template(samples[i*self.max_batch_size:(i+1)*self.max_batch_size], tokenize=False)
            image_inputs = process_vision_info(samples[i*self.max_batch_size:(i+1)*self.max_batch_size])[0]

            # Tokenize the texts and process the images
            if self.max_len is None:
                batch = self.processor(
                    text=texts, images=image_inputs, return_tensors="pt", padding=True
                )
            else:
                batch = self.processor(
                    text=texts, images=image_inputs, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_len
                )

            labels = batch["input_ids"].clone()

            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            if isinstance(self.processor, Qwen2VLProcessor):
                image_tokens = [151652, 151653, 151655]
            else:
                image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]

            # Mask image token IDs in the labels
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            if self.assistant_only:
                # Only keep the assistant's response in the labels only
                # since all samples have the same length user prompt we just find the start of the assistant response
                for i_ in range(len(labels)):
                    starts = torch.where(labels[i_] == self.special_tokens[0])[0]
                    ends = torch.where(labels[i_] == self.special_tokens[2])[0]

                    if len(ends)< len(starts):
                        ends_ = torch.zeros_like(starts)
                        ends_[:len(ends)] = ends
                        ends_[len(ends):] = len(labels[i_]) - 1
                        ends = ends_

                    for j in range(len(starts)):
                        if labels[i_][starts[j]+1] != self.special_tokens[1]:
                            labels[i_][starts[j]:ends[j]+2] = -100

            batch["labels"] = labels.clone()  # Add labels to the batch
            batch["advantage"] = torch.tensor(advantage[i*self.max_batch_size:(i+1)*self.max_batch_size], dtype=torch.float32)
            batch['rewards'] = torch.tensor(rewards[i*self.max_batch_size:(i+1)*self.max_batch_size], dtype=torch.float32)
            batches.append(batch)
        return batches

_spawn_context = mp

class NoDaemonProcess(_spawn_context.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess
