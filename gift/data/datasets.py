from PIL import Image
import json
import numpy as np
from tqdm.auto import trange
import torch
import torchvision
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import pickle
import io
from copy import deepcopy
from datasets import Dataset
import joblib
import base64
from io import BytesIO
import asyncio
import re
from gift.data.utils import extract_code


class MinimalImageCADDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dict: List[dict],
        collate_fn: Callable,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        user_prompt: Optional[
            str
        ] = "Generate the CADQuery code needed to create the CAD for the provided image.",
        assistant_response_template: Optional[
            str
        ] = "Certainly! Based on the provided image, here is the CADQuery code to generate the corresponding CAD model:\n```python\n{code}\n```\n\nThe variable `solid` contains the final CAD model. You can use it to export the model in various formats, such as STEP or STL, using the appropriate methods in the CADQuery library.\n\nIf you have any other questions or would like to modify the model further, feel free to ask!",
        image_augmentation_fn: Optional[Callable] = None,
        basic_image_augmentation: bool = False,
        augmentation_probability: float = 0.5,
        image_size: Tuple[int, int] = (448, 448),
        lazy_image_loading: bool = False,
    ):
        """
        Args:
            data_dict (List[dict]): List of dictionaries containing the dataset.
            collate_fn (Callable): Function to collate the data. (input will be messages in a given sample same as the template in CADLM)
            image_augmentation_fn (Optional[Callable], optional): Function for image augmentation. Defaults to None.
            basic_image_augmentation (bool, optional): Whether to apply basic image augmentation. Defaults to False.
            augmentation_probability (float, optional): Probability of applying augmentation. Defaults to 0.5.
            lazy_image_loading (bool, optional): Whether to delay image decoding until __getitem__. Saves memory. Defaults to False.
        """
        super().__init__()

        self.data_dict = data_dict
        self._collate_fn = collate_fn
        self.lazy_image_loading = lazy_image_loading
        if image_augmentation_fn is not None:
            self.image_augmentation_fn = image_augmentation_fn
        elif basic_image_augmentation:
            self.image_augmentation_fn = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    torchvision.transforms.RandomErasing(
                        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)
                    ),
                    torchvision.transforms.RandomAffine(
                        degrees=45, translate=(0.2, 0.2), scale=(0.75, 1.25)
                    ),
                ]
            )
        else:
            self.image_augmentation_fn = None
        self.augmentation_probability = augmentation_probability

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.assistant_response_template = assistant_response_template
        self.image_size = image_size

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            image = self.data_dict[idx]["image"]
            code = self.data_dict[idx]["code"]

            # Handle lazy image loading (decode from bytes if needed)
            if self.lazy_image_loading and isinstance(image, dict) and "bytes" in image:
                buffer = io.BytesIO(image["bytes"])
                image = Image.open(buffer).convert("RGB")

            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError(f"Image at index {idx} is not a PIL Image or bytes dict")

            # Resize image to standardized size (optimized: only if needed)
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)

            # Apply augmentation if enabled
            if (
                self.image_augmentation_fn is not None
                and np.random.rand() < self.augmentation_probability
            ):
                image_tensor = torchvision.transforms.ToTensor()(image)
                augmented_image_tensor = self.image_augmentation_fn(image_tensor)
                image = torchvision.transforms.ToPILImage()(augmented_image_tensor)

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": self.assistant_response_template.format(code=code),
                        }
                    ],
                },
            ]

            return prompt
        except Exception as e:
            # Log the error and return next valid sample to avoid training interruption
            print(f"Warning: Error loading sample {idx}: {e}. Skipping to next sample.")
            # Return next sample (with wraparound)
            return self.__getitem__((idx + 1) % len(self))

    @staticmethod
    def encode_image(image: Image.Image, im_type="jpeg") -> str:
        buffered = BytesIO()
        image.save(buffered, format=im_type)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def gather_vllm_data(self, image_type="png"):
        prompts = []
        gt = []
        for idx in trange(len(self.data_dict)):
            image = self.data_dict[idx]["image"]
            code = self.data_dict[idx]["code"]

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{self.encode_image(image, im_type=image_type)}"
                            },
                        },
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ]
            prompts.append(prompt)
            gt.append(code)

        return prompts, gt


class CADRLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dict: List[dict],
        collate_fn: Callable,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        user_prompt: Optional[
            str
        ] = "Generate the CADQuery code needed to create the CAD for the provided image.",
        assistant_response_template: Optional[
            str
        ] = "Certainly! Based on the provided image, here is the CADQuery code to generate the corresponding CAD model:\n```python\n{code}\n```\n\nThe variable `solid` contains the final CAD model. You can use it to export the model in various formats, such as STEP or STL, using the appropriate methods in the CADQuery library.\n\nIf you have any other questions or would like to modify the model further, feel free to ask!",
        image_augmentation_fn: Optional[Callable] = None,
        basic_image_augmentation: bool = False,
        augmentation_probability: float = 0.5,
        image_type: str = "png",
        image_size: Tuple[int, int] = (448, 448),
    ):
        """
        Args:
            data_dict (List[dict]): List of dictionaries containing the dataset.
            collate_fn (Callable): Function to collate the data. (input will be messages in a given sample same as the template in CADLM)
            image_augmentation_fn (Optional[Callable], optional): Function for image augmentation. Defaults to None.
            basic_image_augmentation (bool, optional): Whether to apply basic image augmentation. Defaults to False.
            augmentation_probability (float, optional): Probability of applying augmentation. Defaults to 0.5.
        """
        super().__init__()

        self.data_dict = data_dict
        self._collate_fn = collate_fn
        if image_augmentation_fn is not None:
            self.image_augmentation_fn = image_augmentation_fn
        elif basic_image_augmentation:
            self.image_augmentation_fn = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    torchvision.transforms.RandomErasing(
                        p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)
                    ),
                    torchvision.transforms.RandomAffine(
                        degrees=45, translate=(0.2, 0.2), scale=(0.75, 1.25)
                    ),
                ]
            )
        else:
            self.image_augmentation_fn = None
        self.augmentation_probability = augmentation_probability

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.assistant_response_template = assistant_response_template
        self.image_type = image_type
        self.image_size = image_size

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.data_dict[idx]["image"]
        code = self.data_dict[idx]["code"]

        # Resize image to standardized size
        if isinstance(image, Image.Image):
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        if (
            self.image_augmentation_fn is not None
            and np.random.rand() < self.augmentation_probability
        ):
            if isinstance(image, Image.Image):
                image_tensor = torchvision.transforms.ToTensor()(image)
                augmented_image_tensor = self.image_augmentation_fn(image_tensor)
                image = torchvision.transforms.ToPILImage()(augmented_image_tensor)

        vllm_prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{self.image_type};base64,{self.encode_image(image, im_type=self.image_type)}"
                        },
                    },
                    {"type": "text", "text": self.user_prompt},
                ],
            },
        ]

        return {"image": image, "code": code, "vllm_prompt": vllm_prompt}

    @staticmethod
    def encode_image(image: Image.Image, im_type="jpeg") -> str:
        buffered = BytesIO()
        image.save(buffered, format=im_type)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def gather_vllm_data(self, image_type="png"):
        prompts = []
        gt = []
        for idx in trange(len(self.data_dict)):
            image = self.data_dict[idx]["image"]
            code = self.data_dict[idx]["code"]

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{self.encode_image(image, im_type=image_type)}"
                            },
                        },
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ]
            prompts.append(prompt)
            gt.append(code)

        return prompts, gt
