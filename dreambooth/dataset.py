from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, PreTrainedTokenizer


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_dir: Path,
        instance_prompt: str,
        tokenizer: PreTrainedTokenizer,
        class_data_dir: Path | None = None,
        class_prompt: str | None = None,
        class_num: int | None = None,
        size: int = 512,
        center_crop: bool = False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_dir = Path(instance_data_dir)
        if not self.instance_data_dir.exists():
            raise ValueError(
                f"Instance {self.instance_data_dir} images root doesn't exists."
            )

        self.instance_images_path = list(Path(instance_data_dir).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_dir is not None:
            self.class_data_root = Path(class_data_dir)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                (
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.RandomCrop(size)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer,
                self.instance_prompt,
                tokenizer_max_length=self.tokenizer_max_length,
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer,
                    self.class_prompt,
                    tokenizer_max_length=self.tokenizer_max_length,
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        class_prompt: str,
        class_data_dir: Path,
        instance_prompt: str,
        instance_data_dir: Path,
        num_class_images: int = 200,
        with_prior_preservation: bool = False,
        batch_size: int = 1,
        num_workers: int = 2,
        prior_generator: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        prior_generation_precision: str = "fp16",
    ):
        super().__init__()
        self.model_name = model_name
        self.instance_prompt = instance_prompt
        self.instance_data_dir = Path(instance_data_dir)
        self.class_prompt = class_prompt
        self.class_data_dir = class_data_dir
        self.num_class_images = num_class_images
        self.with_prior_preservation = with_prior_preservation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_generator = prior_generator
        self.prior_generation_precision = prior_generation_precision

    def prepare_data(self):
        if self.with_prior_preservation:
            self._generate_class_images()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            subfolder="tokenizer",
            use_fast=False,
        )

        self.train_dataset = DreamBoothDataset(
            instance_data_dir=self.instance_data_dir,
            tokenizer=tokenizer,
            instance_prompt=self.instance_prompt,
            class_prompt=self.class_prompt,
            class_data_dir=self.class_data_dir,
            class_num=self.num_class_images,
            size=512,
        )

        self.val_dataset = PromptDataset(
            prompt=self.instance_prompt,
            num_samples=50,  # Number of iterations
        )

    def _generate_class_images(self):
        self.class_data_dir.mkdir(parents=True, exist_ok=True)
        curr_class_images = len(list(Path(self.class_data_dir).glob("*.jpg")))
        if curr_class_images > self.num_class_images:
            print(f"[✓] Using existing class images at {self.class_data_dir}")
            return

        sample_dataset = PromptDataset(self.class_prompt, self.num_class_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=4)

        num_new_images = self.num_class_images - curr_class_images

        if self.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif self.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif self.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16

        print(f"[✓] Generating {num_new_images} class images to {self.class_data_dir}")
        pipeline = DiffusionPipeline.from_pretrained(
            self.prior_generator,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to("cuda")

        for example in sample_dataloader:
            images = pipeline(example["prompt"]).images
            for i, image in enumerate(images):
                image_filename = (
                    self.class_data_dir
                    / f"{example['index'][i] + curr_class_images}.jpg"
                )
                image.save(image_filename)
        print(
            f"[✓] Generated {self.num_class_images} class images to {self.class_data_dir}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, self.with_prior_preservation
            ),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
