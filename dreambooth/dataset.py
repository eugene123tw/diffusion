from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import pytorch_lightning as pl
from pathlib import Path


class DreamBoothDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: str,
        class_prompt: str = "a photo of a dog",
        class_data_dir: Path = Path("./class_data"),
        num_class_images: int = 200,
        with_prior_preservation: bool = False,
        batch_size: int = 1,
        num_workers: int = 2,
        prior_generator: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        prior_generation_precision: str = "fp16",
    ):
        super().__init__()
        self.image_dir = image_dir
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

    def _generate_class_images(self):
        self.class_data_dir.mkdir(parents=True, exist_ok=True)
        curr_class_images = len(list(Path(self.class_data_dir).glob("*.jpg")))
        if curr_class_images > self.num_class_images:
            print(f"[✓] Using existing class images at {self.class_data_dir}")
            return

        num_new_images = self.num_class_images - curr_class_images

        print(f"[✓] Generating {num_new_images} class images to {self.class_data_dir}")
        pipeline = DiffusionPipeline.from_pretrained(
            self.prior_generator
            torch_dtype=self.prior_generation_precision,
            safety_checker=None,
        ).to("cuda")

        for i in range(self.num_class_images):
            image = pipeline(self.class_prompt).images[0]
            image.save(self.class_data_di / f"{i:04}.jpg")
        print(f"[✓] Generated {self.num_class_images} class images to {self.class_data_dir}")

    def train_dataloader(self):
        