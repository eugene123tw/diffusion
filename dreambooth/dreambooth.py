import itertools

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from torchmetrics.multimodal import CLIPScore
from torchvision import transforms
from transformers import AutoTokenizer, get_scheduler

from dreambooth.utils import encode_prompt, load_text_encoder_class, model_has_vae


class DreamBoothLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name,
        train_text_encoder=False,
        learning_rate=1e-5,
        with_prior_preservation=False,
        prior_loss_weight=1.0,
        num_validation_images=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.train_text_encoder = train_text_encoder
        self.with_prior_preservation = with_prior_preservation
        self.prior_loss_weight = prior_loss_weight
        self.learning_rate = learning_rate
        self.num_validation_images = num_validation_images
        self.clip_score_metric = CLIPScore(
            model_name_or_path="openai/clip-vit-base-patch16"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_fast=False,
        )

        # Load the text encoder
        text_encoder_class = load_text_encoder_class(model_name)
        self.text_encoder = text_encoder_class.from_pretrained(
            model_name,
            subfolder="text_encoder",
        )

        # Freeze the text encoder if not training
        if not train_text_encoder:
            self.text_encoder.requires_grad_(False)

        # Check if the model has a VAE
        self.vae = (
            AutoencoderKL.from_pretrained(model_name, subfolder="vae")
            if model_has_vae(model_name)
            else None
        )

        # Freeze the VAE if it exists
        if self.vae is not None:
            self.vae.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        self.img_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts PIL [0,255] to Tensor [0,1]
            ]
        )

    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pixel_values = batch["pixel_values"]
        if self.vae is not None:
            # Convert images to latent space
            model_input = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
        else:
            model_input = pixel_values

        # Sample noise
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=model_input.device,
        )
        timesteps = timesteps.long()
        noisy_model_input = self.noise_scheduler.add_noise(
            model_input, noise, timesteps
        )

        encoder_hidden_states = encode_prompt(
            self.text_encoder,
            batch["input_ids"],
            batch["attention_mask"],
            text_encoder_use_attention_mask=False,
        )

        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states,
            class_labels=None,
            return_dict=False,
        )[0]

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
            )

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(), target_prior.float(), reduction="mean"
            )

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.with_prior_preservation:
            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pipeline_args = {}
        if self.vae is not None:
            pipeline_args["vae"] = self.vae

        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            **pipeline_args,
        )
        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, **scheduler_args
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        pipeline_args = {"prompt": batch["prompt"]}

        # run inference
        generator = torch.Generator(device=self.device).manual_seed(42)

        with torch.autocast("cuda"):
            pil_image = pipeline(
                **pipeline_args, num_inference_steps=25, generator=generator
            ).images[0]

        generated_tensor = self.img_to_tensor(pil_image).unsqueeze(0)
        self.clip_score_metric.update(
            (generated_tensor * 255).to(torch.uint8).to(self.device),
            batch["prompt"],
        )

    def on_validation_epoch_end(self):
        # Compute the CLIP score at the end of the validation epoch
        clip_score = self.clip_score_metric.compute()
        self.log("val/clip_score", clip_score, on_epoch=True, prog_bar=True)
        self.clip_score_metric.reset()

    # @torch.no_grad()
    # def test_step(self, batch, batch_idx):
    #     pixel_values = batch["pixel_values"]

    def configure_optimizers(self):
        # Implement the optimizer configuration here
        total_steps = self.trainer.estimated_stepping_batches

        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
            if self.train_text_encoder
            else self.unet.parameters()
        )

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-08,
        )

        scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # step-wise update
            },
        }
