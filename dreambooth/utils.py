from pathlib import Path

from diffusers import AutoencoderKL
from huggingface_hub import model_info
from transformers import PretrainedConfig


def load_text_encoder_class(model_name: str, revision: str):
    """Load the text encoder class from the model name and revision."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def model_has_vae(model_name):
    config_file_name = Path("vae", AutoencoderKL.config_name).as_posix()
    files_in_repo = model_info(model_name).siblings
    return any(file.rfilename == config_file_name for file in files_in_repo)


def encode_prompt(
    text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None
):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds
