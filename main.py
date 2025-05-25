from pytorch_lightning.cli import LightningCLI

from dreambooth.dataset import DreamBoothDataModule
from dreambooth.dreambooth import DreamBoothLightningModule


def cli_main():
    LightningCLI(
        model_class=DreamBoothLightningModule,
        datamodule_class=DreamBoothDataModule,
    )


if __name__ == "__main__":
    cli_main()
