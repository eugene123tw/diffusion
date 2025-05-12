from pytorch_lightning.cli import LightningCLI

from dreambooth.dreambooth import DreamBoothLightningModule


def cli_main():
    LightningCLI(
        model_class=DreamBoothLightningModule,
        run=True,
    )


if __name__ == "__main__":
    cli_main()
