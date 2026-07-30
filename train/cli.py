"""
Entry point for all training tasks via Lightning CLI.

Usage:
    python -m train.cli fit   --config train/configs/detection.yaml
    python -m train.cli fit   --config train/configs/span_detection.yaml
    python -m train.cli fit   --config train/configs/attention_span_detection.yaml
    python -m train.cli fit   --config train/configs/relation_classification.yaml
    python -m train.cli test  --config train/configs/detection.yaml --ckpt_path best
"""
from lightning.pytorch.cli import LightningCLI

import train.models.detection                    # noqa: F401  — registers with CLI
import train.models.span_detection               # noqa: F401
import train.models.attention_span_detection     # noqa: F401
import train.models.biaffine_span_detection      # noqa: F401
import train.models.relation_classification      # noqa: F401
import train.data.detection                      # noqa: F401
import train.data.span_detection                 # noqa: F401
import train.data.attention_span_detection       # noqa: F401
import train.data.relation_classification        # noqa: F401


def main() -> None:
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
