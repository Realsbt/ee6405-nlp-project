"""Train an independent scale control; screening never opens test by default."""

from pathlib import Path

from experiments.fusion_scale_controls import (
    SCALE_CONDITIONS, ScaleControlModel, scale_config, scale_dataset,
)
from experiments.train_same_window import main


if __name__ == "__main__":
    main(
        conditions=SCALE_CONDITIONS,
        configuration_factory=scale_config,
        model_factory=ScaleControlModel,
        dataset_factory=scale_dataset,
        validation_only_default=True,
        extra_source_files=(Path(__file__), Path(__file__).with_name("edu_fusion.py")),
    )
