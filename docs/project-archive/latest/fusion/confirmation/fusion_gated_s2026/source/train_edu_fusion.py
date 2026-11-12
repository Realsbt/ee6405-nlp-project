"""Train an EDU fusion candidate; default to validation-only screening."""

from pathlib import Path

from experiments.edu_fusion import (
    EDUFusionModel, FUSION_CONDITIONS, fusion_config, fusion_dataset,
)
from experiments.train_same_window import main


if __name__ == "__main__":
    main(
        conditions=FUSION_CONDITIONS,
        configuration_factory=fusion_config,
        model_factory=EDUFusionModel,
        dataset_factory=fusion_dataset,
        validation_only_default=True,
        extra_source_files=(Path(__file__),),
    )
