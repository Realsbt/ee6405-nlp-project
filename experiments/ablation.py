"""The fixed architecture x objective comparison, without ML dependencies."""

import copy


SEEDS = (3407, 42, 2026)
CONDITIONS = {
    "roberta": ("roberta", False),
    "roberta_supcon": ("roberta", True),
    "race_no_supcon": ("race", False),
    "race": ("race", True),
}
NAMES = {"roberta": "RoBERTa · CE", "roberta_supcon": "RoBERTa · CE + SupCon",
         "race_no_supcon": "RACE · CE", "race": "RACE · CE + SupCon"}
NEURAL_RUNS = tuple(f"{kind}_s{seed}" for kind in CONDITIONS for seed in SEEDS)
MISSING_RUNS = tuple(f"roberta_supcon_s{seed}" for seed in SEEDS) + (
    "race_no_supcon_s42", "race_no_supcon_s2026")


def resolve_config(base, model_type=None, seed=None, use_supcon=None):
    """CLI overrides JSON independently; the old no-SupCon alias is accepted.

    --model-type race chooses only the architecture. Use --supcon to enable
    the objective, or set use_supcon=true in JSON. Nothing silently rewrites it.
    """
    config = copy.deepcopy(base)
    if model_type is not None:
        config["model_type"] = model_type
    if seed is not None:
        config["seed"] = seed
    if config["model_type"] == "race_no_supcon":
        if use_supcon is True:
            raise ValueError("race_no_supcon conflicts with --supcon; use --model-type race")
        config["model_type"] = "race"
        config["use_supcon"] = False
    if use_supcon is not None:
        config["use_supcon"] = use_supcon
    if config["model_type"] not in ("roberta", "race"):
        raise ValueError("model_type must be roberta or race")
    if type(config.get("use_supcon")) is not bool:
        raise ValueError("use_supcon must be a JSON boolean")
    if config["use_supcon"] and (config["supcon_weight"] <= 0 or config["supcon_temperature"] <= 0):
        raise ValueError("Enabled SupCon requires positive weight and temperature")
    return config


def condition_name(config):
    architecture = "race" if config["model_type"] == "race_no_supcon" else config["model_type"]
    return next(name for name, pair in CONDITIONS.items()
                if pair == (architecture, config["use_supcon"]))


def recipe_for_run(base, run):
    kind, seed = run.rsplit("_s", 1)
    architecture, supcon = CONDITIONS[kind]
    return resolve_config(base, architecture, int(seed), supcon)


def training_arguments(run):
    kind, seed = run.rsplit("_s", 1)
    architecture, supcon = CONDITIONS[kind]
    return ["--model-type", architecture, "--supcon" if supcon else "--no-supcon", "--seed", seed]
