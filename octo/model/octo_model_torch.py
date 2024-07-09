import json
import logging
import os
import torch
import numpy as np
from typing import Optional
from octo.utils.spec import ModuleSpec
from octo.model.octo_module import OctoModule

class OctoModel:
    def __init__(self, model, text_processor, config, example_batch, dataset_statistics):
        self.model = model
        self.text_processor = text_processor
        self.config = config
        self.example_batch = example_batch
        self.dataset_statistics = dataset_statistics
        self.model.eval()

    @classmethod
    def load_pretrained(cls, checkpoint_path: str, step: Optional[int] = None) -> "OctoModel":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if checkpoint_path and checkpoint_path.startswith("hf://"):
            if step:
                raise ValueError(
                    "You can't set config['pretrained_step'] when loading from HuggingFace."
                )
            checkpoint_path = _download_from_huggingface(
                checkpoint_path.removeprefix("hf://")
            )

        # load config
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # shim to support old configs
        #if "pred_horizon" in config["model"]["heads"]["action"]["kwargs"]:
            #config["model"]["heads"]["action"]["kwargs"]["action_horizon"] = config["model"]["heads"]["action"]["kwargs"].pop("pred_horizon")

        # load example batch
        example_batch_path = os.path.join(checkpoint_path, "example_batch.msgpack")
        with open(example_batch_path, "rb") as f:
            example_batch = torch.load(f)
        # shim for migrating from "tasks" to "task"
        if "tasks" in example_batch:
            example_batch["task"] = example_batch.pop("tasks")

        logging.debug(
            "Model was trained with observations: %s",
            {k: v.shape for k, v in example_batch["observation"].items()}
        )
        logging.debug(
            "Model was trained with tasks: %s",
            {k: v.shape for k, v in example_batch["task"].items()}
        )

        # load dataset statistics
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
        with open(dataset_statistics_path, "r") as f:
            dataset_statistics = json.load(f)
            dataset_statistics = {k: np.array(v) for k, v in dataset_statistics.items()}

        # Create model instance
        model = OctoModule.create(**config["model"])

        # shim for old checkpoints
        if "timestep_pad_mask" not in example_batch["observation"]:
            example_batch["observation"]["timestep_pad_mask"] = example_batch["observation"]["pad_mask"]

        # Load the model parameters
        params_path = os.path.join(checkpoint_path, "model_params.pth")
        model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))

        # Initialize text processor if present in config
        if config.get("text_processor") is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None

        return cls(
            model=model,
            text_processor=text_processor,
            config=config,
            example_batch=example_batch,
            dataset_statistics=dataset_statistics
        )

def _download_from_huggingface(huggingface_repo_id: str):
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder
