# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        # only parse the commands after the script name
        script_name = filter(lambda x: x.endswith(".py"), sys.argv)
        sys.argv = list(script_name) + sys.argv[1:]
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[-1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[-1]), sys.argv[1:-1])
        # elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
        #     output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


@dataclass
class VLAModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    base_model_name: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "The base model that the model is finetuned from."},
    )

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adatpers.")},
    )
    model_name_or_path: Optional[str] = field(
        default="/home/yunqiliu/octo/checkpoint-40000",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    disable_auto_config: bool = field(default=False, metadata={"help": "Disable auto config."})
    model_type: Optional[str] = field(default="mistral", 
        metadata={"help": "The model type.", "choices": ["mistral", "phi3"],
        })

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_type: str = field(
        default='dataset',
        metadata={"help": "The type of dataset to use for training.",
                  "choices": ['dataset', 'iterable_dataset']}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": ("The number of processes to use for data preprocessing.")},
    )
    data_root: Optional[str] = field(
        default=None,
        metadata={"help": "The root directory of the data."}
    )
    padding_side: Optional[str] = field(
        default='right', metadata={
            "help": "Truncation side to use for the tokenizer.",
            "choices": ["right", "left"],
        }
    )
    num_visual_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": ("The number of visual tokens to use for the model.")},
    )
    num_action_tokens: Optional[int] = field(
        default=256,
        metadata={"help": ("The number of action tokens to use for the model.")},
    )
    num_action_tokens: Optional[int] = field(
        default=256,
        metadata={"help": ("The number of action tokens to use for the model.")},
    )
    num_input_frames: Optional[int] = field(
        default=6,
        metadata={"help": ("The number of input frames to use for the model.")},
    )
    num_output_frames: Optional[int] = field(
        default=1,
        metadata={"help": ("The number of output frames to use for the model.")},
    )
    num_visual_action_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": ("The number of visual tokens to use for the model.")},
    )
    # argument that accepts a list of strings
    static_video_description: Optional[List[str]] = field(
        default_factory=lambda: [""],
        metadata={"help": "The static frame description."}
    )
    data_debug: bool = field(default=False, metadata={"help": "Debug mode for data loading."})
    save_dir: Optional[str] = field(default=None, metadata={"help": "The path to save the predictions."})
    action_before_vision: bool = field(default=False, metadata={"help": "Whether to use vision before action."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "The start index for the dataset."})
    end_idx: Optional[int] = field(default=None, metadata={"help": "The end index for the dataset."})
    src_filepath: Optional[str] = field(default="/home/yunqiliu/octo/examples/robot-pipeline/test_data/sample-bridge.json", metadata={"help": "The source file path."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset.", "choices": ["rt1", "bridge2"]})

@dataclass
class TATSModelArguments:
    """
    Arguments related to the TATS tokenizer
    """

    embedding_dim: int = field (
        default=256,
        metadata={"help": "The dimension of the embeddings."}
    )
    n_codes: int = field (
        default=16384,
        metadata={"help": "The number of codes in the codebook."}
    )
    n_hiddens: int = field (
        default=32,
        metadata={"help": "The number of hidden units in the model."}
    )
    downsample: Tuple[int] = field (
        default=(2, 16, 16),
        metadata={"help": "The downsample factor for the model."}
    )
    disc_channels: int = field (
        default=64,
        metadata={"help": "The number of channels in the discriminator."}
    )
    disc_layers: int = field (
        default=3,
        metadata={"help": "The number of layers in the discriminator."}
    )
    disc_loss_type: str = field (
        default='hinge',
        metadata={"help": "The loss type for the discriminator."}
    )
    i3d_feat: bool = field (
        default=False,
        metadata={"help": "Whether to use I3D features."}
    )
    restart_thres: float = field (
        default=1.0,
        metadata={"help": "The restart threshold."}
    )
    no_random_restart: bool = field (
        default=False,
        metadata={"help": "Whether to use random restart."}
    )
    norm_type: str = field (
        default='batch',
        metadata={"help": "The normalization type."}
    )
    padding_type: str = field (
        default='replicate',
        metadata={"help": "The padding type."}
    )
    action_dim: Tuple[int] = field (
        default=(1, 1, 1, 1, 1, 1, 1),
        metadata={"help": "The number of action dimensions."}
    )
    action_activation: Tuple[str] = field (
        default=('none', 'none', 'none', 'none', 'none', 'none', 'sigmoid'),
        metadata={"help": "The activation function for the action."}
    )
    action_hidden_dim: int = field (
        default=128,
        metadata={"help": "The hidden dimension for the action."}
    )
    video_action_layers: int = field (
        default=12,
        metadata={"help": "The number of action layers."}
    )
    sequence_length: int = field (
        default=6,
        metadata={"help": "The sequence length."}
    )
    resolution: int = field (
        default=256,
        metadata={"help": "The resolution of the image."}
    )
    image_channels: int = field (
        default=3,
        metadata={"help": "The number of channels in the image."}
    )
    weight_path: Optional[str] = field (
        default=None,
        metadata={"help": "The path to the weight file."}
    )
    wo_transformer_residual: bool = field (
        default=True,
        metadata={"help": "Whether to use transformer residual."}
    )
    