import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, MistralModel, PhiModel
from transformers import TrainerCallback
from transformers import Phi3Config, Phi3ForCausalLM, LlamaTokenizer
from transformers import MistralConfig, MistralForCausalLM
from octo.data.dataset import make_single_dataset

sys.path.append('.')
script_path = '/home/yunqiliu/octo/examples/vla/src'
if script_path not in sys.path:
    sys.path.append(script_path)
from src import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig, get_checkpoint, get_datasets
from src import get_VLA_dataset

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)
device=torch.device('cuda')

logger = logging.getLogger(__name__)

def main():
    try:
        print('MASTER_ADDR', os.environ['MASTER_ADDR'])
        print('MASTER_PORT', os.environ['MASTER_PORT'])
        print('NODE_RANK', os.environ['NODE_RANK'])
        print('LOCAL_RANK', os.environ['LOCAL_RANK'])
        print('RANK', os.environ['RANK'])
        print('WORLD_SIZE', os.environ['WORLD_SIZE'])
    except:
        pass

    from huggingface_hub import login
    login(token='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')

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    # The visual modality has 2048 (16384) tokens, and the action modality has 256 tokens, add them to the tokenizer
    # Add special tokens for the visual and action modalities, 
    #     including <bots_i>, <eots_i>, <botp_i>, <eotp_i>, <bov_i>, <eov_i>, <boa_i>, <eoa_i>,
    #               <botp_o>, <eotp_o>, <bov_o>, <eov_o>, <boa_o>, <eoa_o>
    # In total 16384 + vocab_size
    ################
    if model_args.disable_auto_config:
        # both phi3 and mistral use the LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    vocab_size = len(tokenizer)
    # add eos token when when calling tokenizer
    visual_action_tokens_to_add = ['<va' + str(i) + '>' for i in range(0, data_args.num_visual_action_tokens)]
    num_added_visual_action_tokens = tokenizer.add_special_tokens({'additional_special_tokens': visual_action_tokens_to_add})
    special_tokens = ['<bott_i>', '<eott_i>', # task text
                        '<bots_i>', '<eots_i>', # scene text
                        '<botp_i>', '<eotp_i>', # policy text
                        '<bov_i>', '<eov_i>', '<boa_i>', '<eoa_i>', # vision and action tokens
                        '<botp_o>', '<eotp_o>', # output policy text
                        '<bov_o>', '<eov_o>', '<boa_o>', '<eoa_o>'] # output vision and action tokens
    num_added_special_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # For SFT training, padding should be on the right (if overflow occurs)
    tokenizer.padding_side = data_args.padding_side

    #######################
    # Load and pre-process the dataset
    #######################
    #Load the finetuning dataset from aloha
    ###########################################
    logger.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=50,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )
    ############################################3
    ####################################################################
    

    def preprocess_func(example):
        example_new = {}
        example_new['text'] = example['input'] + example['output']
        return example_new

    # only take a little samples for debug
    if training_args.debug:
        print('Debug mode, only take a little samples for training and evaluation')
        train_dataset = train_dataset.select(range(2000))
        eval_dataset = eval_dataset.select(range(100))

    train_dataset = eval_dataset.map(
        preprocess_func,
        num_proc=data_args.preprocessing_num_workers,
        desc="Preprocessing testing dataset",
    )

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        # take a sample from the dataset (iteratable)
        if type(train_dataset) == datasets.IterableDataset:
            for i, example in enumerate(train_dataset.take(3)):
                logger.info(f"Sample {i}: {example['text']}")
        else:
            for i in range(3):
                logger.info(f"Sample {i}: {train_dataset[i]}")
    
    # input always ends by <eoa_i>, use <eoa_i> as the response template
    response_template_id = tokenizer.convert_tokens_to_ids(['<eoa_i>'])
    data_collator = DataCollatorForCompletionOnlyLM(response_template_id, tokenizer=tokenizer)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    # use float16 (V100 does not support bfloat16)
    torch_dtype = torch.float16 if training_args.fp16 else torch.float32

    model_kwargs = dict(
        # revision=model_args.model_revision,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        # trust_remote_code=True,
        use_cache=False if training_args.gradient_checkpointing else True
    )

    if model_args.disable_auto_config:
        if model_args.model_type == 'phi3':
            # configuration = Phi3Config.from_pretrained()
            model = Phi3ForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        elif model_args.model_type == 'mistral':
            # configuration = MistralConfig.from_pretrained(model_args.model_name_or_path)
            model = MistralForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128) # pad to multiple of 128 to improve performance

    ########################
    # Initialize the Trainer
    ########################

    class PrintCallback(TrainerCallback):
        def on_evaluation(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            # print whether this process should save the checkpoint
            print(f'Process {args.local_rank} should save checkpoint: {args.should_save}')

    trainer = SFTTrainer(
        model=model,
        args=training_args,   
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        data_collator=data_collator,
        callbacks=[PrintCallback()] if training_args.debug else None,
        max_seq_length=training_args.max_seq_length,
        dataset_kwargs=training_args.dataset_kwargs,
    )

    ###############
    # Training loop
    ###############

    
    checkpoint = "/home/yunqiliu/octo/checkpoint-40000"
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()

#python finetune.py --output_dir="/home/yunqiliu/octo/examples/vla/finetune_checkpoint"
