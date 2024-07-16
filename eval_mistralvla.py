"""
This script demonstrates how to load and rollout our mistral_vla model.
We use the mistral_vla model from "robot-pipeline" and we use aloha_sim_env_copy as the sim environment.

run this in the terminal before you run eval_mistralvla.py:
export MUJOCO_GL=egl
echo $MUJOCO_GL  
"""
from functools import partial
import sys
import argparse

import json
import os
from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb
import torch

from configs import H4ArgumentParser, DataArguments, VLAModelArguments, TATSModelArguments
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from PIL import Image
from time import time

sys.path.append("path/to/your/act")

# keep this to register ALOHA sim env
from envs.aloha_sim_env_copy import AlohaGymEnvCopy  # noqa

#from octo_modified.model.octo_model_torch import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng
script_path = '/home/yunqiliu/octo/examples/robot-pipeline'
if script_path not in sys.path:
    sys.path.append(script_path)
import pipeline_simulation_with_gt as pipeline
# Now you can import the script
script_path = '/home/yunqiliu/octo/examples/robot-pipeline/tokenizer'
if script_path not in sys.path:
    sys.path.append(script_path)
from tokenizer import VQGANVisionActionEval

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)

#fn:action->action
def sample_actions(actions, metadata, model_vq, model_vla, tokenizer, device):
    print("start sampling actions")
    output_data = {}
    pred_descriptions = {}
    pred_actions = torch.empty(0, 7)
    #这下面的image_format没有用，我们的image是通过aloha sim env进行initialize
    image_format = '/home/v-rundongluo/robotdata/' + \
                    ('bridge2/images_bridge') + \
                    '/outputimage_' + str(metadata['trajectory_id']) + \
                    (('_{}_' + str(metadata['view']))) + \
                    '.png'

    cur_instance_data={}
    cur_instance_data['task_description'] = metadata['task_description']
    cur_instance_data['scene_description'] = metadata['scene_description']
    cur_instance_data['mean'] = metadata['mean']
    cur_instance_data['std'] = metadata['std']

    for start_frame in [-1] + list(range(0, metadata['frame_number'], 6))[:-1]:
        # call the models, override original actions and clip description with the predicted ones
        cur_instance_data['image_paths'] = [image_format.format(metadata['image_indices'][0])] * 6
        cur_instance_data['actions'] = [[0. for _ in range(6)] + [actions[0][-1]]] * 6
        cur_instance_data['clip_description'] = ''
        print("call model in sample actions function")
        cur_instance_data = pipeline.call_models(cur_instance_data, model_vq, model_vla, tokenizer, TATSModelArguments, DataArguments, device)
        print("finish calling ")
        pred_descriptions[start_frame+11 if start_frame!=-1 else 5] = cur_instance_data['clip_description']
        pred_actions = torch.cat((pred_actions, (torch.tensor(cur_instance_data['actions']) * torch.tensor(metadata['std']) + torch.tensor(metadata['mean']))), dim=0)
        torch.cuda.empty_cache()
        break
        # save the frames

    output_data['pred_actions'] = pred_actions.tolist()

    print("let's see the output and actions:", output_data.keys(), output_data['pred_actions'])
    return output_data['pred_actions']




def main(_):
    print("hi, start")
    # setup wandb for logging
    wandb.init(name="eval_aloha_on_model", project="octo")
    # load finetuned model
    #logging.info("Loading finetuned model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = H4ArgumentParser((VLAModelArguments, DataArguments, TATSModelArguments))
    vla_args, data_args, tats_args = parser.parse()
    model_vq = VQGANVisionActionEval(tats_args).to(device)
    model_kwargs = dict(
        use_flash_attention_2=vla_args.use_flash_attention_2,
        torch_dtype=getattr(torch, vla_args.torch_dtype)
    )
    model_vla = AutoModelForCausalLM.from_pretrained(vla_args.model_name_or_path, **model_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(vla_args.model_name_or_path)
    print("finish loading the model")
    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    print("try loading some data for initialization")
    f = open(data_args.src_filepath, 'r')
    lines = f.readlines()
    f.close()
    metadata={}
    #metadata用来储存在运行过程中不变的量。action是变化的，不过metadata["action"]代表的是action的初始值

    for line in lines:
        '''
        1. encode the images and actions
        src file should contains the following entry
        trajectory_id, frame_number, task_description, scene_description, input_clip_description
        image_indices, actions
        '''
        instance_data = json.loads(line) 
        trajectory_id, view = instance_data['trajectory_id'], instance_data['view']   
  
        image_format = '/home/v-rundongluo/robotdata/' + \
                    ('bridge2/images_bridge' if data_args.dataset_name == 'bridge2' else 'RT1/RT1-images') + \
                    '/outputimage_' + str(instance_data['trajectory_id']) + \
                    (('_{}_' + str(instance_data['view'])) if data_args.dataset_name == 'bridge2' else '_{}') + \
                    '.png'
        #image_format没有用
        
        
        metadata['task_description'] = instance_data['task_description']
        metadata['scene_description'] = instance_data['scene_description']
        metadata['mean'] = instance_data['mean']
        metadata['std'] = instance_data['std']
        metadata["frame_number"]=instance_data["frame_number"]

        for start_frame in [-1] + list(range(0, instance_data['frame_number'], 6))[:-1]:
            if start_frame != -1:
                metadata['image_paths'] = [image_format.format(x) for x in instance_data['image_indices'][start_frame:start_frame+6]]
                metadata['actions'] = instance_data['actions'][start_frame-1:start_frame+5] if start_frame > 0 else \
                                                [[0. for _ in range(6)] + [instance_data['actions'][0][-1]]] + instance_data['actions'][start_frame:start_frame+5]
                metadata['clip_description'] = instance_data['descriptions'][str(start_frame+5)]
            else:
                metadata['image_indices']=instance_data['image_indices']
                metadata['image_paths'] = [image_format.format(instance_data['image_indices'][0])] * 6
                metadata['actions'] = [[0. for _ in range(6)] + [instance_data['actions'][0][-1]]] * 6
                metadata['clip_description'] = ''
            break
        metadata['trajectory_id'] = instance_data['trajectory_id']
        metadata['view'] = instance_data['view']
    print("finish loading the data for initialization")
    print("metadata is", metadata.keys())

#metadata["actions"] is the initial action


# action->action
    
    

    env = gym.make("aloha-sim-cube-copy-v0")
    print("env=gym.make.....")

    # wrap env to normalize proprio
    #env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    #env = HistoryWrapper(env, horizon=1)
    #env = RHCWrapper(env, exec_horizon=50)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    '''
    if "action" in model.dataset_statistics:
        print("Key 'action' found in dataset_statistics")
    else:
        print("Key 'action' not found in dataset_statistics")

    # Apply the provided unnormalization_statistics to ensure the actions are scaled correctly.
    # Include randomness in the action sampling process if needed (due to the RNG supplied by supply_rng
    
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], #dataset name here
        ),
    )
    '''

    # running rollouts
    for _ in range(1):
        obs, info = env.reset()
        print("obs, info = env.reset()")
        actions = metadata["actions"]

        # create task specification --> use model utility to create task dict with correct entries
        #language_instruction = env.get_task()["language_instruction"]
        #task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = []
        episode_return = 0.0
        while len(images) < 10:
            '''
            src file should contains the following entry
            trajectory_id, frame_number, task_description, scene_description, input_clip_description
            image_indices, actions
            '''

            #action = model.predict(obs)
            # Get an action from the model based on observation and tasks
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            #cur_instance_data = call_models(cur_instance_data, model_vq, model_vla, tokenizer, tats_args, data_args, device)
            print("iteration starts: we will sample actions")
            actions = sample_actions(actions, metadata, model_vq, model_vla, tokenizer, device)
            #actions (6, 7)
            print("finish sampling actions")

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            # Apply the action to the environment
            action = torch.cat((torch.tensor(actions[0]), torch.tensor(actions[1])))
            #print("this is action:", action)
            obs, reward, done, trunc, info = env.step(action)
            print("the shape of torch.tensor(info['observations']['image_primary']) is:", np.array(info["observations"]["image_primary"]).shape)
            images.append(info["observations"]["image_primary"])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")
         
    
        # log rollout video to wandb -- subsample temporally 2x for faster logging
        print("the shape:", np.array(images).shape)
        
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )
    

if __name__ == "__main__":
    app.run(main)

