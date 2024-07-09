"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    python3 03_eval_finetuned.py --finetuned_path=<path_to_finetuned_aloha_checkpoint>
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
from envs.aloha_sim_env import AlohaGymEnv  # noqa

from octo.model.octo_model_torch import OctoModel
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

def main(_):
    print("hi, start")
    # setup wandb for logging

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
    
    print("try loading some data")
    f = open(data_args.src_filepath, 'r')
    lines = f.readlines()
    f.close()


    for line in lines:
        '''
        1. encode the images and actions
        src file should contains the following entry
        trajectory_id, frame_number, task_description, scene_description, input_clip_description
        image_indices, actions
        '''
        instance_data = json.loads(line) 

        trajectory_id, view = instance_data['trajectory_id'], instance_data['view']
        
        '''
        save_dir = os.path.join(data_args.save_dir, f'{trajectory_id}_{view}')
        save_path = os.path.join(save_dir, 'results.json')
        save_image_dir = os.path.join(save_dir, 'images_pred')
        save_image_dir_gt = os.path.join(save_dir, 'images_gt')
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_image_dir_gt, exist_ok=True)
        '''

        output_data = {}
        pred_descriptions = {}
        pred_actions = torch.empty(0, 7)

        image_format = '/home/v-rundongluo/robotdata/' + \
                    ('bridge2/images_bridge' if data_args.dataset_name == 'bridge2' else 'RT1/RT1-images') + \
                    '/outputimage_' + str(instance_data['trajectory_id']) + \
                    (('_{}_' + str(instance_data['view'])) if data_args.dataset_name == 'bridge2' else '_{}') + \
                    '.png'
        cur_instance_data = {}
        cur_instance_data['task_description'] = instance_data['task_description']
        cur_instance_data['scene_description'] = instance_data['scene_description']
        cur_instance_data['mean'] = instance_data['mean']
        cur_instance_data['std'] = instance_data['std']
        print(f"instance_data['actions'] shape:{np.array(instance_data['actions']).shape}")
        

        for start_frame in [-1] + list(range(0, instance_data['frame_number'], 6))[:-1]:
            print(f"start_frame:{start_frame}")
            if start_frame != -1:
                cur_instance_data['image_paths'] = [image_format.format(x) for x in instance_data['image_indices'][start_frame:start_frame+6]]
                cur_instance_data['actions'] = instance_data['actions'][start_frame-1:start_frame+5] if start_frame > 0 else \
                                                [[0. for _ in range(6)] + [instance_data['actions'][0][-1]]] + instance_data['actions'][start_frame:start_frame+5]
                cur_instance_data['clip_description'] = instance_data['descriptions'][str(start_frame+5)]
            else:
                cur_instance_data['image_paths'] = [image_format.format(instance_data['image_indices'][0])] * 6
                cur_instance_data['actions'] = [[0. for _ in range(6)] + [instance_data['actions'][0][-1]]] * 6
                cur_instance_data['clip_description'] = ''
            print(f"cur_instance_data keys:{cur_instance_data.keys()}")
            # call the models, override original actions and clip description with the predicted ones
            cur_instance_data = pipeline.call_models(cur_instance_data, model_vq, model_vla, tokenizer, TATSModelArguments, DataArguments, device)
            pred_descriptions[start_frame+11 if start_frame!=-1 else 5] = cur_instance_data['clip_description']
            
            pred_actions = torch.cat((pred_actions, (torch.tensor(cur_instance_data['actions']) * torch.tensor(instance_data['std']) + torch.tensor(instance_data['mean']))), dim=0)
            print(f"pred_actions:{pred_actions.shape}")
            print(f"instance_data['mean']:{torch.tensor(instance_data['mean']).shape}")
            print(f"cur_instance_data['actions']:{torch.tensor(cur_instance_data['actions']).shape}")
            # save the frames

            '''
            for i, img in enumerate(cur_instance_data['images']):
                img = (img + 0.5).clamp(0,1).cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                #img.save(os.path.join(save_image_dir, f'{start_frame+6+i if start_frame!=-1 else i}.png'))
            '''
            break

        output_data['trajectory_id'] = instance_data['trajectory_id']
        output_data['view'] = instance_data['view']
        output_data['task_description'] = instance_data['task_description']
        output_data['scene_description'] = instance_data['scene_description']
        stacked_descriptions = {}
        for key, value in pred_descriptions.items():
            stacked_descriptions[int(key)] = {'gt': instance_data['descriptions'][str(key)], 'pred': value}
        output_data['descriptions'] = stacked_descriptions
        # output_data['pred_descriptions'] = pred_descriptions
        output_data['pred_actions'] = pred_actions.tolist()

        print("let's see the output and actions:", output_data.keys(), output_data['pred_actions'])



    env = gym.make("aloha-sim-cube-v0")

    # wrap env to normalize proprio
    #env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    #env = HistoryWrapper(env, horizon=1)
    #env = RHCWrapper(env, exec_horizon=50)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called

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

    # running rollouts
    for _ in range(3):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = [obs["image_primary"][0]]
        episode_return = 0.0
        while len(images) < 400:
            '''
            src file should contains the following entry
            trajectory_id, frame_number, task_description, scene_description, input_clip_description
            image_indices, actions
            '''

            #action = model.predict(obs)
            # Get an action from the model based on observation and tasks
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            #cur_instance_data = call_models(cur_instance_data, model_vq, model_vla, tokenizer, tats_args, data_args, device)
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            # Apply the action to the environment
            obs, reward, done, trunc, info = env.step(actions) 
            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )


if __name__ == "__main__":
    app.run(main)

'''
import gym

# Assuming 'MyModel' is a class that implements the model and has a 'predict' method
class MyModel:
    def __init__(self):
        # Initialize model parameters, load weights, etc.
        pass

    def predict(self, observation):
        # Process the observation and generate an action
        # For this example, let's assume it outputs a random action
        action = env.action_space.sample()
        return action

# Initialize the environment
env = gym.make("aloha-sim-cube-v0")

# Initialize the model
model = MyModel()

# Reset the environment to get the initial observations
obs, info = env.reset()

# Variable to keep track of the total reward
total_reward = 0

# Run the test episode
done = False
while not done:
    # Get an action from the model
    action = model.predict(obs)

    # Apply the action to the environment
    obs, reward, done, _, info = env.step(action)

    # Accumulate the reward
    total_reward += reward

print(f'Total reward for the episode: {total_reward}')

'''