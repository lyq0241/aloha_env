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

run 
export MUJOCO_GL=egl
echo $MUJOCO_GL
"""
from functools import partial
import sys

from absl import app, flags, logging
import gym
import jax
import numpy as np
import torch
import wandb
import jax.numpy as jnp

sys.path.append("path/to/your/act")

# keep this to register ALOHA sim env
from envs.aloha_sim_env import AlohaGymEnv  # noqa

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)

def main(_):
    print("hi, start")
    # setup wandb for logging
    wandb.init(name="eval_aloha_octo-4000", project="octo")

    # load finetuned model
    #logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
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
    '''policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"], #dataset name here
        ),
    )'''

    # running rollouts
    for _ in range(1):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = []
        episode_return = 0.0
        while len(images) < 1010:
            if(len(images)%10==0):
                print(f"len(images):{len(images)}, episode_return:{episode_return}")

            #action = model.predict(obs)
            # Get an action from the model based on observation and tasks
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = model.sample_actions(observations=obs,
                                           tasks=task,
                                           unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
                                           rng=jax.random.PRNGKey(0))
            #(1, 4, 7)
            actions = actions[0] 
            #(4, 7)
            actions = jnp.clip(actions, -1.0, 1.0)
            actions = jnp.array(actions)
            actions_np = np.array(actions)
            action = torch.cat((torch.tensor(actions_np[0]), torch.tensor(actions_np[0])))
            #(14,)
            #print(f"Clipped action: {action}")
            #print("this is action:", action)
            obs, reward, done, trunc, info = env.step(action)
            print(f"len(images):{len(images)}, reward:{reward}")      

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            # Apply the action to the environment
            images.append(info["observations"]["image_primary"][0][1])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")
        print("the shape:", np.array(images).shape)
        #(t, 256, 256, 3)
        #(t, 3, 256, 256)
        print(np.array(images).transpose(0, 3, 1, 2)[::2].shape)

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