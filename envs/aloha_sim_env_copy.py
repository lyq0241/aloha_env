import copy
from typing import List

import dlimp as dl
import gym
import jax.numpy as jnp
import numpy as np


#This is the aloha sim env used for zero-shot evaluation of our mistral_vla model
from sim_env import BOX_POSE, make_sim_env


class AlohaGymEnvCopy(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary"]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images,
                "observations":obs}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info
    #obs, reward, done, trunc, info

    def reset(self, **kwargs):
        # sample new box pose
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]
        ranges = np.vstack([x_range, y_range, z_range])
        cube_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])
        cube_quat = np.array([1, 0, 0, 0])   
        BOX_POSE[0] = np.concatenate([cube_position, cube_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images,
                "observations":obs}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        obs_img_names = ["primary", "wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        return {
            "language_instruction": ["Move the can to the back right corner of the table"],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


# register gym environments
    gym.register(
        "aloha-sim-cube-copy-v0",
        entry_point=lambda: AlohaGymEnvCopy(
            make_sim_env("sim_transfer_cube"), camera_names=["top"]
        ),
    )

'''
observation
{
    "image_primary": np.array([...]),  # An array representing the primary image observation
    "proprio": np.array([...]),        # An array representing proprioceptive data
}

ts = {
    "observation": {
        "images": {
            "top": np.array(...),  # Image from the top camera
            "wrist": np.array(...),  # Image from the wrist camera
        },
        "qpos": np.array(...),  # Proprioceptive data (e.g., joint positions)
    },
    "reward": 1.0,  # Reward received for the action
    "done": False,  # Episode has not ended
    "trunc": False,  # Episode was not truncated
    "info": {}  # Additional information (if any)
}
'''