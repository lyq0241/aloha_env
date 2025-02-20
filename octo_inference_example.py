
from octo.model.octo_model import OctoModel
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import wandb

print("hi, start")
wandb.init(name="eval_aloha_example", project="octo")
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
print(model.dataset_statistics["bridge_dataset"]['proprio']['mean'])
print("finishes loading the model")

# # download one example BridgeV2 image
print("download image")
IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
plt.imshow(img)
print("show image")

# create obs & task dict, run inference
import jax

# add batch + time horizon 1
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["pick up the fork"])
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
print(action.shape)   # [batch, action_chunk, action_dim]

import cv2
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import rlds
import mediapy as media
from PIL import Image
from IPython import display

# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
ds = builder.as_dataset(split='train[:1]')

# sample episode + resize to 256x256 (default third-person cam resolution)
episode = next(iter(ds))
print(episode)

steps = list(episode['steps'])
images = [cv2.resize(np.array(step['observation']['image']), (256, 256)) for step in steps]

# extract goal image & language instruction
goal_image = images[-1]
language_instruction = steps[0]['observation']['natural_language_instruction'].numpy().decode()

# visualize episode
#print(f'Instruction: {language_instruction}')
media.show_video(images, fps=10)

WINDOW_SIZE = 2

# create `task` dict
task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
task = model.create_tasks(texts=[language_instruction])                # for language conditioned

# run inference loop, this model only uses single image observations for bridge
# collect predicted and true actions
pred_actions, true_actions = [], []
for step in tqdm.tqdm(range(0, len(images) - WINDOW_SIZE + 1)):
    input_images = np.stack(images[step : step + WINDOW_SIZE])[None]
    observation = {
        'image_primary': input_images,
        'pad_mask': np.array([[True, True]]),
    }

    # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
    norm_actions = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
    norm_actions = norm_actions[0]   # remove batch
    actions = (
        norm_actions * model.dataset_statistics["bridge_dataset"]['action']['std']
        + model.dataset_statistics["bridge_dataset"]['action']['mean']
    )

    pred_actions.append(actions)
    true_actions.append(np.concatenate(
        (
            steps[step+1]['action']['world_vector'],
            steps[step+1]['action']['rotation_delta'],
            np.array(steps[step+1]['action']['open_gripper']).astype(np.float32)[None]
        ), axis=-1
    ))

    import matplotlib.pyplot as plt

ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

# build image strip to show above actions
img_strip = np.concatenate(np.array(images[::3]), axis=1)

# set up plt figure
figure_layout = [
    ['image'] * len(ACTION_DIM_LABELS),
    ACTION_DIM_LABELS
]
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

# plot actions
pred_actions = np.array(pred_actions).squeeze()
true_actions = np.array(true_actions).squeeze()
for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
  # actions have batch, horizon, dim, in this example we just take the first action for simplicity
  axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
  axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
  axs[action_label].set_title(action_label)
  axs[action_label].set_xlabel('Time in one episode')

axs['image'].imshow(img_strip)
axs['image'].set_xlabel('Time in one episode (subsampled)')
plt.legend()
# Save the figure
plt.savefig("actions_plot.png")

# Log the figure to WandB
wandb.log({"actions_plot": wandb.Image("actions_plot.png")})
print("finished")