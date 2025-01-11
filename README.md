## Simulation Environment to Eval and Finetune VLA Models
![Rollout Demo](https://github.com/lyq0241/aloha_env/raw/main/assets/rollout_video_0_36ce4a4640ac0a49b2b9.gif)

The video above demonstrates the inference process of a VLA model using the Aloha simulation environment.  
The VLA models are fine-tuned with the Aloha Sim dataset and simulation environment to perform bimanual tasks effectively.

|                                                                      |                                                                                                                 |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [OCTO Finetuning](finetune_new_observation_action.py)    | Minimal example for finetuning a pre-trained OCTO models on a small dataset with new observation + action space |
| [OCTO Rollout](eval_octo_model.py)                        | Run a rollout of a pre-trained OCTO policy in a Gym environment                                                 |
| [Mistral_vla_Eval](eval_mistralvla.py)                               | evaluate our pretrained vla model   


##
