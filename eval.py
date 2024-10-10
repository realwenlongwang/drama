import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss
import yaml
from train import parse_args_and_update_config, DotDict, build_world_model, build_agent
from utils import WandbLogger
import pandas as pd
import wandb

def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent, logger):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs=num_envs)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    atari_benchmark_df = pd.read_csv("atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
    atari_pure_name = env_name.split('/')[-1].split('-')[0]
    game_benchmark_df = atari_benchmark_df.get(atari_pure_name)

    final_scores = []
    final_normalised_scores = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    episode_idx = 0
    score_table = {"episode": [], "episode/score": [], "episode/normalised_score": []}
    for algorithm in game_benchmark_df.index[2:]:
        score_table[f"benchmark/normalised_{algorithm}_score"] = []
    with tqdm(total=num_episode, desc="Episodes") as episode_pbar:
        while True:
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                    # action = np.array([action], dtype=int)
                    # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(world_model.device))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).to(world_model.device)
                    # current_obs_tensor = rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255
                    if world_model.model == 'Transformer':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                        # prior_flattened_sample, last_dist_feat = world_model.calc_last_post_feat(context_latent, model_context_action, current_obs_tensor)
                    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        # prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent[:,-1:], model_context_action[:,-1:], inference_params)
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                        # prior_flattened_sample, last_dist_feat = world_model.calc_last_post_feat(context_latent, model_context_action, current_obs_tensor)
                    #TODO: Change this to use the current embedding.
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=True
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).to(world_model.device), "B H W C -> B 1 C H W")/255)
            context_action.append(action)

            obs, reward, done, truncated, info = vec_env.step(action)
            # cv2.imshow("current_obs", process_visualize(obs[0]))
            # cv2.waitKey(10)
            # update current_obs, current_info and sum_reward
            sum_reward += reward
            current_obs = obs
            current_info = info

            done_flag = np.logical_or(done, truncated)
            if done_flag.any():
                # inference_params = InferenceParams(max_seqlen=1, max_batch_size=1)
                for i in range(num_envs):
                    if done_flag[i]:
                        episode_score = sum_reward[i]
                        normalised_score = (episode_score - game_benchmark_df['Random']) / (game_benchmark_df['Human'] - game_benchmark_df['Random'])
                        
                        score_table["episode"].append(episode_idx)
                        score_table["episode/score"].append(episode_score)
                        score_table["episode/normalised_score"].append(normalised_score)

                        for algorithm in game_benchmark_df.index[2:]:
                            denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                            # Check if the denominator is zero
                            if denominator != 0:
                                normalised_score = (sum_reward[i] - game_benchmark_df['Random']) / denominator
                                score_table[f"benchmark/normalised_{algorithm}_score"].append(normalised_score)
                            else:
                                score_table[f"benchmark/normalised_{algorithm}_score"].append(None)

                        final_scores.append(episode_score)
                        final_normalised_scores.append(normalised_score)        
                        sum_reward[i] = 0
                        episode_idx += 1
                        episode_pbar.update(1)  # Update the episode progress bar
                        if len(final_scores) == num_episode:
                            print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_scores)}" + colorama.Style.RESET_ALL)
                            logger.log('episode/average score', np.mean(final_scores), global_step=0)
                            logger.log('episode/average normalised score', np.mean(final_normalised_scores), global_step=0)
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(score_table)
                            
                            # Log the DataFrame to wandb
                            return df




if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Read the YAML configuration file
    with open('config_files/evaluation.yaml', 'r') as file:
        config = yaml.safe_load(file)
   
    
    # Parse the arguments and update the configuration
    config = parse_args_and_update_config(config)   

    config = DotDict(config)
    
    # parse arguments
    # print(colorama.Fore.RED + str(config) + colorama.Style.RESET_ALL)

    device = torch.device(config.BasicSettings.Device)

    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)

    # getting action_dim with dummy env
    dummy_env = build_single_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize)
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device)
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    agent = build_agent(config, action_dim, device=device)
    config.update_or_create('Models.Agent.ActorParamNum', sum([p.numel() for p in agent.actor.parameters()]))
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model = torch.compile(world_model)
        agent = torch.compile(agent)
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = logger.run.dir

    print('Loading models')
    world_model.load_state_dict(torch.load(f"{config.Evaluate.SavePath}/world_model.pth"))
    agent.load_state_dict(torch.load(f"{config.Evaluate.SavePath}/agent.pth"))
    
    scores_table = eval_episodes(
        num_episode=config.Evaluate.EpisodeNum, env_name=config.BasicSettings.Env_name, 
        num_envs=32, image_size=64, world_model=world_model, agent=agent, logger=logger)

    wandb.log({"score_data": wandb.Table(dataframe=scores_table)})


    wandb.finish()