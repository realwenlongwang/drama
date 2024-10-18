import gymnasium
import argparse
import numpy as np
from einops import rearrange
import torch
from collections import deque
from tqdm import tqdm
import colorama
import os
import pandas as pd

from utils import seed_np_torch, WandbLogger
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.world_models import WorldModel
from mamba_ssm import InferenceParams
from line_profiler import profile
import yaml



def build_single_env(env_name, image_size, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env

@profile
def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, batch_length, logger, epoch, global_step):
    epoch_reconstruction_loss_list = []
    epoch_reward_loss_list = []
    epoch_termination_loss_list = []
    epoch_dynamics_loss_list = []
    epoch_dynamics_real_kl_div_list = []
    epoch_representation_loss_list = []
    epoch_representation_real_kl_div_list = []
    epoch_total_loss_list = []
    for e in range(epoch):
        obs, action, reward, termination = replay_buffer.sample(batch_size, batch_length, imagine=False)
        reconstruction_loss, reward_loss, termination_loss, \
        dynamics_loss, dynamics_real_kl_div, representation_loss, \
        representation_real_kl_div, total_loss = world_model.update(obs, action, reward, termination, global_step=global_step, epoch_step=e, logger=logger)

        epoch_reconstruction_loss_list.append(reconstruction_loss)
        epoch_reward_loss_list.append(reward_loss)
        epoch_termination_loss_list.append(termination_loss)
        epoch_dynamics_loss_list.append(dynamics_loss)
        epoch_dynamics_real_kl_div_list.append(dynamics_real_kl_div)
        epoch_representation_loss_list.append(representation_loss)
        epoch_representation_real_kl_div_list.append(representation_real_kl_div)
        epoch_total_loss_list.append(total_loss)
    if logger is not None:
        logger.log("WorldModel/reconstruction_loss", np.mean(epoch_reconstruction_loss_list), global_step=global_step)
        # logger.log("WorldModel/augmented_reconstruction_loss", augmented_reconstruction_loss.item(), global_step=global_step)
        logger.log("WorldModel/reward_loss",np.mean(epoch_reward_loss_list), global_step=global_step)
        logger.log("WorldModel/termination_loss", np.mean(epoch_termination_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_loss", np.mean(epoch_dynamics_loss_list), global_step=global_step)
        logger.log("WorldModel/dynamics_real_kl_div", np.mean(epoch_dynamics_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/representation_loss", np.mean(epoch_representation_loss_list), global_step=global_step)
        logger.log("WorldModel/representation_real_kl_div", np.mean(epoch_representation_real_kl_div_list), global_step=global_step)
        logger.log("WorldModel/total_loss", np.mean(epoch_total_loss_list), global_step=global_step)    

@profile
@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger, global_step):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()
    #TODO: Use the sampl_reward, termination to create a replay buffer critic loss
    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_context_length, imagine=True)
    if world_model.model == 'Transformer':
        latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data(
            agent, sample_obs, sample_action,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=imagine_batch_length,
            log_video=log_video,
            logger=logger, global_step=global_step
        )
    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
         latent, action, old_logits, context_latent, reward_hat, termination_hat = world_model.imagine_data2(
            agent, sample_obs, sample_action,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=imagine_batch_length,
            log_video=log_video,
            logger=logger, global_step=global_step
        )
    return latent, action, old_logits, context_latent, sample_reward, sample_termination, reward_hat, termination_hat

@profile
def joint_train_world_model_agent(config, logdir,
                                  replay_buffer: ReplayBuffer,
                                  world_model: WorldModel, agent: agents.ActorCriticAgent,
                                  logger):
    os.makedirs(f"{logdir}/ckpt", exist_ok=True)


    env = build_single_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize, seed=config.BasicSettings.Seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{config.BasicSettings.Env_name}" + colorama.Style.RESET_ALL)

    atari_benchmark_df = pd.read_csv("atari_performance.csv", index_col='Task', usecols=lambda column: column in ['Task', 'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone', 'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'])
    atari_pure_name = config.BasicSettings.Env_name.split('/')[-1].split('-')[0]
    game_benchmark_df = atari_benchmark_df.get(atari_pure_name)
    
    sum_reward = 0
    current_ob, current_info = env.reset()
    context_obs = deque(maxlen=config.JointTrainAgent.RealityContextLength)
    context_action = deque(maxlen=config.JointTrainAgent.RealityContextLength)

    # sample and train
    for total_steps in tqdm(range(config.JointTrainAgent.SampleMaxSteps // config.JointTrainAgent.NumEnvs)):
        # sample part >>>
        if replay_buffer.ready('world_model'):
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1).to(world_model.device))
                    model_context_action = np.stack(list(context_action))
                    model_context_action = rearrange(torch.Tensor(model_context_action).to(world_model.device), "L -> 1 L")
                    if world_model.model == 'Transformer':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    elif world_model.model == 'Mamba' or world_model.model == 'Mamba2':
                        prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )[0]

            context_obs.append(rearrange(torch.Tensor(current_ob).to(world_model.device), "H W C -> 1 1 C H W")/255)
            context_action.append(action)
        else:
            action = env.action_space.sample()

        ob, reward, terminated, truncated, info = env.step(action)
        replay_buffer.append(current_ob, action, reward, np.logical_or(terminated, info["life_loss"]))

        sum_reward += reward

        done_flag = np.logical_or(terminated, truncated)
        if done_flag:
            logger.log(f"episode/score", sum_reward, global_step=total_steps)
            logger.log(f"episode/length", current_info["episode_frame_number"]//4, global_step=total_steps)  # framskip=4
            logger.log(f"episode/normalised score", (sum_reward - game_benchmark_df['Random'])/(game_benchmark_df['Human'] - game_benchmark_df['Random']), global_step=total_steps)
            logger.log("replay_buffer/length", len(replay_buffer), global_step=total_steps)
            for algorithm in game_benchmark_df.index[2:]:
                denominator = game_benchmark_df[algorithm] - game_benchmark_df['Random']
                if denominator != 0:
                    normalized_score = (sum_reward - game_benchmark_df['Random']) / denominator
                    logger.log(f"benchmark/normalised {algorithm} score", normalized_score, global_step=total_steps)
            
            sum_reward = 0
            ob, info = env.reset()
            context_obs.clear()
            context_action.clear()

        current_ob = ob
        current_info = info


        if replay_buffer.ready('world_model') and total_steps % (config.JointTrainAgent.TrainDynamicsEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeWorldModelAfterSteps:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=config.JointTrainAgent.BatchSize,
                batch_length=config.JointTrainAgent.BatchLength,
                logger=logger,
                epoch=config.JointTrainAgent.TrainDynamicsEpoch,
                global_step=total_steps
            )


        if replay_buffer.ready('behaviour') and total_steps % (config.JointTrainAgent.TrainAgentEverySteps // config.JointTrainAgent.NumEnvs) == 0 and total_steps <= config.JointTrainAgent.FreezeBehaviourAfterSteps:
            log_video = total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0

            imagine_latent, agent_action, old_logits, context_latent, context_reward, context_termination, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=config.JointTrainAgent.ImagineBatchSize,
                imagine_context_length=config.JointTrainAgent.ImagineContextLength,
                imagine_batch_length=config.JointTrainAgent.ImagineBatchLength,
                log_video=log_video,
                logger=logger,
                global_step=total_steps
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logits=old_logits,
                context_latent=context_latent,
                context_reward=context_reward,
                context_termination=context_termination,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
                global_step=total_steps
            )


        if config.JointTrainAgent.SaveModels and total_steps % (config.JointTrainAgent.SaveEverySteps // config.JointTrainAgent.NumEnvs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"{logdir}/ckpt/world_model.pth")
            torch.save(agent.state_dict(), f"{logdir}/ckpt/agent.pth")



def build_world_model(conf, action_dim, device):
    return WorldModel(
        action_dim = action_dim,
        conf = conf, 
        device = device
    ).cuda(device)


def build_agent(conf, action_dim, device):
    if conf.Models.Agent.Policy == 'AC':
        return agents.ActorCriticAgent(
            conf = conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)
    elif conf.Models.Agent.Policy == 'PPO':
        return agents.PPOAgent(
            conf=conf,
            action_dim=action_dim,
            device = device
        ).cuda(device)        


class DotDict(dict):
    """Dictionary with dot notation access."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{item}'")
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_or_create(self, key_path, value):
        keys = key_path.split('.')
        d = self
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = DotDict()
            d = d[key]
        d[keys[-1]] = value

# Function to update nested dictionary
def update_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

# Function to parse and update config from arguments
def parse_args_and_update_config(config, prefix=''):
    parser = argparse.ArgumentParser()

    def add_arguments(config, prefix=''):
        for key, value in config.items():
            if isinstance(value, dict):
                add_arguments(value, prefix + key + '.')
            else:
                parser.add_argument(f'--{prefix}{key}', type=type(value), default=value)

    add_arguments(config, prefix)
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    for arg_key, arg_value in args_dict.items():
        if arg_value is not None:
            keys = arg_key.split('.')
            update_dict(config, keys, arg_value)
    
    return config

def update_model_parameters(config, world_model, agent):
    config.update_or_create('Models.WorldModel.TotalParamNum', sum([p.numel() for p in world_model.parameters()]))
    print(f'World model total parameters: {sum([p.numel() for p in world_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.BackboneParamNum', sum([p.numel() for p in world_model.sequence_model.parameters()]))
    print(f'Dynamic model parameters: {sum([p.numel() for p in world_model.sequence_model.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.EncoderParamNum', sum([p.numel() for p in world_model.encoder.parameters()]))
    print(f'Encoder parameters: {sum([p.numel() for p in world_model.encoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DecoderParamNum', sum([p.numel() for p in world_model.image_decoder.parameters()]))
    print(f'Decoder parameters: {sum([p.numel() for p in world_model.image_decoder.parameters()]):,}')
    
    config.update_or_create('Models.WorldModel.DiscretisationLayerParamNum', sum([p.numel() for p in world_model.dist_head.parameters()]))
    print(f'Discretisation layer parameters: {sum([p.numel() for p in world_model.dist_head.parameters()]):,}')
    
    config.update_or_create('Models.Agent.ActorParamNum', sum([p.numel() for p in agent.actor.parameters()]))
    print(f'Actor parameters: {sum([p.numel() for p in agent.actor.parameters()]):,}')
    
    config.update_or_create('Models.Agent.CriticParamNum', sum([p.numel() for p in agent.critic.parameters()]))
    print(f'Critic parameters: {sum([p.numel() for p in agent.critic.parameters()]):,}')

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


    with open('config_files/train.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config = parse_args_and_update_config(config)   
    config = DotDict(config)

    device = torch.device(config.BasicSettings.Device)
    # set seed
    seed_np_torch(seed=config.BasicSettings.Seed)


    # getting action_dim with dummy env
    dummy_env = build_single_env(config.BasicSettings.Env_name, config.BasicSettings.ImageSize, seed=0)
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = build_world_model(config, action_dim, device=device)
    agent = build_agent(config, action_dim, device=device)
    update_model_parameters(config, world_model, agent)
    if (config.BasicSettings.Compile and os.name != "nt"):  # compilation is not supported on windows
        world_model = torch.compile(world_model)
        agent = torch.compile(agent)
    logger = WandbLogger(config=config, project=config.Wandb.Init.Project, mode=config.Wandb.Init.Mode)
    logdir = logger.run.dir

    # build replay buffer
    replay_buffer = ReplayBuffer(
        config,
        device=device
    )

    # train
    joint_train_world_model_agent(config, logdir, replay_buffer, world_model, agent, logger)

    logger.close()

