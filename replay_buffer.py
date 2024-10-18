import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, config, device="cuda") -> None:
        self.store_on_gpu = config.BasicSettings.ReplayBufferOnGPU
        max_length = config.JointTrainAgent.BufferMaxLength
        obs_shape = (config.BasicSettings.ImageSize, config.BasicSettings.ImageSize, config.BasicSettings.ImageChannel)
        self.device = device

        if self.store_on_gpu:
            self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.sampled_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
            self.imagined_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length), dtype=np.float32)
            self.reward_buffer = np.empty((max_length), dtype=np.float32)
            self.termination_buffer = np.empty((max_length), dtype=np.float32)
            self.sampled_counter = np.zeros((max_length), dtype=np.int32)
            self.imagined_counter = np.zeros((max_length), dtype=np.int32)

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.world_model_warmup_length = config.JointTrainAgent.WorldModelWarmUp
        self.behaviour_warmup_length = config.JointTrainAgent.BehaviourWarmUp
        self.tau = config.JointTrainAgent.Tau
        self.imagination_tau = config.JointTrainAgent.ImaginationTau
        self.alpha = config.JointTrainAgent.Alpha
        self.beta = config.JointTrainAgent.Beta
        self.batch_scale_factor = config.JointTrainAgent.ImagineBatchSize / config.JointTrainAgent.BatchSize

    def ready(self, model_name='world_model'):
        return self.length  > self.world_model_warmup_length if model_name == 'world_model' else self.length  > self.behaviour_warmup_length

    @torch.no_grad()
    def sample(self, batch_size, batch_length, imagine=False):
        if self.store_on_gpu:
            obs_list, action_list, reward_list, termination_list = [], [], [], []
            counts = self.sampled_counter[:self.length + 1 - batch_length]
            imagine_counts = self.imagined_counter[:self.length + 1 - batch_length] / self.batch_scale_factor
            
            if imagine:
                linear_penalty = torch.maximum(torch.zeros_like(counts), counts - imagine_counts)
                score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                score = score / self.imagination_tau
                probabilities = torch.softmax(score, dim=0)
                start_indexes = torch.multinomial(probabilities, batch_size, replacement=False)
            else:
                logits = -counts / self.tau
                probabilities = torch.exp(logits) / torch.sum(torch.exp(logits))
                start_indexes = torch.multinomial(probabilities, batch_size, replacement=False)

            if not imagine:
                self.sampled_counter[start_indexes] += 1
            else:
                self.imagined_counter[start_indexes] += 1

            indexes = start_indexes.unsqueeze(-1).to(self.device) + torch.arange(batch_length, device=self.device)
            
            obs_list.append(self.obs_buffer[indexes])
            action_list.append(self.action_buffer[indexes])
            reward_list.append(self.reward_buffer[indexes])
            termination_list.append(self.termination_buffer[indexes])

            obs = torch.cat(obs_list, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action_list, dim=0)
            reward = torch.cat(reward_list, dim=0)
            termination = torch.cat(termination_list, dim=0)
        else:
            obs_list, action_list, reward_list, termination_list = [], [], [], []

            if batch_size > 0:

                counts = self.sampled_counter[:self.length + 1 - batch_length]
                imagine_counts = self.imagined_counter[:self.length + 1 - batch_length] / self.batch_scale_factor

                if imagine:
                    linear_penalty = np.maximum(np.zeros_like(counts), counts - imagine_counts)
                    score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                    score /= self.imagination_tau
                else:
                    score = -counts / self.tau

                exp_score = np.exp(score - np.max(score))
                probabilities = exp_score / np.sum(exp_score)

                start_indexes = np.random.choice(len(probabilities), size=(batch_size,), replace=False, p=probabilities)

                if not imagine:
                    self.sampled_counter[start_indexes] += 1
                else:
                    self.imagined_counter[start_indexes] += 1 

                indexes = start_indexes[:, np.newaxis] + np.arange(batch_length)

                obs_seq = self.obs_buffer[indexes]
                action_seq = self.action_buffer[indexes]
                reward_seq = self.reward_buffer[indexes]
                termination_seq = self.termination_buffer[indexes]

                obs_seq = torch.from_numpy(obs_seq).float().to(self.device) / 255
                obs_seq = rearrange(obs_seq, "B T H W C -> B T C H W")
                action_seq = torch.from_numpy(action_seq).to(self.device)
                reward_seq = torch.from_numpy(reward_seq).to(self.device)
                termination_seq = torch.from_numpy(termination_seq).to(self.device)

                obs_list.append(obs_seq)
                action_list.append(action_seq)
                reward_list.append(reward_seq)
                termination_list.append(termination_seq)

            obs = torch.cat(obs_list, dim=0) if obs_list else torch.empty(0, device=self.device)
            action = torch.cat(action_list, dim=0) if action_list else torch.empty(0, device=self.device)
            reward = torch.cat(reward_list, dim=0) if reward_list else torch.empty(0, device=self.device)
            termination = torch.cat(termination_list, dim=0) if termination_list else torch.empty(0, device=self.device)

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        self.last_pointer = (self.last_pointer + 1) % (self.max_length)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.tensor(action, device=self.device)
            self.reward_buffer[self.last_pointer] = torch.tensor(reward, device=self.device)
            self.termination_buffer[self.last_pointer] = torch.tensor(termination, device=self.device)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length
