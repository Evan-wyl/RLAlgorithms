# -*- codeing = utf-8 -*-
# @Time : 2021/12/23 8:24
# @Author : Evan_wyl
# @File : DDPG.py

import gym
import random
import copy
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import init
from torch.nn import functional as F

param = dict()
param["BUFFER_SIZE"] = 10000
param['BATCH_SIZE'] = 64
param["TOTAL_TIMES"] = 1000000
param["UPDATE_FREQ"] = 1000
param["FREQ"] = 0
param["EPS"] = 0.1
param['GAMMA'] = 0.99
param["TAU"] = 0.001
param["NOISE_END"] = 0.05
param["NOISE_START"] = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(input):
    if isinstance(input, nn.Linear):
        init.xavier_uniform_(input.weight, gain=1)
        init.xavier_uniform_(input.bias, 0)
    if isinstance(input, nn.Conv2d):
        init.orthogonal_(input.weight, gain="relu")
        init.orthogonal_(input.bias, 0)


class ReplayBuffer(object):
    def __init__(self):
        super(ReplayBuffer, self).__init__()

        self.buffer_size = param["BUFFER_SIZE"]
        self.buffer = []

    def save(self, s, a, s_, r, done):
        self.buffer.append((s, a, s_, r, done))
        if len(self.buffer) > self.buffer_size:
            del self.buffer[0]

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            sample_arr = self.buffer
        else:
            sample_arr = random.sample(self.buffer, batch_size)
        return sample_arr


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_shape[0], 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, action_shape[0])

        self.max_action = torch.FloatTensor(max_action).to(device)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_shape[0] + action_shape[0], 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = F.relu(self.l1(torch.cat([obs, action], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(nn.Module):
    def __init__(self, obs_shape, action_shape, max_action):
        super(DDPG, self).__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = param["GAMMA"]
        self.tau = param["TAU"]

        self.critic = Critic(obs_shape, action_shape).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.actor = Actor(obs_shape, action_shape, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

    def train_model(self, batch_size, replay_buffer):
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.prepare_mini_batch(batch_size, replay_buffer)

        with torch.no_grad():
            target = self.critic_target(next_obs_batch, self.actor_target(next_obs_batch))
            target = reward_batch + (self.gamma * target * done_batch).detach()
        loss_critic = F.mse_loss(target, self.critic(obs_batch, action_batch))

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        loss_actor = - self.critic(obs_batch, self.actor(obs_batch))
        loss_actor = torch.sum(loss_actor)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) +  self.tau * param)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(target_param * (1 - self.tau) + self.tau * param)

    def get_action(self, obs):
        obs = torch.tensor(obs, device=device, dtype=torch.float).view((-1, ) + self.obs_shape)
        action = self.actor(obs)
        return action.cpu().data.numpy().flatten()

    def prepare_mini_batch(self, batch_size, replay_buffer):
        sample_arr = replay_buffer.sample(batch_size)

        obs_arr, action_arr, next_obs_arr, reward_arr, done_arr = zip(*sample_arr)

        obs_batch = torch.tensor(obs_arr, device=device, dtype=torch.float).view((-1, ) + self.obs_shape)
        action_batch = torch.tensor(action_arr, device=device, dtype=torch.float).squeeze().view((-1, 1))
        next_obs_batch = torch.tensor(next_obs_arr, device=device, dtype=torch.float).view((-1, ) + self.obs_shape)
        reward_batch = torch.tensor(reward_arr, device=device, dtype=torch.float).squeeze().view(-1, 1)
        done_batch = torch.tensor(done_arr, device=device, dtype=torch.int8).squeeze().view(-1,1)

        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch

    def save_model(self):
        torch.save(self.critic.state_dict(), "./pretrain_model/DDPG/critic_state_dict.dump")
        torch.save(self.critic_optimizer.state_dict(), "./pretrain_model/DDPG/critic_state_dict_optimizer.dump")

        torch.save(self.actor.state_dict(), "./pretrain_model/DDPG/actor_state_dict.dump")
        torch.save(self.actor_optimizer.state_dict(), "./pretrain_model/DDPG/actor_state_dict_optimizer.dump")

    def load_model(self):
        self.critic.load_state_dict(torch.load("./pretrain_model/DDPG/critic_state_dict.dump"))
        self.critic_optimizer.load_state_dict(torch.load("./pretrain_model/DDPG/critic_state_dict_optimizer.dump"))

        self.actor.load_state_dict(torch.load("./pretrain_model/DDPG/actor_state_dict.dump"))
        self.actor_optimizer.load_state_dict(torch.load("./pretrain_model/DDPG/actor_state_dict_optimizer.dump"))


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")

    print(env.observation_space.shape)
    print("obs high:", env.observation_space.high)
    print("obs low:", env.observation_space.low)
    print("action high:", env.action_space.high)
    print("action low:", env.action_space.low)

    obs = env.reset()
    print(obs)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    max_action = env.action_space.high

    replay_buffer = ReplayBuffer()
    ddpg = DDPG(obs_shape, action_shape, max_action)

    rewards_total = 0
    epoch = 0
    for i in range(1, param["TOTAL_TIMES"] + 1):
        param["NOISE"] = param["NOISE_END"] + (param["NOISE_START"] - param["NOISE_END"]) * np.exp((-1 * i) / 30000)
        param["FREQ"] +=1
        env.render()
        action = ddpg.get_action(obs)
        action = np.clip(np.random.normal(action, param["NOISE"]), -1.0, 1.0)
        next_obs, rewards, done, _ = env.step(action)

        replay_buffer.save(obs, action, next_obs, rewards, done)
        obs = next_obs
        if done:
            obs = env.reset()

        if rewards > 0:
            rewards_total += 1
            epoch += 1
        print("freqs:{}, action:{}, rewards:{}".format(param["FREQ"], action, rewards))
        print("freqs:{}, epoch:{}, success:{}".format(param["FREQ"], epoch, rewards_total))

        ddpg.train_model(param["BUFFER_SIZE"], replay_buffer)

    score = 0
    for _ in range(10):
        while True:
            obs = env.reset()
            action = ddpg.get_action(obs)
            next_obs, rewards, done, _ = env.step(action)
            score += rewards
            obs = next_obs
            if done:
                break
    print("DDPG Score:{%.3f}" % (score / 10))
