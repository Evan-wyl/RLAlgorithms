# -*- codeing = utf-8 -*-
# @Time : 2021/12/27 13:40
# @Author : Evan_wyl
# @File : TD3.py

import random
import numpy as np
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.nn import init

import gym

param = dict()
param["buffer_size"] = 10000
param["total_times"] = 100000000
param["vars"] = 1
param["vars_1"] = 1.5
param["c"] = 0.5
param["tau"] = 0.5
param["freq"] = 0
param["batch_size"] = 256
param["update_freq"] = 1000
param['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_init(input):
    if isinstance(input, nn.Linear):
        init.xavier_uniform_(input.weight, gain=1)
        # init.xavier_uniform_(input.bias, 1)
    if isinstance(input, nn.Conv2d):
        init.orthogonal_(input.weight, gain="relu")
        init.orthogonal_(input.bias, 0)


class ReplayBuffer(object):
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.buffer = []
        self.buffer_size = param["buffer_size"]

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            return self.buffer
        else:
            return random.sample(self.buffer, batch_size)

    def save(self, s, a, s_, r, done):
        self.buffer.append((s, a, s_, r, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_shape[0], 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, action_shape[0])

        self.max_action = torch.tensor(max_action, dtype=torch.float, device=param["device"])

        # self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return torch.tanh(x) * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_shape[0] + action_shape[0], 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

        # self.apply(weight_init)

    def forward(self, obs, action):
        # torch.cat:对应维度相加
        x = F.relu(self.l1(torch.cat([obs, action], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class TD3(nn.Module):
    def __init__(self, obs_shape, action_shape, max_action):
        super(TD3, self).__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.max_action = max_action

        self.critic1 = Critic(obs_shape=obs_shape, action_shape=action_shape).to(param["device"])
        self.critic2 = Critic(obs_shape=obs_shape, action_shape=action_shape).to(param['device'])
        self.critic1_optimizer = optim.Adam(params=self.critic1.parameters(), lr=0.001)
        self.critic2_optimizer = optim.Adam(params=self.critic2.parameters(), lr=0.001)

        self.actor = Actor(obs_shape=obs_shape, action_shape=action_shape, max_action=max_action).to(param['device'])
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=0.001)

        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_target = copy.deepcopy(self.actor)

    def get_action(self, obs, vars):
        obs = torch.tensor(obs, dtype=torch.float, device=param["device"]).view((-1, ) + self.obs_shape)
        # action = self.actor(obs)
        return self.actor(obs).cpu().data.numpy().flatten() + random.normalvariate(0, vars)

    def prepare_mini_batch(self, batch_size, buffer):
        sample_batch = buffer.sample(batch_size=batch_size)
        obs_arr, action_arr, next_obs_arr, reward_arr, done_arr = zip(*sample_batch)

        obs_batch = torch.tensor(obs_arr, dtype=torch.float, device=param["device"]).view((-1,) + self.obs_shape)
        action_batch = torch.tensor(action_arr, dtype=torch.float, device=param["device"]).squeeze().view((-1, 1))
        next_obs_batch = torch.tensor(next_obs, dtype=torch.float, device=param['device']).view((-1,) + self.obs_shape)
        reward_batch = torch.tensor(reward_arr, dtype=torch.float, device=param["device"]).squeeze().view((-1, 1))
        done_batch = torch.tensor(done_arr, dtype=torch.float, device=param["device"]).squeeze().view((-1, 1))

        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch

    def train_model(self, batch_size, buffer):
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = self.prepare_mini_batch(batch_size=batch_size, buffer=buffer)

        with torch.no_grad():
            actions = self.actor(obs_batch) + torch.clip(torch.tensor(random.normalvariate(0, param["vars_1"])),
                                                         -param["c"], param["c"])
            min_q = torch.min(self.critic1(obs_batch, actions), self.critic2(obs_batch, actions)).view((-1,1))
            target_c = reward_batch + min_q
        loss_c1 = F.mse_loss(target_c, self.critic1(obs_batch, actions))
        self.critic1_optimizer.zero_grad()
        loss_c1.backward()
        self.critic1_optimizer.step()

        loss_c2 = F.mse_loss(target_c, self.critic2(obs_batch, actions))
        self.critic2_optimizer.zero_grad()
        loss_c2.backward()
        self.critic2_optimizer.step()

        if param["freq"] % param["update_freq"] == 0:
            loss_actor = self.critic1(obs_batch, self.actor(obs_batch))
            loss_actor = torch.sum(loss_actor)
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            for params, params_target in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                params_target.data.copy_(params_target * (1 - param["tau"]) + params * param["tau"])

            for params, params_target in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                params_target.data.copy_(params_target * (1 - param["tau"]) + params * param["tau"])

            for params, params_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                params_target.data.copy_(params_target * (1 - param["tau"]) + params * param["tau"])

    def save_model(self):
        torch.save(self.critic1.state_dict(), "./pretrain_model/TD3/critic1_state_dict.dump")
        torch.save(self.critic2.state_dict(), "./pretrain_model/TD3/critic2_state_dict.dump")
        torch.save(self.critic1_optimizer.state_dict(), "./pretrain_model/TD3/critic1_optimizer_state_dict.dump")
        torch.save(self.critic2_optimizer.state_dict(), "./pretrain_model/TD3/critic2_optimizer_state_dict.dump")

        torch.save(self.actor.state_dict(), "./pretrain_model/TD3/actor_state_dict.dump")
        torch.save(self.actor_optimizer.state_dict(), "./pretrain_model/TD3/actor_optimizer_state_dict.dump")

    def load_model(self):
        self.critic1.load_state_dict(torch.load("./pretrain_model/TD3/critic1_state_dict.dump"))
        self.critic2.load_state_dict(torch.load("./pretrain_model/TD3/critic2_state_dict.dump"))
        self.critic1_optimizer.load_state_dict(torch.load("./pretrain_model/TD3/critic1_optimizer_state_dict.dump"))
        self.critic2_optimizer.load_state_dict(torch.load("./pretrain_model/TD3/critic2_optimizer_state_dict.dump"))

        self.actor.load_state_dict(torch.load("./pretrain_model/TD3/actor_state_dict.dump"))
        self.actor_optimizer.load_state_dict(torch.load("./pretrain_model/TD3/actor_optimizer_state_dict.dump"))


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    # 向量: env.action_space env.observation_space
    print("action_space.shape: ", env.action_space.shape)
    print("observation_space.shape: ", env.observation_space.shape)
    print("action_space.high: ", env.action_space.high)
    print("action_space.low: ", env.action_space.low)

    max_action = env.action_space.high
    action_shape = env.action_space.shape
    obs_shape = env.observation_space.shape

    td3 = TD3(obs_shape=obs_shape, action_shape=action_shape, max_action=max_action)
    replay_buffer = ReplayBuffer()

    total_rewards = 0
    epoch = 0
    obs = env.reset()
    for i in range(1, param["total_times"] + 1):
        param["freq"] += 1
        action = td3.get_action(obs=obs, vars=param["vars"])
        next_obs, rew, done, _ = env.step(action)
        env.render()

        replay_buffer.save(obs, action, next_obs, rew, done)

        obs = next_obs
        if done:
            obs = env.reset()
            epoch += 1

        if rew > 0:
            total_rewards += rew
        print("freqs:{},action:{},rewards:{}".format(param["freq"], action, rew))
        print("freqs:{},epoch:{},total_rewards:{}".format(param["freq"],epoch, total_rewards))

        batch_size = param["batch_size"]
        td3.train_model(batch_size, replay_buffer)

        if i % 10000 == 0:
            td3.save_model()

    rew = 0
    obs = env.reset()
    while True:
        action = td3.get_action(obs, param["vars"])
        next_obs, rewards, done, _ = env.step(action)
        rew += rewards
        obs = next_obs
        if done:
            break
    print("Total_Rewards:{}".format(rew))
