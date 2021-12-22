# -*- codeing = utf-8 -*-
# @Time : 2021/12/16 21:16
# @Author : Evan_wyl
# @File : DQN.py

import gym
import random
import copy
import numpy as np
import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn import init

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

param = dict()
param["TARGET_NET_UPDATE_FREQ"] = 1000
param["SAVE_MODEL_FREQ"] = 10000


def weight_init_(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.orthogonal_(m.weight, gain='relu')
        init.constant_(m.bias, 0)


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        super(ReplayBuffer, self).__init__()

        self.buffer_size = buffer_size
        self.buffer = []

    def save(self, obs, action, reward, next_state, done):
        self.buffer.append((obs, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            del self.buffer[0]

    def sample(self, batch_size):
        obs_arr, action_arr, reward_arr, next_state_arr, done_arr = zip(
            *random.sample(self.buffer, batch_size))
        return obs_arr, action_arr, reward_arr, next_state_arr, done_arr


class Citic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Citic, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.loss = []

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        self.linear1 = nn.Linear(self.feature_size(), 512)
        self.linear2 = nn.Linear(512, self.num_actions)
        self.train(weight_init_)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)


class DQN(object):
    def __init__(self, input_shape, num_actions, gamma):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.freq = 0

        self.critic = Citic(self.input_shape, self.num_actions).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

    def training(self, batch_size, buffers):
        obs_batch, action_batch, reward_batch, next_state_batch, done_batch = self.pre_minibatch(batch_size, buffers)

        current_q = self.critic(obs_batch).gather(1, action_batch)

        with torch.no_grad():
            batch_size = min(len(buffers.buffer), batch_size)
            target_q = torch.zeros(batch_size, device=device, dtype=torch.float).view(batch_size,1)
            # print(target_q.shape)
            # print(batch_size)
            for i in range(batch_size):
                if done_batch[i]:
                    target_q[i] = 0
                else:
                    target = self.critic_target(next_state_batch[i].view((-1,) + self.input_shape)).max()
                    target_q[i] = torch.mul(self.gamma, target)
            # print(reward_batch)
            target_q += reward_batch

        diff = target_q - current_q
        loss = self.huber(diff).mean()
        print("freqs:{}, loss:{}".format(freqs, loss))
        self.critic_optimizer.zero_grad()
        loss.backward()
        for params in self.critic.parameters():
            params.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        self.freq += 1

        if self.freq % param["TARGET_NET_UPDATE_FREQ"] == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())

        if self.freq % param["SAVE_MODEL_FREQ"] == 0:
            torch.save(self.critic.state_dict(), "./pretrain_model/DQN/dqn_{}.dump".format(self.freq))

    def testing(self, env):
        obs = env.reset()
        results = 0
        while True:
            env.render()
            action = self.get_action(obs, eps)
            next_obs, reward, done, _ = env.step(action)
            results += reward
            obs = next_obs
            if done:
                break
        return results

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def pre_minibatch(self, batch_size, buffers):
        if len(buffers.buffer) < batch_size:
            batch_size = len(buffers.buffer)
        obs_arr, action_arr, reward_arr, next_state_arr, done_arr = buffers.sample(batch_size)

        obs_arr = torch.tensor(obs_arr, device=device, dtype=torch.float).view((-1,) + self.input_shape)
        action_arr = torch.tensor(action_arr, device=device, dtype=torch.long).squeeze().view(-1, 1)
        reward_arr = torch.tensor(reward_arr, device=device, dtype=torch.float).squeeze().view(-1, 1)
        next_state_arr = torch.tensor(next_state_arr, device=device, dtype=torch.float).view((-1,) + self.input_shape)
        done_arr = torch.tensor(done_arr, device=device, dtype=torch.bool).squeeze().view(-1, 1)

        return obs_arr, action_arr, reward_arr, next_state_arr, done_arr

    def get_action(self, obs, eps):
        obs = torch.transpose(torch.FloatTensor(obs).transpose(0, 2), 1, 2)
        with torch.no_grad():
            if np.random.random() >= eps:
                obs = obs.unsqueeze(0).to(device)
                action = self.critic(obs)
                action = torch.argmax(action)
                action = action.item()
            else:
                action = np.random.randint(0, self.num_actions)
            return action

    def save_model(self):
        torch.save(self.critic.state_dict(), "./pretrain_model/DQN/critic.dump")

    def load_model(self):
        self.critic.load_state_dict(torch.load("./pretrain_model/DQN/critic.dump"))
        self.critic_target.load_state_dict(self.critic.state_dict())


if __name__ == '__main__':
    env = gym.make("PongNoFrameskip-v4")
    obs = env.reset()

    random.seed(2021)
    torch.random.manual_seed(2021)

    input_shape = torch.transpose(torch.FloatTensor(obs).transpose(0, 2), 1, 2).shape
    num_actions = env.action_space.n
    batch_size = 32

    dqn = DQN(input_shape, num_actions, 0.99)

    buffers = ReplayBuffer(buffer_size=100000)

    epsilon_by_frame = lambda frame_idx: 0.01 + (1.0 - 0.01) * math.exp(-1. * frame_idx / 30000)

    freqs = 0
    episode = 0
    rewards_1 = 0
    for _ in range(1, 1000001):

        eps = epsilon_by_frame(_)

        freqs += 1
        env.render()
        action = dqn.get_action(obs, eps)
        next_obs, reward, done, _ = env.step(action)
        if reward == 1:
            rewards_1 += reward
        print("freqs:{}, episode:{}, action:{}".format(freqs, episode, action))
        print("freqs:{}, episode:{}, reward:{}".format(freqs, episode, reward))
        print("freqs:{}, episode:{}, rewards_1:{}".format(freqs, episode, rewards_1))
        buffers.save(obs, action, reward, next_obs, done)
        obs = next_obs

        dqn.training(batch_size, buffers)

        if done:
            episode += 1
            obs = env.reset()

    dqn.save_model()
    score = dqn.testing(env)
    print("score: ", score)
    env.close()
