import os
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
import json

import numpy as np

BATCH_SIZE = 64
NUM_STACK_FRAME = 4
CAPACITY = 10000
GAMMA = 0.99

TD_ERROR_EPSILON = 0.0001

NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 50
value_loss_coef = 0.5
entropy_coef = 0.5
max_grad_norm = 0.5
config_clip = 0.2


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

class ReplayMemory():
    def __init__(self, CAPACITY, json_file_dir):
        self.capacity = CAPACITY
        self.global_npy_list = []
        self.index = 0
        self.json_file_dir = json_file_dir

    def push(self, state, action, state_next, reward, done, json_now_time):
        if len(self.global_npy_list) < self.capacity:
            self.global_npy_list.append(None)
        
        state = state.cpu().detach().numpy()
        action = action
        state_next = state_next.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        Transition_dict = Transition(state, action, state_next, reward, done)._asdict()
        np.save(os.path.join(self.json_file_dir , f"dict_{str(json_now_time)}.npy"), Transition_dict)

        self.global_npy_list[self.index] = os.path.join(self.json_file_dir , f"dict_{str(json_now_time)}.npy")
        self.index = (self.index + 1) % self.capacity # 保存しているindexを1つずらす　→　1001 % self.capacity = 1となる

    def sample(self, batch_size):
        batch_npy_list = random.sample(self.global_npy_list, batch_size)
        # print("json_list : ", batch_npy_list)
        memory = []
        for npy in batch_npy_list:
            npy_object = np.load(npy, allow_pickle=True).item()
            state = torch.from_numpy(npy_object["state"]).clone()
            action = npy_object["action"]
            state_next = torch.from_numpy(npy_object["next_state"]).clone()
            reward = torch.from_numpy(npy_object["reward"]).clone()
            done = npy_object["done"]
            memory.append(Transition(state, action, state_next, reward, done))
        return memory
    
    def __len__(self):
        return len(self.global_npy_list)

class TDerrorMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, td_error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity
    
    def __len__(self):
        return len(self.memory)
    
    def get_prioritized_indexes(self, batch_size):
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)
        return indexes

    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors

class Actor(nn.Module):
    def __init__(self, n_in, n_mid, n_out, action_space_high, action_space_low):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv2d_1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) # [1フレーム目, 2フレーム目, 3フレーム目, 4フレーム目]
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.batch_norm_1 = nn.InstanceNorm2d(32)
        self.batch_norm_2 = nn.InstanceNorm2d(64)
        self.batch_norm_3 = nn.InstanceNorm2d(64)

        self.fc = nn.Linear(64 * 9 * 9, 512)

        # Actor
        self.fc2 = nn.Linear(512, len(action_space_high)) # [壁x, 壁y, 車x, 車y, 人x, 人y, ゴール]など

        self.action_center = (action_space_high + action_space_low)/2
        self.action_scale = action_space_high - self.action_center
        self.action_range = action_space_high - action_space_low

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = self.batch_norm_1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.batch_norm_2(x)
        x = F.relu(self.conv2d_3(x))
        x = self.batch_norm_3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        actor_output = F.tanh(self.fc2(x)) # -1〜1にまとめる
        return actor_output

    def act(self, x, episode):
        std = 0.5
        action = self(x)
        action = action.detach()
        noise = torch.normal(action, std=std)
        env_action = torch.clip(action + noise, -1, 1)

        return env_action * self.action_scale + self.action_center , env_action

class Critic(nn.Module):
    def __init__(self, obs_shape1, obs_shape2, obs_shape3):
        super(Critic, self).__init__()
        self.conv2d_1 = nn.Conv2d(5, 32, kernel_size=8, stride=4) #[1フレーム目, 2フレーム目, 3フレーム目, 4フレーム目, 拡張した行動の値]
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.upsample = nn.Upsample(size=(100, 100), mode='bicubic')

        self.batch_norm_1 = nn.InstanceNorm2d(32)
        self.batch_norm_2 = nn.InstanceNorm2d(64)
        self.batch_norm_3 = nn.InstanceNorm2d(64)

        self.fc = nn.Linear(64 * 9 * 9, 512)

        # Critic
        self.critic = nn.Linear(512, 1) # 価値V側
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, actions):
        actions = actions.unsqueeze(1).unsqueeze(1) # [batch_size, 1, 100, 100]まで拡張
        actions = self.upsample(actions)
        
        x = torch.cat((x, actions), dim = 1)
        x_2 = x
        x = F.relu(self.conv2d_1(x))
        x = self.batch_norm_1(x)
        x = F.relu(self.conv2d_2(x))
        x = self.batch_norm_2(x)
        x = F.relu(self.conv2d_3(x))
        x = self.batch_norm_3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        critic_output1 = self.critic(x)

        x_2 = F.relu(self.conv2d_1(x_2))
        x_2 = self.batch_norm_1(x_2)
        x_2 = F.relu(self.conv2d_2(x_2))
        x_2 = self.batch_norm_2(x_2)
        x_2 = F.relu(self.conv2d_3(x_2))
        x_2 = self.batch_norm_3(x_2)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2 = F.relu(self.fc(x_2))
        critic_output2 = self.critic(x_2)
        
        return critic_output1, critic_output2

    def get_value(self, x):
        value = self(x)
        return value

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, 2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, 2, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 128, 2, stride=2)
        self.conv2d_4 = nn.Conv2d(128, 256, 2, stride=2)
        # self.conv2d_5 = nn.Conv2d(256, 1, 1, stride=1) # ボルトネック層

        self.batch_norm_1 = nn.InstanceNorm2d(32)
        self.batch_norm_2 = nn.InstanceNorm2d(64)
        self.batch_norm_3 = nn.InstanceNorm2d(128)
        self.batch_norm_4 = nn.InstanceNorm2d(256)

        self.fc = nn.Linear(6 * 6 * 256, 1)

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv2d_1(input)
        x = self.activation(x)
        x = self.batch_norm_1(x)

        x = self.conv2d_2(x)
        x = self.activation(x)
        x = self.batch_norm_2(x)

        x = self.conv2d_3(x)
        x = self.activation(x)
        x = self.batch_norm_3(x)

        x = self.conv2d_4(x)
        x = self.activation(x)
        x = self.batch_norm_4(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class Brain:
    def __init__(self, actor, critic, discriminator, json_file_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayMemory(CAPACITY, json_file_dir)
        self.td_memory = TDerrorMemory(CAPACITY)

        # actor(行動価値を求める)は２つ用意する
        self.main_actor = actor
        self.main_actor = self.init_weight(self.main_actor)
        self.target_actor = actor
        self.target_actor = self.init_weight(self.target_actor)

        # critic(状態価値を求める)は２つ用意する
        self.main_critic = critic
        self.main_critic = self.init_weight(self.main_critic)
        self.target_critic = critic
        self.target_critic = self.init_weight(self.target_critic)

        # Discriminator(GAN識別器)を用意
        self.discriminator = discriminator
        self.discriminator = self.init_weight(self.discriminator)

        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), lr=0.0001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)

        self.actor_update_interval = 2

    def init_weight(self, net):
        if isinstance(net, nn.Linear):
            nn.init.kaming_uniform_(net.weight.data)
            if net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        if isinstance(net, nn.Conv2d):
            nn.init.kaming_uniform_(net.weight.data)
            if net.bias is not None:
                nn.init.constant_(net.bias, 0.0)
        return net
    
    def make_minibatch(self, episode):
        # メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 各変数をメモリバッチに対する形に変形
        # transitions は　1stepごとの(state, action, state_next, reward)がBATCH_SIZE分格納されている
        # これを(state x BATCH_SIZE, action x BATCH_SIZER, state_next x BATCH_SIZE, state_next x BATCH_SIZE, reward x BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))
        # 各変数の要素をミニバッチに対応する形に変形する
        # 1x4がBATCH_SIZE分並んでいるところを　BATCH_SIZE x 4にする
        state_batch = torch.cat(batch.state).detach().to(self.device)
        action_batch = torch.cat(batch.action).detach().to(self.device)
        reward_batch = torch.cat(batch.reward).detach().to(self.device)
        next_state_batch = torch.cat(batch.next_state).detach().to(self.device)
        done_batch = batch.done

        return batch, state_batch, action_batch, reward_batch, next_state_batch, done_batch
