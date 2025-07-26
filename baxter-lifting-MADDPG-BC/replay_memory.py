import random
import numpy as np
import os
import pickle


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = int(capacity)
        self.buffer = []
        self.position = int(0)
        self.is_expert = []  # 新增：标记是否为专家数据
    def push(self, state, action, reward, next_state, done,is_expert=False):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.is_expert.append(None)  # 新增
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.is_expert[self.position] = is_expert  # 新增
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, expert_ratio=0.7):
        # 计算专家数据和智能体数据的数量
        expert_size = int(batch_size * expert_ratio)
        agent_size = batch_size - expert_size

        # 分离专家数据和智能体数据
        expert_indices = [i for i, exp in enumerate(self.is_expert) if exp]
        agent_indices = [i for i, exp in enumerate(self.is_expert) if not exp]

        # 采样
        expert_batch = random.sample(expert_indices, min(expert_size, len(expert_indices)))
        agent_batch = random.sample(agent_indices, min(agent_size, len(agent_indices)))

        # 合并批次
        indices = expert_batch + agent_batch
        batch = [self.buffer[i] for i in indices]
        is_expert_flags = [self.is_expert[i] for i in indices]  # 新增：专家标记

        # 解压批次
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done, is_expert_flags

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
