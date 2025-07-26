import random
import numpy as np
import os
import pickle


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ParallelReplayMemory:
    def __init__(self, main_capacity, temp_capacity, seed):
        # 修复：使用正确的初始化方式
        self.main_buffer = ReplayMemory(main_capacity, seed)
        self.temp_buffer = ReplayMemory(temp_capacity, seed)

    def push(self, state, action, reward, next_state, done):
        # 同时存入主缓冲区和临时缓冲区
        self.main_buffer.push(state, action, reward, next_state, done)
        self.temp_buffer.push(state, action, reward, next_state, done)

    def sample(self, main_batch_size, temp_batch_size):
        # 从两个缓冲区分别采样
        main_batch = self.main_buffer.sample(main_batch_size)
        temp_batch = self.temp_buffer.sample(temp_batch_size)

        # 合并批次 (修复括号错误)
        state_batch = np.concatenate([main_batch[0], temp_batch[0]], axis=0)
        action_batch = np.concatenate([main_batch[1], temp_batch[1]], axis=0)
        reward_batch = np.concatenate([main_batch[2], temp_batch[2]], axis=0)
        next_state_batch = np.concatenate([main_batch[3], temp_batch[3]], axis=0)
        mask_batch = np.concatenate([main_batch[4], temp_batch[4]], axis=0)

        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch

    def __len__(self):
        return len(self.main_buffer) + len(self.temp_buffer)

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