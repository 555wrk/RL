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

    def push(self, state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal

    def __len__(self):
        return len(self.buffer)


class ParallelReplayMemory:
    def __init__(self, main_capacity, temp_capacity, seed):
        # 修复：使用正确的初始化方式
        self.main_buffer = ReplayMemory(main_capacity, seed)
        self.temp_buffer = ReplayMemory(temp_capacity, seed)
        self.episode_trajectory = []  # 存储当前episode的轨迹

    def push(self, state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal):
        # 同时存入主缓冲区和临时缓冲区
        self.main_buffer.push(state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal)
        self.temp_buffer.push(state, action, reward, next_state, done, goal, achieved_goal, next_achieved_goal)

        # 将当前transition添加到episode_trajectory
        self.episode_trajectory.append((
            state, action, reward, next_state, done, goal,
            achieved_goal.copy(), next_achieved_goal.copy()
        ))

        if done:
            # 当前episode结束，应用HER
            self.apply_her_future()
            self.episode_trajectory = []  # 重置轨迹

    def apply_her_future(self, k=4):
        episode_length = len(self.episode_trajectory)
        for idx in range(episode_length):
            # 为当前步骤生成k个不同的未来目标（核心修改）
            for _ in range(k):
                state, action, reward, next_state, done, orig_goal, achieved_goal, next_achieved_goal = \
                self.episode_trajectory[idx]
                # 新增：从当前索引之后的轨迹中随机选择未来目标（符合文档future策略）
                future_idx = np.random.randint(idx, episode_length)
                future_goal = self.episode_trajectory[future_idx][6]  # 取未来状态的achieved_goal作为新目标
                # 获取物块尺寸（与环境中一致）
                block_dims = [0.012, 0.35, 0.012]
                # 计算新目标对应的左右抓取点（基于新目标future_goal）
                future_left_grasp = future_goal + np.array([0, block_dims[1] / 1.5, block_dims[2] / 2])
                future_right_grasp = future_goal + np.array([0, -block_dims[1] / 1.5, block_dims[2] / 2])

                # 获取next_state中双臂末端位置（需要从next_state中解析，假设next_state包含双臂位置）
                # next_state是23维，前3位是左臂末端位置，4-6位是右臂末端位置（参考get_physic_state）
                left_ee_pos_next = next_state[0:3]
                right_ee_pos_next = next_state[3:6]

                # 计算HER场景下的三个距离
                d_left_her = np.linalg.norm(left_ee_pos_next - future_left_grasp)
                d_right_her = np.linalg.norm(right_ee_pos_next - future_right_grasp)
                d_block_her = np.linalg.norm(next_achieved_goal - future_goal)

                # 遵循原奖励函数的条件，T1和T2与环境中一致
                T1 = 0.02
                T2 = 0.02
                new_reward = 1.0 if (d_left_her < T1 and d_right_her < T1 and d_block_her < T2) else 0.0

                # 存入主缓冲区
                self.main_buffer.push(
                    state, action, new_reward, next_state, done,
                    future_goal, achieved_goal, next_achieved_goal
                )

    def sample(self, main_batch_size, temp_batch_size):
        # 从两个缓冲区分别采样
        main_batch = self.main_buffer.sample(main_batch_size)
        temp_batch = self.temp_buffer.sample(temp_batch_size)

        # 合并批次
        state_batch = np.concatenate([main_batch[0], temp_batch[0]], axis=0)
        action_batch = np.concatenate([main_batch[1], temp_batch[1]], axis=0)
        reward_batch = np.concatenate([main_batch[2], temp_batch[2]], axis=0)
        next_state_batch = np.concatenate([main_batch[3], temp_batch[3]], axis=0)
        mask_batch = np.concatenate([main_batch[4], temp_batch[4]], axis=0)
        goal_batch = np.concatenate([main_batch[5], temp_batch[5]], axis=0)

        return state_batch, action_batch, reward_batch, next_state_batch, mask_batch, goal_batch

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