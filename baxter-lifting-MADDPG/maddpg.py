import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import DeterministicPolicy
import os
import time


class MADDPG:
    def __init__(self, num_inputs, action_dims, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.action_dims = action_dims
        self.total_action_dim = sum(action_dims)

        # 创建演员网络
        self.actors = [
            DeterministicPolicy(num_inputs, action_dims[0], args.hidden_size, None).to(self.device),
            DeterministicPolicy(num_inputs, action_dims[1], args.hidden_size, None).to(self.device)
        ]

        # 创建评论家网络
        self.critic = CentralizedCritic(num_inputs, self.total_action_dim, args.hidden_size).to(self.device)
        self.critic_target = CentralizedCritic(num_inputs, self.total_action_dim, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # 优化器
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=args.lr) for actor in self.actors
        ]
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)

        # 创建保存文件夹
        self.save_folder = self._create_save_folder(args.seed)

    def _create_save_folder(self, seed):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        folder_path = os.path.join('checkpoints', f"{timestamp}_seed_{seed}")
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def select_action(self, state, agent_idx, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            return self.actors[agent_idx](state).detach().cpu().numpy()[0]
        else:
            return self.actors[agent_idx].sample(state)[0].detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # 从经验池采样
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        # 转换为张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # 拆分左右臂动作
        left_action_batch = action_batch[:, :self.action_dims[0]]
        right_action_batch = action_batch[:, self.action_dims[0]:]

        with torch.no_grad():
            # 目标演员动作
            next_left_action = self.actors[0](next_state_batch)
            next_right_action = self.actors[1](next_state_batch)
            next_actions = torch.cat([next_left_action, next_right_action], dim=1)

            # 目标评论家Q值 - 关键修复：取两个Q值的最小值
            target_q1, target_q2 = self.critic_target(next_state_batch, next_actions)
            target_q = torch.min(target_q1, target_q2)
            next_q_value = reward_batch + mask_batch * self.gamma * target_q

        # 更新评论家
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, next_q_value.detach()) + F.mse_loss(current_q2, next_q_value.detach())

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 更新演员
        actor_losses = []
        for i, actor in enumerate(self.actors):
            # 当前演员动作
            current_action = actor(state_batch)

            # 组合动作
            if i == 0:  # 左臂
                actions = torch.cat([current_action, right_action_batch.detach()], dim=1)
            else:  # 右臂
                actions = torch.cat([left_action_batch.detach(), current_action], dim=1)

            # 演员损失 - 使用第一个Q值计算
            actor_q1, _ = self.critic(state_batch, actions)
            actor_loss = -actor_q1.mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            actor_losses.append(actor_loss.item())

        # 软更新目标网络
        soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_losses[0], actor_losses[1]

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_folder, f"maddpg_{env_name}_{suffix}.pt")

        torch.save({
            'actor_left_state_dict': self.actors[0].state_dict(),
            'actor_right_state_dict': self.actors[1].state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'actor_left_optimizer_state_dict': self.actor_optimizers[0].state_dict(),
            'actor_right_optimizer_state_dict': self.actor_optimizers[1].state_dict()
        }, ckpt_path)
        print(f'保存模型到 {ckpt_path}')

    def load_checkpoint(self, ckpt_path, evaluate=False):
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            self.actors[0].load_state_dict(checkpoint['actor_left_state_dict'])
            self.actors[1].load_state_dict(checkpoint['actor_right_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optimizers[0].load_state_dict(checkpoint['actor_left_optimizer_state_dict'])
            self.actor_optimizers[1].load_state_dict(checkpoint['actor_right_optimizer_state_dict'])

            if evaluate:
                for actor in self.actors:
                    actor.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                for actor in self.actors:
                    actor.train()
                self.critic.train()
                self.critic_target.train()
            print(f'从 {ckpt_path} 加载模型')
        else:
            print(f'警告: 未找到模型文件 {ckpt_path}')


class CentralizedCritic(nn.Module):
    def __init__(self, num_inputs, total_action_dim, hidden_size):
        super(CentralizedCritic, self).__init__()
        self.input_norm = nn.LayerNorm(num_inputs + total_action_dim)
        # Q1 网络
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + total_action_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加归一化
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加归一化
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Q2 网络
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + total_action_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加归一化
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加归一化
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        sa = self.input_norm(sa)  # 输入归一化
        return self.q1(sa), self.q2(sa)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)