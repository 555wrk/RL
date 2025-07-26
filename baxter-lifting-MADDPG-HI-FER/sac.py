import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from util import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import math
import time


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        #self.device = torch.device("cuda" if args.cuda else "cpu")
        self.device = torch.device('cpu')
        # Critic网络
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        # target_Critic网络：初始化时通过 hard_update 直接复制主网络参数，后续通过软更新（EMA）缓慢同步。
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        # Actor策略网络
        if self.policy_type == "Gaussian":  # GaussianPolicy:高斯策略
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:  # 自动熵调节参数
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.tensor(math.log(self.alpha), requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:  # DeterministicPolicy:确定性策略
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # 创建独立的保存文件夹
        #self.save_folder = self._create_save_folder()
        self.save_folder = self._create_save_folder(args.seed)
    def _create_save_folder(self,seed):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        #folder_path = os.path.join('checkpoints', timestamp)
        folder_path = os.path.join('checkpoints', f"{timestamp}_seed_{seed}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    # 动作选择函数
    def select_action(self, state, evaluate=False):
        #print(self.device)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    # 参数更新函数：从经验回放缓冲区采样数据，更新策略网络和价值网络
    def update_parameters(self, memory, batch_size, updates, bc_weight=0.5):
        # 采样批次数据，包含专家标记
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, is_expert = memory.sample(batch_size, expert_ratio=0.5)
        # 转换为张量
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        is_expert = torch.BoolTensor(is_expert).to(self.device)  # 新增：专家标记张量
        # ==================== 1. 更新Critic网络 ====================
        with torch.no_grad():  # 计算目标Q值
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)  # 利用批量采样数据s'计算a'，近似期望运算
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)  # 得到目标Q'
        # 更新Critic价值网络
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        # ==================== 2. 更新策略网络 ====================
        pi, log_pi, _ = self.policy.sample(state_batch)  # 策略网络采样当前状态的动作 pi，并计算其对数概率 log_pi。

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # RL损失
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # BC损失（仅对专家数据）
        if is_expert.any():
            expert_states = state_batch[is_expert]
            expert_actions = action_batch[is_expert]
            policy_expert_actions, _, _ = self.policy.sample(expert_states)
            bc_loss = F.mse_loss(policy_expert_actions, expert_actions)
        else:
            bc_loss = torch.tensor(0.0).to(self.device)

        # 组合损失
        total_policy_loss = policy_loss + bc_weight * bc_loss

        self.policy_optim.zero_grad()
        total_policy_loss.backward()#求解组合损失
        self.policy_optim.step()
        # ==================== 3. 更新温度参数 ====================
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # ==================== 4. 目标网络更新 ====================
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters(模型加载与保存):保存策略网络、主价值网络、目标价值网络的参数，以及优化器状态。
    # 作用：用于保存训练过程中的 “最佳模型” 。即当智能体在评估中表现优于之前时，保存当前智能体的网络参数（包括策略网络、主
    # Critic网络、目标Critic网络 ）以及优化器状态等。方便后续直接加载最佳模型进行测试、部署或继续微调训练等操作。

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(self.save_folder, f"sac_checkpoint_{env_name}_{suffix}.pt")
        print('Saving style_models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),  # 策略网络参数
                    'critic_state_dict': self.critic.state_dict(),  # # 主 Critic 网络参数
                    'critic_target_state_dict': self.critic_target.state_dict(),  # 目标 Critic 网络参数
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),  # Critic 优化器状态
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)  # 策略优化器状态

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading style_models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

