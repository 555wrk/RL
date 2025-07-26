'''
因为实际的训练的模型是在vscode上通过远程连接服务器的GPU进行训练并保存的，导致模型加载后默认尝试使用 CUDA 设备。
实际的，模型导入确实在本地pycharm的cpu上进行实验的。所以这里的代码进行了强制模型使用 CPU 设备，
修改测试脚本，在加载模型后强制将模型的所有参数和内部设备属性切换为 CPU
'''

import argparse
import os
import numpy as np
import torch

# 导入双臂环境
from baxter_bullet_env.baxter_gym import BaxterDualArmEnv
# 注释：不使用FrameStack，避免状态堆叠导致维度异常
# from util import FrameStack


def parse_args():
    parser = argparse.ArgumentParser(description='双臂协同抓取测试')
    parser.add_argument('--env-name', default='BaxterDualArmGrasp')
    parser.add_argument('--epoch_step', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, metavar='N', help='随机种子')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 10000)

    # 创建双臂环境（不使用FrameStack，保持原始23维状态）
    env = BaxterDualArmEnv(  # 直接使用原始环境，不堆叠状态
        renders=True,
        camera_view=False,
        pygame_renders=False,
        max_episode_steps=args.epoch_step
    )

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载训练好的代理，并强制使用CPU
    agent = torch.load(
        "./checkpoints/20250716-152528_seed_6713/maddpg_best_model_BaxterDualArmGrasp.pt",
        map_location=torch.device('cpu'),
        weights_only=False
    )

    # 强制修改模型内部的device属性为CPU
    agent.device = torch.device('cpu')
    for actor in agent.actors:
        actor.to(agent.device)
    agent.critic.to(agent.device)
    agent.critic_target.to(agent.device)

    episodes = 10  # 测试10个episode

    for episode in range(episodes):
        print(f"测试 episode_{episode + 1}/{episodes}")
        state = env.reset()  # 此时state是23维，与训练时一致
        step = 0
        episode_reward = 0
        done = False

        while not done and step < args.epoch_step:
            # 状态已是23维，无需额外处理
            state_np = state

            # 分别为左臂（0）和右臂（1）选择动作
            left_action = agent.select_action(state_np, agent_idx=0, evaluate=True)
            right_action = agent.select_action(state_np, agent_idx=1, evaluate=True)
            # 合并动作（左臂4维 + 右臂4维，共8维）
            action = np.concatenate((left_action, right_action))

            next_state, reward, done, info = env.step(action)
            env.render()
            episode_reward += reward
            step += 1
            state = next_state

            # 打印中间信息
            if step % 20 == 0:
                print(f"Step {step}, Reward: {reward:.2f}, Total: {episode_reward:.2f}")
                if 'collision' in info:
                    print(f"碰撞: {info['collision']}")

        print(f"Episode {episode + 1} 完成, 总奖励: {episode_reward:.2f}, 步数: {step}")

    env.close()