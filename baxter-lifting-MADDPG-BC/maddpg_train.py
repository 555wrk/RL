import argparse
import os
import datetime
import numpy as np
import torch
from maddpg import MADDPG
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import signal
import sys
import pickle  # 用于保存数据到文件

# 导入双臂环境
from baxter_bullet_env.baxter_gym import BaxterDualArmEnv
from util import FrameStack

# 全局变量：标记是否需要中断保存
interrupted = False


def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print('\n检测到中断，正在保存当前模型...')


signal.signal(signal.SIGINT, signal_handler)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # --------------------环境参数---------------------
    parser.add_argument('--env-name', default='BaxterDualArmGrasp')
    parser.add_argument('--epoch_step', default=100)

    # -------------------SAC代理参数----------------------
    parser.add_argument('--end_epoch', type=int, default=5000)
    parser.add_argument('--policy', default="Deterministic",  # 改为确定性策略
                        help='策略类型: Gaussian | Deterministic (默认: Deterministic)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='每10个episode评估一次策略 (默认: True)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='奖励折扣因子 (默认: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='目标平滑系数(τ) (默认: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='温度参数α，决定熵项相对于奖励的相对重要性 (默认: 0.2)')
    parser.add_argument('--alpha_lr', type=float, default=1e-4,
                        help='alpha的学习率 (默认: 1e-4)')
    parser.add_argument(
        '--automatic_entropy_tuning',
        action='store_true',
        default=True,
        help='自动调整α (默认: True)'
    )
    parser.add_argument('--seed', type=int, default=-1,
                        help='随机种子')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='批大小 (默认: 256)')
    parser.add_argument('--num_steps', type=int, default=5e5,
                        help='最大步数 (默认: 500000)')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='隐藏层大小 (默认: 512)')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='每个环境步的模型更新次数 (默认: 1)')
    parser.add_argument('--start_steps', type=int, default=20000,
                        help='采样随机动作的步数 (默认: 3000)')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        help='目标网络更新间隔 (默认: 1)')
    parser.add_argument('--replay_size', type=int, default=2e6,
                        help='回放缓冲区大小 (默认: 1000000)')
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='使用CUDA (默认: True)')

    # 检查点保存间隔参数：
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='保存检查点的间隔episode数 (默认: 100)')
    parser.add_argument('--expert_data', default="expert_data.pkl", help='专家数据文件路径')
    parser.add_argument('--bc_weight', type=float, default=0.5, help='行为克隆损失权重')
    return parser.parse_args()


if __name__ == "__main__":
    print("训练代码的工作目录:", os.getcwd())
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 10000)

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建双臂环境
    env = BaxterDualArmEnv(
        renders=False,#渲染2
        camera_view=False,#渲染1
        pygame_renders=False,
        max_episode_steps=args.epoch_step
    )

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 初始化代理
    agent = MADDPG(
        num_inputs=23,
        action_dims=[4, 4],  # 左臂4维动作, 右臂4维动作
        args=args
    )
    #print(f"observation_space:{env.observation_space.shape[0]}")
    # 确保保存目录存在
    os.makedirs(agent.save_folder, exist_ok=True)
    best_model_path = os.path.join(agent.save_folder, f"sac_best_model_{args.env_name}.pt")
    final_model_path = os.path.join(agent.save_folder, f"sac_final_model_{args.env_name}.pt")
    checkpoint_path = os.path.join(agent.save_folder, f"sac_checkpoint_{args.env_name}")

    # TensorBoard日志
    log_folder_name = f"log/SAC_{args.env_name}_{args.seed}_{args.policy}_{'autotune' if args.automatic_entropy_tuning else ''}_{current_time}"
    writer = SummaryWriter(log_folder_name)

    # 经验回放
    memory = ReplayMemory(args.replay_size, args.seed)

    # 加载专家数据到回放缓冲区
    if os.path.exists(args.expert_data):
        print(f"加载专家数据: {args.expert_data}")
        with open(args.expert_data, 'rb') as f:
            expert_data = pickle.load(f)

        for episode in expert_data:
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            success = episode['success']

            # 构建转移样本
            for i in range(len(states) - 1):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = states[i + 1]
                done = 0.0  # 非终止状态

                # 添加到缓冲区（标记为专家数据）
                memory.push(state, action, reward, next_state, done, is_expert=True)

            # 添加终止状态
            if len(states) > 0:
                last_state = states[-1]
                last_action = actions[-1]
                last_reward = rewards[-1]
                memory.push(last_state, last_action, last_reward, last_state, 1.0, is_expert=True)

        print(f"已加载 {len(expert_data)} 条专家轨迹")
    else:
        print(f"警告: 未找到专家数据文件 {args.expert_data}")

    # 训练循环
    total_numsteps = 0
    updates = 0
    best_avg_reward = -np.inf
    best_episode = 0

    # 动态调整BC权重
    initial_bc_weight = args.bc_weight
    bc_decay_rate = 1   #0.995

    try:
        for i_episode in range(1, int(args.end_epoch + 1)):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = env.reset()
            # 动态衰减BC权重
            current_bc_weight = initial_bc_weight * (bc_decay_rate ** i_episode)

            while not done:
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()  # 随机动作
                else:
                    # 替换为MADDPG动作选择
                    left_action = agent.select_action(state, agent_idx=0, evaluate=False)  # 训练时 evaluate=False
                    right_action = agent.select_action(state, agent_idx=1, evaluate=False)
                    action = np.concatenate([left_action, right_action])

                if len(memory) > args.batch_size:
                    # 更新网络
                    for i in range(args.updates_per_step):
                        # 替换为MADDPG更新
                        critic_loss, left_actor_loss, right_actor_loss = agent.update_parameters(
                            memory, args.batch_size, updates, bc_weight=current_bc_weight
                        )

                        # 记录损失
                        writer.add_scalar('loss/critic', critic_loss, updates)
                        writer.add_scalar('loss/left_actor', left_actor_loss, updates)
                        writer.add_scalar('loss/right_actor', right_actor_loss, updates)
                        writer.add_scalar('params/bc_weight', current_bc_weight, updates)  # 新增：记录BC权重
                        updates += 1

                next_state, reward, done, info = env.step(action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # 忽略时间限制的done信号
                mask = 1 if episode_steps == env.max_episode_steps else float(not done)

                # 存入经验回放
                memory.push(state, action, reward, next_state, mask)
                state = next_state

                # 提前终止
                if total_numsteps > args.num_steps:
                    done = True
                    break

            writer.add_scalar('reward/train', episode_reward, i_episode)
            print("Episode: {}, 总步数: {}, 本episode步数: {}, 奖励: {:.2f}".format(
                i_episode, total_numsteps, episode_steps, episode_reward))

            # 评估代理
            if i_episode % 20 == 0 and args.eval:
                print(f"评估代理: Episode {i_episode}")
                avg_reward = 0.
                episodes = 5

                for ep in range(episodes):
                    state = env.reset()
                    test_step = 0
                    episode_reward = 0
                    done = False

                    while not done and test_step < args.epoch_step:
                        left_action = agent.select_action(state, agent_idx=0, evaluate=True)
                        right_action = agent.select_action(state, agent_idx=1, evaluate=True)
                        action = np.concatenate([left_action, right_action])
                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        test_step += 1
                        state = next_state

                    avg_reward += episode_reward
                    print(f"评估 {ep + 1}/{episodes}, 奖励: {episode_reward:.2f}")

                avg_reward /= episodes

                # 保存最佳完整模型
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_episode = i_episode
                    try:
                        torch.save(agent, best_model_path)
                        print(f"保存最佳完整模型到 {best_model_path}，奖励: {avg_reward:.2f}，episode: {best_episode}")
                    except Exception as e:
                        print(f"保存最佳模型时出错: {e}")

                # 记录到TensorBoard
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
                print("----------------------------------------")
                print("测试周期: {}, 平均奖励: {:.2f}".format(i_episode, avg_reward))
                print("----------------------------------------")

            # 定期保存检查点
            if i_episode % args.checkpoint_interval == 0:
                try:
                    torch.save(agent, f"{checkpoint_path}_ep{i_episode}.pt")
                    print(f"保存检查点到 {checkpoint_path}_ep{i_episode}.pt")
                except Exception as e:
                    print(f"保存检查点时出错: {e}")

            # 检测中断
            if interrupted:
                try:
                    torch.save(agent, f"{checkpoint_path}_interrupt_ep{i_episode}.pt")
                    print(f"中断保存当前模型到 {checkpoint_path}_interrupt_ep{i_episode}.pt")
                except Exception as e:
                    print(f"中断保存失败: {e}")
                break

    except KeyboardInterrupt:
        print("\n训练被手动中断")
        if not interrupted:  # 如果信号处理未触发保存
            try:
                torch.save(agent, f"{checkpoint_path}_manual_interrupt.pt")
                print(f"手动中断保存模型到 {checkpoint_path}_manual_interrupt.pt")
            except Exception as e:
                print(f"手动中断保存失败: {e}")

    # 训练结束后保存最终模型
    if not interrupted:
        try:
            torch.save(agent, final_model_path)
            print(f"保存最终完整模型到 {final_model_path}")
        except Exception as e:
            print(f"保存最终模型时出错: {e}")

    env.close()