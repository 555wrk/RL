import pickle
import numpy as np



def view_expert_episodes(file_path="baxter_bullet_env/expert_data.pkl"):
    """查看按轨迹保存的专家数据，每个轨迹为一个字典"""
    try:
        with open(file_path, 'rb') as f:
            all_episodes = []
            # 循环读取所有轨迹数据（可能有多个批次）
            while True:
                try:
                    data_batch = pickle.load(f)
                    all_episodes.extend(data_batch)
                except EOFError:
                    break

            print(f"===== 专家数据概览 =====")
            print(f"总轨迹数: {len(all_episodes)}")
            if len(all_episodes) == 0:
                print("警告：数据文件为空！")
                return

            # 统计成功轨迹数
            successful = sum(1 for ep in all_episodes if ep['success'])
            print(f"成功轨迹: {successful}/{len(all_episodes)} ({successful / len(all_episodes) * 100:.2f}%)")

            # 计算平均轨迹长度
            avg_length = np.mean([len(ep['states']) for ep in all_episodes])
            print(f"平均轨迹长度: {avg_length:.2f} 步\n")

            # 选择一个典型轨迹展示（优先成功轨迹）
            sample_episode = next((ep for ep in all_episodes if ep['success']), all_episodes[0])
            print(f"===== 轨迹 #{sample_episode['episode_id']} 详细信息 =====")
            print(f"轨迹长度: {len(sample_episode['states'])} 步")
            print(f"抓取结果: {'成功' if sample_episode['success'] else '失败'}\n")

            # 展示轨迹中的第一个步骤
            print("--- 步骤0 详细信息 ---")
            state = sample_episode['states'][0]
            action = sample_episode['actions'][0]
            reward = sample_episode['rewards'][0]

            print("\n--- State ---")
            print(f"形状: {state.shape}")
            print(f"左臂末端位置: {state[0:3]}")
            print(f"右臂末端位置: {state[3:6]}")
            print(f"物体位置: {state[6:9]}")
            print(f"物体欧拉角状态: {state[9:12]}")
            print(f"左夹爪状态: {state[12]:.2f}")  # 假设左夹爪状态在索引12（根据实际状态维度调整）
            print(f"右夹爪状态: {state[13]:.2f}")
            print(f"左臂到目标状态: {state[14:17]}")
            print(f"右臂到目标状态: {state[17:20]}")
            print(f"双臂之间状态: {state[20:23]}")
            print("\n--- Action ---")
            print(f"形状: {action.shape}")
            print(f"左臂控制量: {action[0:3]}")
            print(f"左夹爪状态: {action[3]:.2f}")
            print(f"右臂控制量: {action[4:7]}")
            print(f"右夹爪状态: {action[7]:.2f}")

            print(f"\n--- Reward ---: {reward:.4f}")

            # 展示前10个轨迹的平均奖励
            if len(all_episodes) > 10:
                print("\n===== 前10个轨迹的平均奖励 =====")
                for i in range(min(10, len(all_episodes))):
                    ep = all_episodes[i]
                    avg_reward = np.mean(ep['rewards'])
                    result = "成功" if ep['success'] else "失败"
                    print(f"轨迹{i}: 平均奖励={avg_reward:.4f} | 结果={result}")

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except Exception as e:
        print(f"查看数据时出错: {str(e)}")




def get_episode_by_id(episode_id, all_episodes):
    """根据ID获取特定轨迹"""
    for ep in all_episodes:
        if ep['episode_id'] == episode_id:
            return ep
    return None


if __name__ == "__main__":
    # 可指定文件路径
    view_expert_episodes()