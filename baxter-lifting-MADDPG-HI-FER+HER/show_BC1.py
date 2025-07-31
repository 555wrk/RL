import pickle
import numpy as np


def view_expert_episodes(
    file_path="baxter_bullet_env/expert_data.pkl",
    target_episode_id=None,
    target_index=None,  # 新增：按轨迹索引查询
    show_all_ids=False
):
    """
    支持按ID或索引查询轨迹，索引为0~59（数据集中的顺序位置）
    """
    try:
        with open(file_path, 'rb') as f:
            all_episodes = []
            while True:
                try:
                    data_batch = pickle.load(f)
                    all_episodes.extend(data_batch)
                except EOFError:
                    break

            if len(all_episodes) == 0:
                print("警告：数据文件为空！")
                return

            # 显示数据概览
            print(f"===== 专家数据概览 =====")
            print(f"总轨迹数: {len(all_episodes)}")
            successful = sum(1 for ep in all_episodes if ep['success'])
            print(f"成功轨迹: {successful}/{len(all_episodes)} ({successful / len(all_episodes) * 100:.2f}%)")
            avg_length = np.mean([len(ep['states']) for ep in all_episodes])
            print(f"平均轨迹长度: {avg_length:.2f} 步\n")

            # 显示所有轨迹的索引、ID及状态
            if show_all_ids:
                print("===== 所有轨迹索引、ID及状态 =====")
                for i, ep in enumerate(all_episodes):
                    print(f"轨迹索引 {i} | ID: {ep['episode_id']} | 成功: {'是' if ep['success'] else '否'} | 长度: {len(ep['states'])} 步")
                return

            # 优先按索引查询（解决ID重复问题）
            if target_index is not None:
                if 0 <= target_index < len(all_episodes):
                    target_ep = all_episodes[target_index]
                    print(f"===== 轨迹索引 {target_index}（ID: {target_ep['episode_id']}）完整数据 =====")
                    print(f"轨迹长度: {len(target_ep['states'])} 步")
                    print(f"抓取结果: {'成功' if target_ep['success'] else '失败'}\n")

                    # 遍历每一步显示详细信息（同之前逻辑）
                    for step in range(len(target_ep['states'])):
                        print(f"--- 步骤 {step} 详细信息 ---")
                        state = target_ep['states'][step]
                        action = target_ep['actions'][step]
                        reward = target_ep['rewards'][step]

                        print("\n--- State ---")
                        print(f"形状: {state.shape}")
                        print(f"左臂末端位置: {state[0:3]}")
                        print(f"右臂末端位置: {state[3:6]}")
                        print(f"物体位置: {state[6:9]}")
                        print(f"物体欧拉角状态: {state[9:12]}")
                        print(f"左夹爪状态: {state[12]:.2f}")
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

                        print(f"\n--- Reward ---: {reward:.4f}\n")
                        print("-" * 50)
                else:
                    print(f"错误：索引 {target_index} 超出范围（有效范围0~{len(all_episodes)-1}）")
                return

            # 按ID查询（兼容原有功能，ID重复时返回第一个匹配项）
            if target_episode_id is not None:
                target_ep = get_episode_by_id(target_episode_id, all_episodes)
                if target_ep:
                    # 打印ID对应的轨迹（若有多个，仅显示第一个）
                    index = all_episodes.index(target_ep)  # 显示该ID在数据集中的索引
                    print(f"===== 轨迹ID {target_episode_id}（索引 {index}）完整数据 =====")
                    # 后续步骤信息打印（同上）
                else:
                    print(f"错误：未找到ID为 {target_episode_id} 的轨迹")
                return

            # 未指定参数时显示示例轨迹
            sample_episode = next((ep for ep in all_episodes if ep['success']), all_episodes[0])
            print(f"===== 示例轨迹（索引 {all_episodes.index(sample_episode)}，ID {sample_episode['episode_id']}）=====")
            # 示例信息打印（同上）

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
    except Exception as e:
        print(f"查看数据时出错: {str(e)}")


def get_episode_by_id(episode_id, all_episodes):
    for ep in all_episodes:
        if ep['episode_id'] == episode_id:
            return ep
    return None


if __name__ == "__main__":
    # 用法1：先查看所有轨迹的索引和ID（确认目标轨迹的索引）
    # view_expert_episodes(show_all_ids=True)

    # 用法2：按索引查询（推荐，无歧义）
    view_expert_episodes(target_index=45)  # 例如查看索引56的轨迹