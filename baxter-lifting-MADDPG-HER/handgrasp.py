# test_collision.py
import numpy as np
from baxter_gym import BaxterDualArmEnv



def handgrasp_detection():
    env = BaxterDualArmEnv(renders=True, pygame_renders=False)
    env.reset()

    print("\n=== 测试开始 ===")

    for i in range(100):
        # 获取当前位置
        state = env.get_physic_state()
        left_ee_pos = np.array(state[0:3])
        right_ee_pos = np.array(state[3:6])
        block_pos = np.array(state[6:9])  # 物体中心位置
        block_dims = [0.012, 0.3, 0.012]

        # 物体左侧目标抓取点（沿细长方向的两端）
        block_left_target = block_pos + np.array([0, block_dims[1] / 2,0])
        # 物体右侧目标抓取点
        block_right_target = block_pos + np.array([0, -block_dims[1] / 2, 0])
        action = np.array([block_left_target[0]-left_ee_pos[0], block_left_target[1]-left_ee_pos[1], block_left_target[2]-0.1, 0,
                           block_right_target[0]-right_ee_pos[0], block_right_target[1]-right_ee_pos[1], block_right_target[2]-0.1, 0])  # 左臂向右，右臂向左
        _, _, _, info = env.step(action)

        distance1 = np.linalg.norm(left_ee_pos - block_left_target)
        print(f"Step {i}:左臂位置: {left_ee_pos}")
        print(f"Step {i}:右臂位置: {right_ee_pos}")
        distance2 = np.linalg.norm(right_ee_pos - block_right_target)
        print(f"Step {i}:左臂目标位置: {block_left_target}")
        print(f"Step {i}:右臂目标位置: {block_right_target}")
        print(f"Step {i}: 左臂距离 = {distance1:.3f}m")
        print(f"Step {i}: 右臂距离 = {distance2:.3f}m")

    print("\n=== 测试完成 ===")
    env.close()


if __name__ == "__main__":
    handgrasp_detection()