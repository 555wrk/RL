import numpy as np
from baxter_bullet_env.baxter_gym import BaxterDualArmEnv


def test_collision_detection():
    env = BaxterDualArmEnv(renders=True, pygame_renders=False)
    env.reset()

    print("\n=== 碰撞检测测试开始 ===")

    # 测试1：无碰撞时检测
    print("\n测试1：无碰撞状态")
    no_collision = env._arm_collision()
    print(f"碰撞检测结果: {no_collision} (应为False)")

    # 测试2：末端执行器接近，碰撞时暂停
    print("\n测试2：末端执行器接近（碰撞时暂停）")
    for i in range(50):
        action = np.array([0.2, -0.5, 0, 0, 0.2, 0.5, 0, 0])  # 左臂向右，右臂向左
        _, _, _, info = env.step(action)

        # 获取当前位置
        state = env.get_physic_state()
        left_ee_pos = np.array(state[0:3])
        right_ee_pos = np.array(state[3:6])
        distance = np.linalg.norm(left_ee_pos - right_ee_pos)
        print(f"Step {i}: 双臂距离 = {distance:.3f}m")

        if info["collision"]:
            print("检测到碰撞! 仿真已暂停")
            print(f"碰撞时的末端执行器位置:")
            print(f"左臂: {left_ee_pos}")
            print(f"右臂: {right_ee_pos}")

            # 碰撞后保持当前状态，允许用户观察
            input("按Enter继续...")
            break

    # 测试3：碰撞后分离
    print("\n测试3：碰撞后分离")
    for i in range(20):
        action = np.array([0, 0.5, 0, 0, 0, -0.5, 0, 0])  # 左臂向左，右臂向右
        _, _, _, info = env.step(action)

        state = env.get_physic_state()
        left_ee_pos = np.array(state[0:3])
        right_ee_pos = np.array(state[3:6])
        distance = np.linalg.norm(left_ee_pos - right_ee_pos)
        print(f"Step {i}: 分离后距离 = {distance:.3f}m")

        if not info["collision"]:
            print("碰撞解除!")
            break

    print("\n=== 碰撞检测测试完成 ===")
    env.close()


if __name__ == "__main__":
    test_collision_detection()