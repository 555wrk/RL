import numpy as np
from baxter_gym import BaxterDualArmEnv
import time
import pybullet as p
from baxter import BaxterArm


class GripperTester:
    def __init__(self, env):
        self.env = env
        self.baxterId = env.baxterId
        self.robot_left = env.robot_left
        self.robot_right = env.robot_right
        self.test_steps = 100  # 每个动作的模拟步数

    def test_gripper_control(self):
        """测试夹爪控制功能"""
        print("\n=== 夹爪控制测试开始 ===")

        # 初始状态：张开夹爪
        print("1. 张开夹爪...")
        self._set_gripper_state(1.0)  # 1.0表示完全张开
        self._simulate_steps(self.test_steps)
        self._print_gripper_state()

        # 闭合夹爪
        print("\n2. 闭合夹爪...")
        self._set_gripper_state(-1.0)  # -1.0表示完全闭合
        self._simulate_steps(self.test_steps)
        self._print_gripper_state()

        # 半开夹爪
        print("\n3. 半开夹爪...")
        self._set_gripper_state(0)  # 0表示半开
        self._simulate_steps(self.test_steps)
        self._print_gripper_state()

        # 再次闭合夹爪
        '''print("\n4. 再次闭合夹爪...")
        self._set_gripper_state(-1.0)
        self._simulate_steps(self.test_steps)
        self._print_gripper_state()'''

        # 张开夹爪（测试结束）
        print("\n5. 张开夹爪（测试结束）")
        self._set_gripper_state(1.0)
        self._simulate_steps(self.test_steps)

        print("\n=== 夹爪控制测试完成 ===")

        # 返回测试结果
        left_state = self.robot_left.get_gripper_state()
        right_state = self.robot_right.get_gripper_state()
        return abs(left_state - 1.0) < 0.1 and abs(right_state - 1.0) < 0.1

    def _set_gripper_state(self, gripper_state):
        """设置夹爪状态"""
        self.robot_left.gripper_control(gripper_state)
        self.robot_right.gripper_control(gripper_state)

        # 添加调试文本显示当前夹爪状态
        state_text = "OPEN" if gripper_state > 0 else "CLOSED"
        p.addUserDebugText(
            f"Gripper: {state_text} ({gripper_state:.2f})",
            [0.5, 0, 0.5],
            textColorRGB=[0, 1, 0] if gripper_state > 0 else [1, 0, 0],
            textSize=1.5,
            lifeTime=0.5
        )

    def _simulate_steps(self, steps):
        """执行指定步数的物理模拟"""
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(0.01)  # 减慢模拟速度以便观察

    def _print_gripper_state(self):
        """打印当前夹爪状态"""
        left_state = self.robot_left.get_gripper_state()
        right_state = self.robot_right.get_gripper_state()
        print(f"当前夹爪状态：左={left_state:.2f}，右={right_state:.2f}")

    def visualize_gripper(self):
        """可视化夹爪位置和状态"""
        # 添加夹爪位置标记
        for i in range(200):
            # 获取夹爪位置
            left_pos = p.getLinkState(self.baxterId, self.robot_left.finger_tips_a_id)[0]
            right_pos = p.getLinkState(self.baxterId, self.robot_right.finger_tips_a_id)[0]

            # 添加位置标记
            p.addUserDebugPoint(
                pointPosition=left_pos,
                pointColorRGB=[1, 0, 0],  # 红色表示左臂
                pointSize=5,
                lifeTime=0.1
            )
            p.addUserDebugPoint(
                pointPosition=right_pos,
                pointColorRGB=[0, 0, 1],  # 蓝色表示右臂
                pointSize=5,
                lifeTime=0.1
            )

            p.stepSimulation()
            time.sleep(0.05)


def main():
    # 创建环境
    env = BaxterDualArmEnv(renders=True, pygame_renders=False)
    env.reset()

    # 创建夹爪测试器
    tester = GripperTester(env)

    # 执行夹爪测试
    success = tester.test_gripper_control()

    # 可视化夹爪（可选）
    # tester.visualize_gripper()

    # 显示测试结果
    print(f"\n夹爪测试结果：{'成功' if success else '失败'}")

    # 保持窗口打开以便观察
    print("\n按Enter键退出...")
    input()

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()