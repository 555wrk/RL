# dual_arm_grasping_improved.py
import numpy as np
from baxter_bullet_env.baxter_gym import BaxterDualArmEnv
import time
import pybullet as p
from baxter_bullet_env.baxter import BaxterArm

class DualArmGrasper:
    def __init__(self, env):
        self.env = env
        self.block_dims = [0.012, 0.35, 0.012]  # 物体尺寸 [半长, 半宽, 半高]
        self.approach_height = 0.1  # 接近高度（物体上方）
        self.grasp_height = 0.11 # 抓取高度（低于物体顶部）
        self.lift_height = 0.1 # 提升高度
        self.tolerance = 0.025  # 容差放宽到2cm

        # 初始化关键ID（从环境中获取）
        self.baxterId = None
        self.blockUid = None
        self.robot_left = None
        self.robot_right = None
        self.finger_tips_a_id = None
        self.finger_tips_b_id = None

    def execute_grasp(self):
        """执行完整的双臂协同抓取流程"""
        state = self.env.reset()
        print("\n=== 开始抓取流程 ===")

        # 从环境中获取关键信息
        self.baxterId = self.env.baxterId
        self.blockUid = self.env.blockUid
        self.robot_left = self.env.robot_left
        self.robot_right = self.env.robot_right
        self.finger_tips_a_id = self.robot_left.finger_tips_a_id
        self.finger_tips_b_id = self.robot_left.finger_tips_b_id  # 假设左右臂夹爪ID结构相同

        # 1. 移动到物体正上方
        block_pos = np.array(state[6:9])
        block_top_z = block_pos[2] + self.block_dims[2]  # 物体顶部Z坐标
        block_left_target = block_pos + np.array([0, self.block_dims[1] / 1.5, block_top_z + self.approach_height])
        block_right_target = block_pos + np.array([0, -self.block_dims[1] / 1.5, block_top_z + self.approach_height])

        # 移动到物体上方（张开夹爪）
        success = self.move_to_target(block_left_target, block_right_target, gripper_state=1.0)
        if not success:
            print("步骤1失败：无法移动到物体上方")
            return False
        print("步骤1完成：移动到物体上方")
        # 2. 下降到抓取高度（物体顶部下方）
        block_left_target[2] -= self.grasp_height
        block_right_target[2]-= self.grasp_height

        success = self.move_to_target(block_left_target, block_right_target, gripper_state=1.0)
        if not success:
            return False
        print(f"步骤2完成：下降到抓取高度")
        # 3. 闭合夹爪抓取物体
        '''success = self.control_gripper( gripper_state=-1.0)  # -1表示闭合
        if not success:
            return False'''
        print("\n2. 闭合夹爪...")
        self._print_gripper_state()
        self._set_gripper_state(-1.0)  # -1.0表示完全闭合
        for _ in range(100):  # 增加模拟步数到100
            p.stepSimulation()
            time.sleep(0.01)
        self._print_gripper_state()
        if not self._grasped():
            print(f"步骤3失败：未接触物体")
            return False
        print(f"步骤3完成：接触物体")
        # 4. 提升物体
        block_left_target[2] = block_top_z + self.lift_height
        block_right_target[2] = block_top_z + self.lift_height

        success = self.move_to_target(block_left_target, block_right_target, gripper_state=-1.0)
        if not success:
            print(f"步骤4失败：无法提升物体")
            return False

        print(f"步骤4完成：提升物体")
        return True


    def move_to_target(self, left_target, right_target, gripper_state=1.0, max_steps=250):
        """移动双臂到目标位置（右臂专用优化）"""

        for step in range(max_steps):
            # 获取当前状态
            state = self.env.get_physic_state()
            left_ee_pos = np.array(state[0:3])
            right_ee_pos = np.array(state[3:6])

            # 计算左臂误差
            left_error = left_target - left_ee_pos
            left_distance = np.linalg.norm(left_error)
            # 计算右臂误差
            right_error = right_target - right_ee_pos
            right_distance = np.linalg.norm(right_error)

            left_action = left_error
            #left_action = np.clip(left_action, -0.1, 0.1)
            right_action = right_error
            #right_action =np.clip(right_action, -0.1, 0.1)

            # 检查是否到达目标
            left_arrived = left_distance < self.tolerance
            right_arrived = right_distance < self.tolerance

            # 打印实时距离（调试用）
            if step % 10 == 0:
                print(f"[Step {step}] 左臂误差: {left_distance:.4f}m | 右臂误差: {right_distance:.4f}m")

            if left_arrived and right_arrived:
                print(f"移动完成于步骤 {step}")
                return True

            # 构建动作向量
            action = np.concatenate([
                left_action, [gripper_state],
                right_action, [gripper_state]
            ])

            # 执行动作
            self.env.step(action)

        # 最终位置检查
        state = self.env.get_physic_state()
        left_ee_pos = np.array(state[0:3])
        right_ee_pos = np.array(state[3:6])
        left_final_dist = np.linalg.norm(left_target - left_ee_pos)
        right_final_dist = np.linalg.norm(right_target - right_ee_pos)

        #print(f"移动完成: 左臂最终误差 {left_final_dist:.4f}m | 右臂最终误差 {right_final_dist:.4f}m")
        return left_final_dist < self.tolerance and right_final_dist < self.tolerance

    def _grasped(self):
        """优化抓取检测 - 基于单臂成功经验"""
        # 左臂接触检测
        left_contact_a = p.getContactPoints(self.baxterId, self.blockUid, self.robot_left.finger_tips_a_id)
        left_contact_b = p.getContactPoints(self.baxterId, self.blockUid, self.robot_left.finger_tips_b_id)
        left_grasped = left_contact_a != () and left_contact_b != ()  # 要求双尖端接触

        # 右臂接触检测
        right_contact_a = p.getContactPoints(self.baxterId, self.blockUid, self.robot_right.finger_tips_a_id)
        right_contact_b = p.getContactPoints(self.baxterId, self.blockUid, self.robot_right.finger_tips_b_id)
        right_grasped = right_contact_a != () and right_contact_b != ()  # 要求双尖端接触
        # 打印接触点数量（调试用）
        print(f"[接触检测] 左臂A: {len(left_contact_a)}, 左臂B: {len(left_contact_b)}")
        print(f"[接触检测] 右臂A: {len(right_contact_a)}, 右臂B: {len(right_contact_b)}")
        return left_grasped and right_grasped

    def control_gripper(self, gripper_state=-1.0, max_steps=30):
        """控制夹爪状态"""
        for step in range(max_steps):
            # 保持当前位置，只更新夹爪状态
            action = np.array([0, 0, 0, gripper_state, 0, 0, 0, gripper_state])
            self.env.step(action)
        return True
    def _set_gripper_state(self, gripper_state):
        """设置夹爪状态"""
        print(f"设置夹爪状态: {gripper_state}")
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

    def _print_gripper_state(self):
        """打印当前夹爪状态"""
        left_state = self.robot_left.get_gripper_state()
        right_state = self.robot_right.get_gripper_state()
        print(f"当前夹爪状态：左={left_state:.2f}，右={right_state:.2f}")

def main():
    # 创建环境
    env = BaxterDualArmEnv(renders=True, pygame_renders=False)

    # 创建抓取器
    grasper = DualArmGrasper(env)

    # 执行抓取测试
    success_rate = 0
    num_tests = 5

    for i in range(num_tests):
        print(f"\n=== 测试 {i + 1}/{num_tests} ===")
        success = grasper.execute_grasp()
        success_rate += success
        time.sleep(1)  # 观察结果

    print(f"\n测试完成：成功率 {success_rate / num_tests:.2%}")

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()