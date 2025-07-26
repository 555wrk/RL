import os, inspect
import pybullet as p
import numpy as np
import math
import pybullet_data
import time

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
os.sys.path.insert(0, parent_dir)
pb_path = pybullet_data.getDataPath()
baxter_path = "./models/baxter_description/urdf/baxter.urdf"


class BaxterArm:
    """
    Baxter 单臂控制类 - 集成KUKA式精准控制
    """

    def __init__(self, baxter_id, arm="left", timeStep=0.01, os_min=[-1, -1, -1], os_max=[1, 1, 1]):
        self.arm = arm  # "left" or "right"

        # 定义工作空间范围
        self.os_min = os_min
        self.os_max = os_max
        self.timeStep = timeStep

        self.maxVelocity = .4
        self.maxForce = 200  # 关节最大驱动力
        self.fingerForce = 25  # 夹爪力

        # 根据手臂类型设置关节ID
        if arm == "left":
            self.endEffector_id = 48
            self.finger_a_id = 49
            self.finger_b_id = 51
            self.finger_tips_a_id = 50
            self.finger_tips_b_id = 52
        else:  # right arm
            self.endEffector_id = 26
            self.finger_a_id = 27
            self.finger_b_id = 29
            self.finger_tips_a_id = 28
            self.finger_tips_b_id = 30

        self.gripper_open = 0.020833
        self.gripper_close = 0

        # 加载baxter模型
        self.baxter_path = baxter_path
        self.baxterId = baxter_id
        self.numJoints = p.getNumJoints(self.baxterId)

        # 控制所有关节
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.baxterId, i)
            if jointInfo[3] > -1:  # 所有可动关节
                self.motorIndices.append(i)

        # 标记当前臂的关节索引
        self.arm_indices = [34, 35, 36, 37, 38, 40, 41, 49, 51] if arm == "left" else [12, 13, 14, 15, 16, 18, 19, 27,
                                                                                       29]

        # 定义关节范围
        self.ll, self.ul, self.jr = self.getJointRanges(self.baxterId)

        # 设置初始姿势
        self.rp = (0.0, 0.8462368022380607, -1.630866219617225,
                   -0.08415513820451972, 2.0843043904431457,
                   0.002042621644835222, 1.1254427955750927,
                   -0.1461959296684458, 0.0, 0.0,
                   -0.8462368022380607, -1.630866219617225,
                   -0.08415513820451972, 2.0843043904431457,
                   -0.002042621644835222, 1.1254427955750927,
                   -0.1461959296684458, 0.0, 0.0)

        # 初始末端位置
        self.ini_pos = [0.5, 0.2, 0.2] if arm == "left" else [0.5, -0.2, 0.2]

        # 配置物理引擎参数（与ik_kuka.py保持一致）
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.001,  # 1ms细分步长，提高模拟精度
            numSubSteps=10,  # 每个控制步细分10个子步
            solverResidualThreshold=1e-7  # 提高约束求解精度
        )
        p.setRealTimeSimulation(1)  # 启用实时模拟

        self.reset(self.ini_pos)

    def getJointRanges(self, bodyId, includeFixed=False):
        # 返回所有关节的范围
        lowerLimits, upperLimits, jointRanges = [], [], []
        for i in range(p.getNumJoints(bodyId)):
            jointInfo = p.getJointInfo(bodyId, i)
            if includeFixed or jointInfo[3] > -1:
                ll, ul = jointInfo[8:10]
                lowerLimits.append(ll)
                upperLimits.append(ul)
                jointRanges.append(ul - ll)
        return [lowerLimits, upperLimits, jointRanges]

    def reset(self, pose_gripper):
        """重置机器人状态"""
        p.resetBasePositionAndOrientation(self.baxterId, [0.000, 0.000000, 0.00000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])

        # 逆运动学计算
        orn = p.getQuaternionFromEuler([0, -math.pi, math.pi / 2])
        pose = pose_gripper[0:3] if len(pose_gripper) > 3 else pose_gripper
        jointPoses = p.calculateInverseKinematics(
            self.baxterId, self.endEffector_id,
            pose, orn,
            lowerLimits=self.ll, upperLimits=self.ul,
            jointRanges=self.jr, restPoses=self.rp,
            solver=p.IK_SDLS, maxNumIterations=200, residualThreshold=1e-5
        )

        # 设置关节状态
        for i in range(len(self.motorIndices)):
            p.resetJointState(self.baxterId, self.motorIndices[i], jointPoses[i])

        # 设置夹爪状态
        p.resetJointState(self.baxterId, self.finger_a_id, self.gripper_close)
        p.resetJointState(self.baxterId, self.finger_b_id, self.gripper_close)

    def accurateCalculateInverseKinematics(self, targetPos, threshold=0.001, maxIter=100):
        """精确逆运动学求解（集成KUKA式迭代优化）"""
        # 关节阻尼参数（减少关节角度突变）
        joint_damping = [0.1] * len(self.motorIndices)

        closeEnough = False
        iter = 0
        dist2 = 1e30
        while (not closeEnough and iter < maxIter):
            jointPoses = p.calculateInverseKinematics(
                self.baxterId, self.endEffector_id, targetPos,
                lowerLimits=self.ll, upperLimits=self.ul,
                jointRanges=self.jr, restPoses=self.rp,
                jointDamping=joint_damping  # 关键参数：关节阻尼
            )

            for i in range(len(self.motorIndices)):
                p.resetJointState(self.baxterId, self.motorIndices[i], jointPoses[i])

            ls = p.getLinkState(self.baxterId, self.endEffector_id)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1

        # 打印迭代结果（调试用）
        if iter >= maxIter:
            print(f"[警告] IK迭代达到最大次数({maxIter})，最终误差: {math.sqrt(dist2):.6f}m")

        return jointPoses

    def osc(self, motorCommands):
        """操作空间控制（优化动态参数）"""
        d_position = motorCommands  # 相对位移量

        # 获取当前状态
        state = p.getLinkState(self.baxterId, self.endEffector_id)
        end_pose = np.array(state[4])

        # 计算新位置
        new_end_pose = end_pose + d_position
        new_end_pose = np.clip(new_end_pose, self.os_min, self.os_max)

        # 精确逆运动学求解
        jointPoses = self.accurateCalculateInverseKinematics(new_end_pose)

        # 根据手臂类型选择关节索引
        if self.arm == "left":
            joint_indices = self.motorIndices[-9:]  # 左臂关节索引
            target_positions = jointPoses[-9:]
        else:
            joint_indices = self.motorIndices[1:10]  # 右臂关节索引
            target_positions = jointPoses[1:10]

            # 修复：将参数名`bodyUniqueId`改为`bodyIndex`
        p.setJointMotorControlArray(
            bodyIndex=self.baxterId,  # 修正参数名：bodyUniqueId → bodyIndex
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[self.maxForce] * len(joint_indices)
        )

        # 更新物理引擎（关键：确保指令被实时执行）
        update_steps = int(self.timeStep / 0.001)  # 按时间步长细分模拟步
        for _ in range(update_steps):
            p.stepSimulation()
            time.sleep(0.001)  # 与物理引擎步长同步

        # 计算并打印实际位置误差（调试用）
        actual_pose = p.getLinkState(self.baxterId, self.endEffector_id)[4]
        error = np.linalg.norm(actual_pose - new_end_pose)
        print(f"{self.arm}臂位移: {np.round(d_position, 4)} | 末端误差: {error:.6f}m")

    def gripper_control(self, gripper_state):
        """控制夹爪"""
        gripper_state = (gripper_state + 1) * 0.5
        gripper_state_a = np.clip(gripper_state * self.gripper_open, self.gripper_close, self.gripper_open)
        gripper_state_b = np.clip(-gripper_state * self.gripper_open, -self.gripper_open, self.gripper_close)

        p.setJointMotorControl2(
            bodyUniqueId=self.baxterId,
            jointIndex=self.finger_a_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripper_state_a,
            force=self.fingerForce
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.baxterId,
            jointIndex=self.finger_b_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=gripper_state_b,
            force=self.fingerForce
        )

    def get_gripper_state(self):
        """获取夹爪状态 (-1:闭合, 1:打开)"""
        pos_a = p.getJointState(self.baxterId, self.finger_a_id)[0]
        return (pos_a / self.gripper_open - 0.5) * 2

    def gripper_close_open(self, gripper_command):
        """开关夹爪命令"""
        if gripper_command == "close":
            target_position = self.gripper_close
        elif gripper_command == "open":
            target_position = self.gripper_open
        else:
            target_position = p.getJointState(self.baxterId, self.finger_a_id)[0]

        p.setJointMotorControl2(
            bodyUniqueId=self.baxterId,
            jointIndex=self.finger_a_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
            force=self.fingerForce
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.baxterId,
            jointIndex=self.finger_b_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=-target_position,
            force=self.fingerForce
        )


def move_to_target(left_arm, right_arm, left_target, right_target, gripper_state=1.0, max_steps=250):
    """双臂协同移动到目标位置（优化版）"""
    # 动态步长参数
    min_step = 0.002  # 最小步长（2mm）
    max_step = 0.01  # 最大步长（10mm）

    for step in range(max_steps):
        # 获取当前状态
        left_state = p.getLinkState(left_arm.baxterId, left_arm.endEffector_id)
        right_state = p.getLinkState(right_arm.baxterId, right_arm.endEffector_id)
        left_ee_pos = np.array(left_state[4])
        right_ee_pos = np.array(right_state[4])

        # 计算左臂位移（动态限制步长）
        left_displacement = left_target - left_ee_pos
        left_dist = np.linalg.norm(left_displacement)
        left_step = min(max(left_dist * 0.1, min_step), max_step)
        left_displacement = left_displacement / left_dist * left_step if left_dist > 1e-6 else 0

        # 计算右臂位移（动态限制步长）
        right_displacement = right_target - right_ee_pos
        right_dist = np.linalg.norm(right_displacement)
        right_step = min(max(right_dist * 0.1, min_step), max_step)
        right_displacement = right_displacement / right_dist * right_step if right_dist > 1e-6 else 0

        # 打印实时距离（调试用）
        if step % 10 == 0:
            print(f"[Step {step}] 左臂误差: {left_dist:.4f}m | 右臂误差: {right_dist:.4f}m")

        # 执行动作
        left_arm.osc(left_displacement)
        right_arm.osc(right_displacement)
        left_arm.gripper_control(gripper_state)
        right_arm.gripper_control(gripper_state)

        # 检查是否到达目标
        if left_dist < 0.001 and right_dist < 0.001:
            print(f"移动完成于步骤 {step}")
            return True

    # 最终位置检查
    left_dist = np.linalg.norm(left_target - np.array(p.getLinkState(left_arm.baxterId, left_arm.endEffector_id)[4]))
    right_dist = np.linalg.norm(
        right_target - np.array(p.getLinkState(right_arm.baxterId, right_arm.endEffector_id)[4]))

    print(f"[警告] 达到最大步数，最终误差 - 左: {left_dist:.4f}m | 右: {right_dist:.4f}m")
    return left_dist < 0.001 and right_dist < 0.001


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pb_path)
    p.resetDebugVisualizerCamera(2., 135, 0., [0.52, 0.2, np.pi / 4.])

    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
    baxter_id = p.loadURDF("./models/baxter_description/urdf/baxter.urdf", useFixedBase=True)

    # 初始化双臂
    left_arm = BaxterArm(baxter_id, arm="left")
    right_arm = BaxterArm(baxter_id, arm="right")

    # 控制夹爪
    left_arm.gripper_control(1.0)  # 打开
    right_arm.gripper_control(1.0)  # 打开
    ''' # 测试精准控制
    print("===== 测试双臂协同精准控制 =====")

    # 目标位置（左右臂对称）
    left_target = [0.5, 0.3, 0.4]
    right_target = [0.5, -0.3, 0.4]

    # 移动到初始位置
    print(f"移动到初始位置: 左={left_target}, 右={right_target}")
    move_to_target(left_arm, right_arm, left_target, right_target, gripper_state=1.0)
    time.sleep(1)

    # 协同抓取测试 - 缩小间距
    print("执行协同抓取: 双臂向中心移动")
    left_target[1] -= 0.1  # 左臂向左移动0.1m
    right_target[1] += 0.1  # 右臂向右移动0.1m
    move_to_target(left_arm, right_arm, left_target, right_target, gripper_state=-1.0)  # 闭合夹爪
    time.sleep(1)

    # 提升物体
    print("提升物体")
    left_target[2] += 0.2  # 左臂向上移动0.2m
    right_target[2] += 0.2  # 右臂向上移动0.2m
    move_to_target(left_arm, right_arm, left_target, right_target, gripper_state=-1.0)
    time.sleep(1)

    # 放下物体
    print("放下物体")
    left_target[2] -= 0.2  # 左臂向下移动0.2m
    right_target[2] -= 0.2  # 右臂向下移动0.2m
    move_to_target(left_arm, right_arm, left_target, right_target, gripper_state=1.0)  # 打开夹爪

    print("测试完成")

    while True:
        p.stepSimulation()
        time.sleep(0.01)'''