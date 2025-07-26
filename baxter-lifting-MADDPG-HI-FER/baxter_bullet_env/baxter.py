"""
Baxter 双臂控制模块
主要修改点：
1. 类名改为BaxterArm
2. 添加arm参数区分左右臂
3. 根据左右臂设置不同的关节ID
4. 添加get_gripper_state方法
5. 重置时设置另一臂的安全位置
"""

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


class BaxterArm:  # 修改：类名改为BaxterArm
    """
    Baxter 单臂控制类
    """

    def __init__(self, baxter_id, arm="left", timeStep=0.01, os_min=[-1, -1, -1], os_max=[1, 1, 1]):
        # 修改：添加arm参数
        self.arm = arm  # "left" or "right"

        # 定义工作空间范围
        self.os_min = os_min
        self.os_max = os_max
        self.timeStep = timeStep

        self.maxVelocity = .4
        self.maxForce = 200 # 原200
        self.fingerForce =25 #原10

        # 根据手臂类型设置不同的关节ID
        if arm == "left":
            self.endEffector_id = 48
            self.finger_a_id = 49
            self.finger_b_id = 51
            self.finger_tips_a_id = 50
            self.finger_tips_b_id = 52

        else:  # right arm
            self.endEffector_id = 26  # 修改：右臂末端执行器ID
            self.finger_a_id = 27     # 修改：右臂夹爪关节A
            self.finger_b_id = 29     # 修改：右臂夹爪关节B
            self.finger_tips_a_id = 28 # 修改：右臂夹爪尖端A
            self.finger_tips_b_id = 30 # 修改：右臂夹爪尖端B

        self.gripper_open = 0.020833
        self.gripper_close = 0

        # 加载baxter模型
        self.baxter_path = baxter_path
        self.baxterId = baxter_id
        # 关键修改：先定义numJoints属性
        self.numJoints = p.getNumJoints(self.baxterId)
        # 控制所有关节（与单臂一致）
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.baxterId, i)
            if jointInfo[3] > -1:  # 所有可动关节
                self.motorIndices.append(i)

                # 标记当前臂的关节索引
            self.arm_indices = [34, 35, 36, 37, 38, 40, 41, 49, 51] if arm == "left" else [12, 13, 14, 15, 16, 18, 19,
                                                                                           27, 29]
        # 定义关节范围
        self.ll, self.ul, self.jr = self.getJointRanges(self.baxterId)

        # 设置初始姿势
        '''self.rp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   -0.8462368022380607, -1.630866219617225,
                   -0.08415513820451972, 2.0843043904431457,
                   -0.002042621644835222, 1.1254427955750927,
                   -0.1461959296684458, 0.0, 0.0)'''
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
        self.reset(self.ini_pos)

    def getJointRanges(self, bodyId, includeFixed=False):
        # 返回所有关节的范围（与单臂一致）
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
        """
        重置机器人状态
        """
        # 仅重置基座位置（双臂共享）
        #if self.arm == "left":
        p.resetBasePositionAndOrientation(self.baxterId, [0.000, 0.000000, 0.00000],
                                              [0.000000, 0.000000, 0.000000, 1.000000])

        # 逆运动学计算
        orn = p.getQuaternionFromEuler([0, -math.pi, math.pi/2])
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



    def osc(self, motorCommands):
        """
        操作空间控制
        """
        d_position = motorCommands#相对位移量
        if self.arm == "left":
            orientation = p.getQuaternionFromEuler([0, -math.pi, math.pi/2])
        else:
            orientation = p.getQuaternionFromEuler([0, -math.pi, math.pi/2 ])

        # 获取当前状态
        state = p.getLinkState(self.baxterId, self.endEffector_id)
        end_pose = np.array(state[4])
        #print(f"{self.arm}当前位置: {end_pose}")
        # 计算新位置
        new_end_pose = end_pose + d_position
        new_end_pose = np.clip(new_end_pose, self.os_min, self.os_max)
        #print(f"目标位移: {d_position}")
        #print(f"目标位置: {new_end_pose}")
        # 逆运动学求解
        jointPoses = p.calculateInverseKinematics(
            self.baxterId, self.endEffector_id,
            new_end_pose, orientation,
            lowerLimits=self.ll, upperLimits=self.ul,
            jointRanges=self.jr, restPoses=self.rp,
            solver=p.IK_SDLS,  # 使用更稳定的Damped Least Squares算法
            maxNumIterations=200, residualThreshold=1e-5
        )
        # 根据手臂类型选择正确的关节索引
        if self.arm == "left":
            # 左臂使用最后9个索引
            joint_indices = self.motorIndices[-9:]  # 索引为 [34, 35, 36, 37, 38, 40, 41, 49, 51]
            target_positions = jointPoses[-9:]
        else:
            # 右臂使用第2到第10个索引
            joint_indices = self.motorIndices[1:10]  # 索引为 [12, 13, 14, 15, 16, 18, 19, 27, 29]
            target_positions = jointPoses[1:10]  # 注意：这里假设jointPoses的顺序与motorIndices对应
        p.setJointMotorControlArray(
            bodyUniqueId=self.baxterId,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[self.maxForce] * len(joint_indices)
        )
        # 更新物理引擎（关键步骤）
        '''update_steps = int(0.5 / self.timeStep)  # 运行0.5秒
        for _ in range(update_steps):
            p.stepSimulation()
            time.sleep(self.timeStep)  # 与时间步长同步'''
        # 打印实际执行后的位置
        #actual_pose = p.getLinkState(self.baxterId, self.endEffector_id)[4]
        #print(f"{self.arm}执行后位置: {actual_pose}")  # 调试信息
    def gripper_control(self, gripper_state):
        """
        控制夹爪
        """
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
        # 新增：用于观测空间
        pos_a = p.getJointState(self.baxterId, self.finger_a_id)[0]
        return (pos_a / self.gripper_open - 0.5) * 2

    def gripper_close_open(self, gripper_command):
        """
        开关夹爪命令
        """
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


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pb_path)
    p.resetDebugVisualizerCamera(2., 135, 0., [0.52, 0.2, np.pi / 4.])

    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
    baxter_id = p.loadURDF("./models/baxter_description/urdf/baxter.urdf", useFixedBase=True)

    # 测试双臂
    left_arm = BaxterArm(baxter_id, arm="left")
    right_arm = BaxterArm(baxter_id, arm="right")

    # 先让左臂向上移动 0.1 米
   # print("左臂向上移动 0.2 米")
    #left_arm.osc([0.2, 0.2, 0.2])
    #time.sleep(2)
    # 最后让左臂向左移动 0.1 米
    #print("左臂向下移动 0.2 米")
    #left_arm.osc([-0.1, -0.1, -0.1])
    #time.sleep(2)
    # 最后让左臂向左移动 0.1 米
    #left_arm.osc([0, 0, 0])

    # 控制夹爪
    left_arm.gripper_control(1.0)  # 打开
    right_arm.gripper_control(1.0)  # 打开