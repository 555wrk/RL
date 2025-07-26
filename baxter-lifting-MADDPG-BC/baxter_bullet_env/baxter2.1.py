"""
Baxter 双臂控制模块
主要修改点：
1. 类名改为BaxterArm
2. 添加arm参数区分左右臂
3. 根据左右臂设置不同的关节ID
4. 添加get_gripper_state方法
5. 绘制期望和实际轨迹
6. 采用单个关节控制以设置位置和速度增益参数
此代码主要目的是实验新的机械臂控制方法，以双臂分别进行圆圈轨迹的运动来实验。
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


class BaxterArm:
    """
    Baxter 单臂控制类
    """

    def __init__(self, baxter_id, arm="left", timeStep=0.01, os_min=[-1, -1, -1], os_max=[1, 1, 1]):
        self.arm = arm
        self.os_min = os_min
        self.os_max = os_max
        self.timeStep = timeStep
        self.maxVelocity = .4
        self.maxForce = 200
        self.fingerForce = 25
        self.gripper_open = 0.020833
        self.gripper_close = 0

        if arm == "left":
            self.endEffector_id = 48
            self.finger_a_id = 49
            self.finger_b_id = 51
            self.finger_tips_a_id = 50
            self.finger_tips_b_id = 52
        else:
            self.endEffector_id = 26
            self.finger_a_id = 27
            self.finger_b_id = 29
            self.finger_tips_a_id = 28
            self.finger_tips_b_id = 30

        self.baxter_path = baxter_path
        self.baxterId = baxter_id
        self.numJoints = p.getNumJoints(self.baxterId)
        self.motorIndices = []
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.baxterId, i)
            if jointInfo[3] > -1:
                self.motorIndices.append(i)

        self.arm_indices = [34, 35, 36, 37, 38, 40, 41, 49, 51] if arm == "left" else [
            12, 13, 14, 15, 16, 18, 19, 27, 29]
        self.ll, self.ul, self.jr = self.getJointRanges(self.baxterId)
        self.rp = (0.0, 0.8462368022380607, -1.630866219617225,
                   -0.08415513820451972, 2.0843043904431457,
                   0.002042621644835222, 1.1254427955750927,
                   -0.1461959296684458, 0.0, 0.0,
                   -0.8462368022380607, -1.630866219617225,
                   -0.08415513820451972, 2.0843043904431457,
                   -0.002042621644835222, 1.1254427955750927,
                   -0.1461959296684458, 0.0, 0.0)
        self.ini_pos = [0.5, 0.2, 0.2] if arm == "left" else [0.5, -0.2, 0.2]
        self.reset(self.ini_pos)
        self.prevPose = [0, 0, 0]
        self.prevPose1 = [0, 0, 0]
        self.hasPrevPose = 0
        self.trailDuration = 0  # 设置为0，线条不会自动消失

    def getJointRanges(self, bodyId, includeFixed=False):
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
        p.resetBasePositionAndOrientation(self.baxterId, [0.000, 0.000000, 0.00000],
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        orn = p.getQuaternionFromEuler([0, -math.pi, math.pi / 2])
        pose = pose_gripper[0:3] if len(pose_gripper) > 3 else pose_gripper
        jointPoses = p.calculateInverseKinematics(
            self.baxterId, self.endEffector_id,
            pose, orn,
            lowerLimits=self.ll, upperLimits=self.ul,
            jointRanges=self.jr, restPoses=self.rp,
            solver=p.IK_SDLS, maxNumIterations=200, residualThreshold=1e-5
        )
        '''for i in range(len(self.motorIndices)):
            p.resetJointState(self.baxterId, self.motorIndices[i], jointPoses[i])'''
        # 只重置当前臂的关节
        for i in self.arm_indices:
            if i in self.motorIndices:
                idx = self.motorIndices.index(i)
                p.resetJointState(self.baxterId, i, jointPoses[idx])
        p.resetJointState(self.baxterId, self.finger_a_id, self.gripper_close)
        p.resetJointState(self.baxterId, self.finger_b_id, self.gripper_close)


    def accurateCalculateInverseKinematics(self, targetPos, threshold, maxIter):
        closeEnough = False
        iter = 0
        dist2 = 1e30
        while (not closeEnough and iter < maxIter):
            jointPoses = p.calculateInverseKinematics(self.baxterId, self.endEffector_id, targetPos,
                                                      lowerLimits=self.ll, upperLimits=self.ul,
                                                      jointRanges=self.jr, restPoses=self.rp)
            # 只重置当前臂的关节
            for i in self.arm_indices:
                if i in self.motorIndices:
                    idx = self.motorIndices.index(i)
                    p.resetJointState(self.baxterId, i, jointPoses[idx])

            ls = p.getLinkState(self.baxterId, self.endEffector_id)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1
        return jointPoses

    def osc(self, motorCommands):
        d_position = motorCommands
        if self.arm == "left":
            orientation = p.getQuaternionFromEuler([0, -math.pi, math.pi / 2])
        else:
            orientation = p.getQuaternionFromEuler([0, -math.pi, math.pi / 2])

        state = p.getLinkState(self.baxterId, self.endEffector_id)
        end_pose = np.array(state[4])
        new_end_pose = end_pose + d_position
        new_end_pose = np.clip(new_end_pose, self.os_min, self.os_max)

        threshold = 0.001
        maxIter = 100
        jointPoses = self.accurateCalculateInverseKinematics(new_end_pose, threshold, maxIter)

        if self.arm == "left":
            joint_indices = self.motorIndices[-9:]
            target_positions = jointPoses[-9:]
        else:
            joint_indices = self.motorIndices[1:10]
            target_positions = jointPoses[1:10]

            # 使用当前臂的关节索引
            joint_indices = self.arm_indices
            target_positions = []

            # 提取当前臂的目标位置
            for i in joint_indices:
                if i in self.motorIndices:
                    idx = self.motorIndices.index(i)
                    target_positions.append(jointPoses[idx])

            # 控制当前臂的关节
            for i, joint_index in enumerate(joint_indices):
                p.setJointMotorControl2(
                    bodyIndex=self.baxterId,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_positions[i],
                    targetVelocity=0,
                    force=self.maxForce,
                    positionGain=1,
                    velocityGain=0.1
                )

        ls = p.getLinkState(self.baxterId, self.endEffector_id)
        if self.hasPrevPose:
            p.addUserDebugLine(self.prevPose, new_end_pose, [0, 1, 0], 1, self.trailDuration)
            p.addUserDebugLine(self.prevPose1, ls[4], [1, 0, 0], 1, self.trailDuration)
        self.prevPose = new_end_pose
        self.prevPose1 = ls[4]
        self.hasPrevPose = 1

    def gripper_control(self, gripper_state):
        gripper_state = (gripper_state + 1) * 0.5
        gripper_state_a = np.clip(gripper_state * self.gripper_open, self.gripper_close,
                                  self.gripper_open)
        gripper_state_b = np.clip(-gripper_state * self.gripper_open, -self.gripper_open,
                                  self.gripper_close)
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
        pos_a = p.getJointState(self.baxterId, self.finger_a_id)[0]
        return (pos_a / self.gripper_open - 0.5) * 2

    def gripper_close_open(self, gripper_command):
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

    left_arm = BaxterArm(baxter_id, arm="left")
    right_arm = BaxterArm(baxter_id, arm="right")
    # 分别重置双臂到初始位置
    left_arm.reset(left_arm.ini_pos)
    right_arm.reset(right_arm.ini_pos)

    t_left = 0  # 左臂运动参数
    t_right = 0  # 右臂运动参数

    error_sum_left = 0
    error_sum_right = 0
    steps = 1000

    # 左臂圆形轨迹参数
    center_left = left_arm.ini_pos
    radius_left = 0.1

    # 右臂圆形轨迹参数 - 使用右臂的初始位置作为中心
    center_right = right_arm.ini_pos
    radius_right = 0.1

    # 双臂运动方向可以不同（例如一个顺时针一个逆时针）
    direction_left = 1  # 1表示逆时针，-1表示顺时针
    direction_right = -1  # 与左臂相反方向

    for i in range(steps):
        # 计算左臂目标位置
        pos_left = [
            center_left[0],
            center_left[1] + radius_left * math.cos(t_left),
            center_left[2] + radius_left * math.sin(t_left)
        ]

        # 计算右臂目标位置
        pos_right = [
            center_right[0],
            center_right[1] + radius_right * math.cos(t_right),
            center_right[2] + radius_right * math.sin(t_right)
        ]

        # 获取当前末端位置
        left_ee_pos = np.array(p.getLinkState(left_arm.baxterId, left_arm.endEffector_id)[4])
        right_ee_pos = np.array(p.getLinkState(right_arm.baxterId, right_arm.endEffector_id)[4])

        # 计算位移指令
        d_position_left = np.array(pos_left) - left_ee_pos
        d_position_right = np.array(pos_right) - right_ee_pos

        # 控制双臂运动
        left_arm.osc(d_position_left)
        right_arm.osc(d_position_right)

        p.stepSimulation()
        time.sleep(0.01)

        # 计算左臂跟踪误差
        actual_pos_left = p.getLinkState(left_arm.baxterId, left_arm.endEffector_id)[4]
        error_left = np.linalg.norm(np.array(actual_pos_left) - np.array(pos_left))
        error_sum_left += error_left

        # 计算右臂跟踪误差
        actual_pos_right = p.getLinkState(right_arm.baxterId, right_arm.endEffector_id)[4]
        error_right = np.linalg.norm(np.array(actual_pos_right) - np.array(pos_right))
        error_sum_right += error_right

        # 更新运动参数
        t_left += 0.01 * direction_left
        t_right += 0.01 * direction_right

    # 计算平均误差
    average_error_left = error_sum_left / steps
    average_error_right = error_sum_right / steps
    print(f"左臂平均跟踪误差: {average_error_left}")
    print(f"右臂平均跟踪误差: {average_error_right}")

    # 打开夹爪
    left_arm.gripper_control(1.0)
    right_arm.gripper_control(1.0)