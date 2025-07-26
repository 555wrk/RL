'''
双臂协同抓取环境 - 基于单臂抓取环境扩展
主要修改点：
1. 添加右臂控制器支持
2. 扩展观测空间到23维
3. 动作空间扩展到8维
4. 添加协同奖励函数
5. 添加防碰撞约束
'''

import os
import sys
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.chdir(current_dir)
from baxter3 import BaxterArm  # 修改：使用新的BaxterArm类
import random
import pygame
import torch


class BaxterDualArmEnv(gym.Env):  # 修改：类名改为BaxterDualArmEnv
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, urdfRoot=None, actionRepeat=1,
                 isEnableSelfCollision=True, max_episode_steps=100,
                 renders=True, pygame_renders=True,
                 camera_view=False, reward_type="dense"):  # 修改：默认关闭相机视图

        # 初始化参数
        self._timeStep = 1. / 240.
        self.control_time = 1 / 20
        self._urdfRoot = urdfRoot or pybullet_data.getDataPath()
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._camera_view = camera_view
        self._reward_types = reward_type
        self._observation = []
        self.action_dim = 8  # 修改：动作维度扩展到8 (左臂4 + 右臂4)
        self.v_end = 0.05
        self.v_gripper = 1
        self._envStepCounter = 0
        self.max_episode_steps = max_episode_steps
        self._renders = renders
        self._pygame_render = pygame_renders
        self._width = 512
        self._height = 512
        self.terminated = 0
        self._p = p


        # 连接物理服务器
        self._init_physics()

        # 初始化观察空间
        obs = self.reset()
        # 修改：仅使用物理状态空间
        self.observation_space = spaces.Box(low=-1, high=1, shape=obs.shape, dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=float)

    def _init_physics(self):
        """初始化物理连接"""
        print("正在连接物理服务器...")
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, 0.2, np.pi / 4.])
        else:
            p.connect(p.DIRECT)

        # 设置资源搜索路径
        print(f"设置资源搜索路径: {pybullet_data.getDataPath()}")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self._urdfRoot:
            print(f"添加自定义资源路径: {self._urdfRoot}")
            p.setAdditionalSearchPath(self._urdfRoot)

    def reset(self):
        """重置环境"""
        print("\n=== 正在重置双臂环境 ===")
        self.terminated = 0
        self._envStepCounter = 0
        self._grasped_time = 0

        p.resetSimulation()
        #p.setPhysicsEngineParameter(numSolverIterations=150)
        # 提高仿真精度的关键参数设置
        p.setPhysicsEngineParameter(
            fixedTimeStep=1 / 1000,  # 1ms时间步长
            numSubSteps=20,  # 每个物理步细分10个子步
            numSolverIterations=150,  # 提高约束求解精度
            solverResidualThreshold=1e-7,
            contactBreakingThreshold=0.001
        )
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        # 加载基础环境
        try:
            print("正在加载地面...")
            plane_path = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
            print(f"地面路径: {plane_path}")
            if not os.path.exists(plane_path):
                raise FileNotFoundError(f"plane.urdf not found at {plane_path}")
            p.loadURDF(plane_path, [0, 0, -0.9])
            print("地面加载成功")

            # 加载其他模型
            self._load_models()

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            p.disconnect()
            raise

        # 稳定环境
        print("稳定环境中...")
        for _ in range(int(self.control_time / self._timeStep) * 10):
            p.stepSimulation()

        # 获取初始物体高度
        block_pose, _ = p.getBasePositionAndOrientation(self.blockUid)
        self.block_init_z = block_pose[2]
        print(f"物体初始高度: {self.block_init_z:.3f}")

        # 初始化渲染
        if self._pygame_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self._width, self._height))
        obs = self.getObservation()
        #print(f"Reset observation shape: {obs.shape}")  # 添加打印语句
        return self.getObservation()

    def _load_models(self):
        """加载所有URDF模型"""
        print("\n--- 开始加载模型 ---")

        # 1. 加载墙体
        '''try:
            wall_path = os.path.abspath(os.path.join("models", "objects", "block", "black_wall.urdf"))
            print(f"[1/5] 正在加载墙体: {wall_path}")
            print(f"路径存在: {os.path.exists(wall_path)}")

            if not os.path.exists(wall_path):
                raise FileNotFoundError(f"墙体文件不存在: {wall_path}")

            self.wall_id = p.loadURDF(wall_path, [-1, -1, -0.9], useFixedBase=True)
            print("墙体加载成功")

        except Exception as e:
            print(f"警告: 墙体加载失败 - {str(e)}")
            print("尝试加载备用墙体...")
            self.wall_id = p.loadURDF("cube.urdf",
                                      basePosition=[-1, -1, -0.9],
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                      globalScaling=2.0,
                                      useFixedBase=True)
            print("备用立方体墙体已加载")
'''
        # 2. 加载机器人
        try:
            baxter_path = os.path.abspath(os.path.join(
                "../../baxter-lifting-pybullet-position_observeBC/baxter_bullet_env/models", "baxter_description", "urdf", "baxter.urdf"))
            print(f"\n[2/5] 正在加载机器人: {baxter_path}")
            print(f"路径存在: {os.path.exists(baxter_path)}")

            if not os.path.exists(baxter_path):
                raise FileNotFoundError(f"机器人文件不存在: {baxter_path}")

            self.baxterId = p.loadURDF(baxter_path, useFixedBase=True)
            print("机器人加载成功")

        except Exception as e:
            print(f"机器人加载失败: {str(e)}")
            raise

        # 3. 加载桌子
        try:
            table_path = os.path.abspath(os.path.join(
                "../../baxter-lifting-pybullet-position_observeBC/baxter_bullet_env/models", "objects", "table", "table.urdf"))
            print(f"\n[3/5] 正在加载桌子: {table_path}")
            print(f"路径存在: {os.path.exists(table_path)}")

            if not os.path.exists(table_path):
                raise FileNotFoundError(f"桌子文件不存在: {table_path}")

            self.table_id = p.loadURDF(table_path, [0.7, 0, -0.9], [0.0, 0.0, 0, 1])
            print("桌子加载成功")

        except Exception as e:
            print(f"桌子加载失败: {str(e)}")
            raise

        # 4. 初始化机器人控制器（双臂）
        try:
            print("\n[4/5] 初始化双臂控制器...")
            # 修改：创建左右臂控制器
            self.robot_left = BaxterArm(self.baxterId, arm="left")
            self.robot_right = BaxterArm(self.baxterId, arm="right")
            print("双臂控制器初始化完成")

        except Exception as e:
            print(f"机器人控制器初始化失败: {str(e)}")
            raise

        # 5. 加载长方体物体
        try:
            print("\n[5/5] 加载长方体物体...")
            block_path = os.path.abspath(os.path.join(
                "../../baxter-lifting-pybullet-position_observeBC/baxter_bullet_env/models", "objects", "rectangle_block.urdf"))
            print(f"正在加载长方体: {block_path}")
            print(f"路径存在: {os.path.exists(block_path)}")

            if not os.path.exists(block_path):
                # 使用 createMultiBody 动态创建细长长方体
                print("长方体URDF文件不存在，动态创建细长长方体...")

                # 获取桌子位置，计算物体放置位置
                table_pos, table_orn = p.getBasePositionAndOrientation(self.table_id)
                #print(f"桌子放置位置: {table_pos}")
                table_z = table_pos[2]
                # 随机化物体位置
                obj_pose = [
                    table_pos[0] + random.random() * 0.1 - 0.1,  # x方向在桌子中心附近随机偏移
                    table_pos[1] + random.random() * 0.1 - 0.1,  # y方向在桌子中心附近随机偏移
                    table_z + 0.88  # z方向保持在桌面上方
                ]
                #obj_pose = [0.6, 0, table_z + 0.75]  # 物体底部与桌面接触（0.04为物体高度的一半）
                print(f"物体放置位置: {obj_pose}")

                # 创建视觉形状
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[ 0.012,0.35, 0.012],  # 0.036m×0.6x0.036m的长方体
                    rgbaColor = [0, 0, 1, 1]
                )

                # 创建碰撞形状（与视觉形状一致）
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[ 0.012, 0.35,0.012]
                )

                # 创建多体（物体）
                self.blockUid = p.createMultiBody(
                    baseMass=0.3,  # 物体质量
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=obj_pose
                )

                print(f"动态创建的细长长方体成功，尺寸: 0.036m×0.6x0.036m")
                # 【新增】沿y轴创建标记点
                block_width = 0.6  # 根据URDF实际宽度调整
                self._create_block_markers(obj_pose, block_width)
            else:
                # 如果存在URDF文件，继续使用原方法加载
                table_pos, table_orn = p.getBasePositionAndOrientation(self.table_id)
                table_z = table_pos[2]
                obj_pose = [0.5, 0.4, table_z + 0.05]
                self.blockUid = p.loadURDF(block_path, obj_pose, globalScaling=0.3)
                print(f"长方体加载成功，位置: {obj_pose}")

            # 设置物体物理参数（根据需要调整）
            p.changeDynamics(self.blockUid, -1,
                             mass=0.3,  # 降低质量便于抓取
                             lateralFriction=1.5,  # 增加摩擦力防止滑落
                             spinningFriction=0.01,
                             rollingFriction=0.001,
                             restitution=0  # 无弹性
                             )
            p.changeDynamics(self.table_id, -1,
                             lateralFriction=1,
                             restitution=0)

            # 设置夹爪物理参数（保持不变）
            for robot in [self.robot_left, self.robot_right]:
                p.changeDynamics(self.baxterId, robot.finger_tips_a_id,
                                 lateralFriction=1,
                                 spinningFriction=0.001,
                                 rollingFriction=0.0005)
                p.changeDynamics(self.baxterId, robot.finger_tips_b_id,
                                 lateralFriction=1,
                                 spinningFriction=0.001,
                                 rollingFriction=0.0005)
            print("物理参数设置完成")

        except Exception as e:
            print(f"物体加载失败: {str(e)}")
            raise

    def getObservation(self):
        """获取观察值"""
        # 修改：仅返回物理状态
        return self.get_physic_state()

    def get_physic_state(self):
        """获取物理状态 - 扩展为23维"""
        # 左臂状态
        left_ee_pos = list(p.getLinkState(self.baxterId, self.robot_left.endEffector_id)[4])
        left_gripper_state = self.robot_left.get_gripper_state()

        # 右臂状态
        right_ee_pos = list(p.getLinkState(self.baxterId, self.robot_right.endEffector_id)[4])
        right_gripper_state = self.robot_right.get_gripper_state()

        # 物体状态
        block_pos, block_orn = p.getBasePositionAndOrientation(self.blockUid)
        block_euler = p.getEulerFromQuaternion(block_orn)

        # 相对位置
        '''left_to_block = np.array(block_pos) - np.array(left_ee_pos)
        right_to_block = np.array(block_pos) - np.array(right_ee_pos)
        between_arms = np.array(right_ee_pos) - np.array(left_ee_pos)'''
        block_dims = [0.012, 0.35, 0.012]
        left_target = block_pos + np.array([0, block_dims[1] / 1.5, block_dims[2] / 2])
        # 物体右侧目标抓取点
        right_target = block_pos + np.array([0, -block_dims[1] / 1.5, block_dims[2] / 2])
        left_to_block = np.array(left_target) - np.array(left_ee_pos)
        right_to_block = np.array(right_target) - np.array(right_ee_pos)
        between_arms = np.array(right_ee_pos) - np.array(left_ee_pos)
        # 组合所有状态
        return np.concatenate((
            left_ee_pos,             # 左臂末端位置 (3)
            right_ee_pos,            # 右臂末端位置 (3)
            block_pos,               # 物体位置 (3)
            block_euler,             # 物体欧拉角 (3)
            [left_gripper_state],    # 左夹爪状态 (1)
            [right_gripper_state],   # 右夹爪状态 (1)
            left_to_block,           # 左臂到物体向量 (3)
            right_to_block,          # 右臂到物体向量 (3)
            between_arms             # 双臂间向量 (3)
        ), axis=0)  # 总计 3+3+3+3+1+1+3+3+3 = 23维

    def step(self, action):
        """执行动作 - 处理双臂动作"""
        self._envStepCounter += 1

        # 拆分动作
        left_action = action[0:4]
        right_action = action[4:8]

        # 预测新位置
        left_ee_pos = np.array(p.getLinkState(self.baxterId, self.robot_left.endEffector_id)[4])
        right_ee_pos = np.array(p.getLinkState(self.baxterId, self.robot_right.endEffector_id)[4])

        predicted_left_pos = left_ee_pos + left_action[0:3] * self.v_end
        predicted_right_pos = right_ee_pos + right_action[0:3] * self.v_end

        # 检查预测位置是否会碰撞
        predicted_distance = np.linalg.norm(predicted_left_pos - predicted_right_pos)

        if predicted_distance < 0.2:  # 20cm安全阈值
            # 如果预测会碰撞，调整动作
            if predicted_distance < 0.15:  # 紧急规避
                # 创建分离向量
                separation_vector = (predicted_left_pos - predicted_right_pos)
                if np.linalg.norm(separation_vector) < 0.01:
                    separation_vector = np.array([0.1, 0, 0])  # 防止零向量

                # 归一化并放大
                separation_vector = separation_vector / np.linalg.norm(separation_vector) * 0.2

                # 调整动作
                left_action[0:3] += separation_vector
                right_action[0:3] -= separation_vector

        # 应用左臂动作
        d_pose_left = left_action[0:3] * self.v_end
        self.robot_left.osc(d_pose_left)
        d_gripper_left = left_action[3] * self.v_gripper
        self.robot_left.gripper_control(d_gripper_left)

        # 应用右臂动作
        d_pose_right = right_action[0:3] * self.v_end
        self.robot_right.osc(d_pose_right)
        d_gripper_right = right_action[3] * self.v_gripper
        self.robot_right.gripper_control(d_gripper_right)

        # 更新模拟
        for _ in range(int(self.control_time / self._timeStep)):
            p.stepSimulation()

        # 获取结果
        obs = self.getObservation()
        #print(f"Step observation shape: {obs.shape}")  # 添加打印语句
        done = self._envStepCounter >= self.max_episode_steps or self._success()
        reward = self._reward()
        info = {
            "success": "True" if self._success() else "False",
            "collision": self._arm_collision()  # 新增：碰撞信息
        }
        #print(f"自碰撞信息：{self._arm_collision()}")

        return obs, reward, done, info

    def _success(self):
        """任务成功条件：物体抬升超过10cm"""
        block_pose, _ = p.getBasePositionAndOrientation(self.blockUid)
        return block_pose[2] - self.block_init_z > 0.05

    def _reward(self):
        """改进的双臂协同抓取奖励函数"""
        # 从观测状态中提取关键信息
        state = self.get_physic_state()
        left_ee_pos = np.array(state[0:3])  # 左臂末端位置
        right_ee_pos = np.array(state[3:6])  # 右臂末端位置
        block_pos = np.array(state[6:9])  # 物体中心位置
        block_euler = state[9:12]  # 物体欧拉角
        left_gripper = state[12]  # 左夹爪状态
        right_gripper = state[13]  # 右夹爪状态

        # 物体尺寸
        block_dims = [0.012, 0.35, 0.012]
        reward = 0.0  # 初始化为0
        reward -= 0.1
        # ==================== 1. 分阶段奖励设计 ====================
        # 阶段1: 双臂接近物体 (基础奖励)
        # 物体左侧目标抓取点（沿细长方向的两端）
        block_left_target = block_pos + np.array([0, block_dims[1] / 1.5, block_dims[2] / 2])
        # 物体右侧目标抓取点
        block_right_target = block_pos + np.array([0, -block_dims[1] / 1.5, block_dims[2] / 2])

        # ==================== 2. 改进接近奖励 ====================
        left_to_left_dist = np.linalg.norm(left_ee_pos - block_left_target)
        right_to_right_dist = np.linalg.norm(right_ee_pos - block_right_target)

        dist_reward = 2.0 - (np.tanh(5.0 * left_to_left_dist) + np.tanh(5.0 * right_to_right_dist))
        reward += dist_reward
        # 优化1：新增双臂同步奖励
        height_diff_arms = abs(left_ee_pos[2] - right_ee_pos[2])
        sync_reward = 1.0 if height_diff_arms < 0.05 else 0.0
        reward += sync_reward
        # 2. 抓取状态奖励 (关键改进)
        if self._sigal_grasped():
            reward += 10.0  # 给予显著的抓取奖励

        if self._grasped():
            reward += 25.0  # 给予显著的抓取奖励

            # 3. 抬升奖励 (只在抓取后计算)
            height_diff = block_pos[2] - self.block_init_z
            # 分阶段高度奖励 (类似单臂设计)
            if height_diff > 0.02:
                reward += 1.0
            if height_diff > 0.04:
                reward += 1.0
            if height_diff > 0.06:
                reward += 1.0
            if height_diff > 0.08:
                reward += 1.0


        # 5. 成功奖励 (要求同时满足抓取和抬升条件)
        if self._success():
            reward += 50.0

        # 6. 碰撞惩罚
        if self._arm_collision():
            reward -= 1.0

        # 调试信息（可选）
        if self._envStepCounter % 10 == 0:
            grasped = self._grasped()
            singal_grasped = self._grasped()
            success = self._success()
            height_diff = block_pos[2] - self.block_init_z if grasped else 0
            collision = self._arm_collision()
            print(f"[Step {self._envStepCounter}] Reward: {reward:.2f} | 协同: {sync_reward:.2f} "
                  f"接近: {dist_reward:.2f} |接触: {singal_grasped} | 抓取: {grasped} | "
                  f"抬升: {height_diff:.3f} | 成功: {success} | "
                  f"碰撞: {collision}")

        return reward

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

        return left_grasped and right_grasped

    def _sigal_grasped(self):
        # 检测左臂单个尖端接触
        left_contact_a = p.getContactPoints(self.baxterId, self.blockUid, self.robot_left.finger_tips_a_id)
        left_contact_b = p.getContactPoints(self.baxterId, self.blockUid, self.robot_left.finger_tips_b_id)
        left_single_tip_contact = (left_contact_a != ()) or (left_contact_b != ())  # 任意一个尖端接触

        # 检测右臂单个尖端接触
        right_contact_a = p.getContactPoints(self.baxterId, self.blockUid, self.robot_right.finger_tips_a_id)
        right_contact_b = p.getContactPoints(self.baxterId, self.blockUid, self.robot_right.finger_tips_b_id)
        right_single_tip_contact = (right_contact_a != ()) or (right_contact_b != ())  # 任意一个尖端接触
        return left_single_tip_contact or right_single_tip_contact

    # 添加辅助方法获取接触状态
    def _get_left_contacts(self):
        contacts = []
        for link_id in [self.robot_left.finger_tips_a_id, self.robot_left.finger_tips_b_id]:
            link_contacts = p.getContactPoints(
                bodyA=self.baxterId,
                bodyB=self.blockUid,
                linkIndexA=link_id
            )
            if link_contacts:
                contacts.extend(link_contacts)
        return contacts

    def _get_right_contacts(self):
        contacts = []
        for link_id in [self.robot_right.finger_tips_a_id, self.robot_right.finger_tips_b_id]:
            link_contacts = p.getContactPoints(
                bodyA=self.baxterId,
                bodyB=self.blockUid,
                linkIndexA=link_id
            )
            if link_contacts:
                contacts.extend(link_contacts)
        return contacts

    def _valid_grasp_points(self, left_contacts, right_contacts, block_pos):
        """
        验证接触点位置是否合理
        - 左臂接触点应在物体左侧（y > block_pos[1]）
        - 右臂接触点应在物体右侧（y < block_pos[1]）
        """
        # 检查左臂接触点
        left_valid = False
        for contact in left_contacts:
            contact_pos = contact[5]  # 接触点位置
            if contact_pos[1] > block_pos[1]:  # y坐标大于物体中心
                left_valid = True
                break

        # 检查右臂接触点
        right_valid = False
        for contact in right_contacts:
            contact_pos = contact[5]  # 接触点位置
            if contact_pos[1] < block_pos[1]:  # y坐标小于物体中心
                right_valid = True
                break

        return left_valid and right_valid

    def _create_block_markers(self, block_pos, block_width):
        """
        在物体两侧（沿y轴方向）创建红色标记点
        :param block_pos: 物体中心位置 [x, y, z]
        :param block_width: 物体沿y轴的宽度（0.4m）
        """
        block_dims = [0.012, 0.35, 0.012]
        # 左侧标记点（y轴负方向）
        #left_marker_pos =  block_pos + np.array([0, block_dims[1] / 1.5, block_dims[2] / 2])
        #print(f"左侧目标位置{left_marker_pos}")
        #block_dims = [0.036, 0.6, 0.036]  # 完整尺寸 = 半尺寸 × 2
        #left_marker_pos = block_pos + np.array([0, block_dims[1] / 3, block_dims[2] / 2])
        #print(f"左侧标记点：{left_marker_pos}")
        # 右侧标记点（y轴正方向）
        #right_marker_pos =  block_pos + np.array([0, -block_dims[1] / 1.5, block_dims[2] / 2])
        #print(f"右侧标记点：{right_marker_pos}")
        # 创建红色小球视觉形状
        '''marker_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,  # 增大半径更易观察
            rgbaColor=[1, 0, 0, 1]
        )

        # 创建左侧标记点
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_visual_shape,
            basePosition=left_marker_pos.tolist()
        )
        # 创建右侧标记点
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_visual_shape,
            basePosition=right_marker_pos.tolist()
        )'''

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

        return left_grasped and right_grasped

    def _arm_collision(self):
        """
        基于距离的双臂碰撞检测
        不依赖物理引擎的接触点检测
        """
        # 1. 定义安全距离阈值
        min_safe_distance = 0.15  # 15cm

        # 2. 检测末端执行器距离
        left_ee_pos = np.array(p.getLinkState(self.baxterId, self.robot_left.endEffector_id)[4])
        right_ee_pos = np.array(p.getLinkState(self.baxterId, self.robot_right.endEffector_id)[4])
        ee_distance = np.linalg.norm(left_ee_pos - right_ee_pos)

        # 3. 检测关键关节距离
        min_distance = float('inf')

        # 左臂关键关节
        left_joints = [34, 35, 36, 37, 38, 40, 41, 49, 51]
        # 右臂关键关节
        right_joints = [12, 13, 14, 15, 16, 18, 19, 27, 29]

        for left_link in left_joints:
            left_link_pos = np.array(p.getLinkState(self.baxterId, left_link)[4])

            for right_link in right_joints:
                right_link_pos = np.array(p.getLinkState(self.baxterId, right_link)[4])
                distance = np.linalg.norm(left_link_pos - right_link_pos)
                if distance < min_distance:
                    min_distance = distance

        # 4. 判断是否碰撞
        return ee_distance < min_safe_distance or min_distance < min_safe_distance

    def render(self, mode='human', close=False):
        if self._pygame_render:
            if mode != "rgb_array":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
            # 添加顶视摄像头
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0.5, 0, 1.2],
                cameraTargetPosition=[0.5, 0, 0],
                cameraUpVector=[0, 1, 0]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            _, _, img, _, _ = p.getCameraImage(
                width=self._width, height=self._height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            # 处理图像用于显示
            img = np.array(img)[:, :, :3]  # 取RGB通道
            surf = pygame.surfarray.make_surface(img.transpose((1, 0, 2)))
            self.screen.blit(surf, (0, 0))
            pygame.display.update()

    def __del__(self):
        if p.isConnected():
            p.disconnect()
            print("物理服务器已断开")

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return [seed]

if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(project_root, "baxter_bullet_env", "models")

    print(f"项目根目录: {project_root}")
    print(f"模型目录: {models_dir}")
    print(f"目录存在: {os.path.exists(models_dir)}")

    env = BaxterDualArmEnv(
        renders=True,
        urdfRoot=models_dir,
        pygame_renders=False
    )

    try:
        for i in range(100):
            action = np.random.uniform(-1, 1, size=8)  # 8维随机动作
            obs, reward, done, _ = env.step(action)
            print(f"Step {i}, Reward: {reward:.2f}")
            env.render()
            if done:
                env.reset()
    except KeyboardInterrupt:
        print("手动终止运行")
    finally:
        del env