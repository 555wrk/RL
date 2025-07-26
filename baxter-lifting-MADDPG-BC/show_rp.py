import pybullet as p
import pybullet_data
import time
import math

# 连接物理引擎并设置环境
p.connect(p.GUI)
p.setRealTimeSimulation(1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

# 加载地面和Baxter模型
p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
baxter_id = p.loadURDF("./baxter_bullet_env/models/baxter_description/urdf/baxter.urdf", useFixedBase=True)

# 定义预设的关节角度参数
'''rp_angles = (
    0.0, 0.8462368022380607, -1.630866219617225,
    -0.08415513820451972, 2.0843043904431457,
    0.002042621644835222, 1.1254427955750927,
    -0.1461959296684458, 0.0, 0.0,
    -0.8462368022380607, -1.630866219617225,
    -0.08415513820451972, 2.0843043904431457,
    -0.002042621644835222, 1.1254427955750927,
    -0.1461959296684458, 0.0, 0.0
)'''
rp_angles = (
    0.0, 0, 0,
    0, 0,
    0, 0,
    0, 0.0, 0.0,
    -0.8462368022380607, -1.630866219617225,
    -0.08415513820451972, 2.0843043904431457,
    -0.002042621644835222, 1.1254427955750927,
    -0.1461959296684458, 0.0, 0.0
)
# 获取所有可动关节索引
motor_indices = []
for i in range(p.getNumJoints(baxter_id)):
    joint_info = p.getJointInfo(baxter_id, i)
    if joint_info[3] > -1:  # 筛选出可控制的关节
        motor_indices.append(i)

# 根据文档定义关节角度限制（弧度）
joint_limits = [
    # 左臂关节
    (-2.147, 1.047),     # S1: -123°~+60°
    (-0.05, 2.618),      # E1: -2.864°~+150°
    (-1.5707, 2.094),    # W1: -90°~+120°
    (-1.7016, 1.7016),   # S0: -97.494°~+97.494°
    (-3.0541, 3.0541),   # E0: -174.987°~+174.987°
    (-3.059, 3.059),     # W0: ±175.25°
    (-3.059, 3.059),     # W2: ±175.25°
    # 右臂关节
    (-2.147, 1.047),     # S1
    (-0.05, 2.618),      # E1
    (-1.5707, 2.094),    # W1
    (-1.7016, 1.7016),   # S0
    (-3.0541, 3.0541),   # E0
    (-3.059, 3.059),     # W0
    (-3.059, 3.059)      # W2
]

# 补充剩余关节限制
while len(joint_limits) < len(motor_indices):
    joint_limits.append((-math.pi/2, math.pi/2))

# 应用预设角度（确保在限制范围内）
for i in range(len(motor_indices)):
    if i < len(rp_angles):
        clamped_angle = max(joint_limits[i][0], min(joint_limits[i][1], rp_angles[i]))
        p.resetJointState(baxter_id, motor_indices[i], clamped_angle)

# 添加用户调试参数（使用默认命名）
position_control_group = []
for i in range(len(motor_indices)):
    min_angle, max_angle = joint_limits[i]
    initial_angle = rp_angles[i] if (i < len(rp_angles)) else (min_angle + max_angle) / 2
    initial_angle = max(min_angle, min(max_angle, initial_angle))
    # 使用默认命名"joint{i}"
    position_control_group.append(p.addUserDebugParameter(
        f'joint{i}', min_angle, max_angle, initial_angle
    ))

print("已加载符合硬件参数的关节控制，可手动调整角度...")

try:
    while True:
        time.sleep(0.01)
        parameter = [p.readUserDebugParameter(param_id) for param_id in position_control_group]

        for i in range(len(motor_indices)):
            joint_info = p.getJointInfo(baxter_id, motor_indices[i])
            p.setJointMotorControl2(
                bodyUniqueId=baxter_id,
                jointIndex=motor_indices[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=parameter[i],
                force=joint_info[10],
                maxVelocity=joint_info[11]
            )
        p.stepSimulation()

except KeyboardInterrupt:
    print("程序已停止。")
finally:
    p.disconnect()