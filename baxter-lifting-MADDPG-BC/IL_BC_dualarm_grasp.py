import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from BC_final import DualArmGrasper, BaxterDualArmEnv
import time
from torch.optim.lr_scheduler import StepLR

# 步骤1: 加载专家数据
def load_expert_data():
    existing_data = []
    if os.path.exists('expert_data.pkl'):
        with open('expert_data.pkl', 'rb') as f:
            while True:
                try:
                    existing_data.extend(pickle.load(f))
                except EOFError:
                    break
    states = []
    actions = []
    for episode in existing_data:
        states.extend(episode['states'])
        actions.extend(episode['actions'])
    states = np.array(states)
    actions = np.array(actions)
    return states, actions

# 步骤2: 定义更复杂的神经网络模型
class BCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 添加 Dropout 正则化

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 步骤3: 训练模型
def train_model(model, states, actions, epochs=200, batch_size=32, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # 学习率调度器

    num_samples = states.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            batch_states = torch.FloatTensor(states[batch_indices])
            batch_actions = torch.FloatTensor(actions[batch_indices])

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()

        scheduler.step()  # 更新学习率

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

    return model

# 步骤4: 测试模型
def test_model(model, env):
    grasper = DualArmGrasper(env)
    num_tests = 10
    success_rate = 0

    for i in range(num_tests):
        print(f"\n=== 测试 {i + 1}/{num_tests} ===")
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = model(state_tensor).detach().numpy()[0]
            next_state, _, done, _ = env.step(action)
            state = next_state

        # 简单判断是否抓取成功，这里假设提升物体后即为成功
        block_pos = np.array(env.get_physic_state()[6:9])
        block_top_z = block_pos[2] + grasper.block_dims[2]
        if block_top_z > grasper.lift_height:
            success_rate += 1
        time.sleep(1)  # 观察结果

    print(f"\n测试完成：成功率 {success_rate / num_tests:.2%}")

def main():
    # 创建环境
    env = BaxterDualArmEnv(renders=True, pygame_renders=False)

    # 加载专家数据
    states, actions = load_expert_data()

    # 定义模型
    input_dim = states.shape[1]
    output_dim = actions.shape[1]
    model = BCNetwork(input_dim, output_dim)

    # 训练模型
    model = train_model(model, states, actions)

    # 测试模型
    test_model(model, env)

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()