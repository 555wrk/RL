import torch
print(torch.cuda.is_available())  # 应输出 True 表示支持 CUDA
print(torch.version.cuda)         # 输出 PyTorch 关联的 CUDA 版本