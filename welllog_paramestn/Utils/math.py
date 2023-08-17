"""
一些数学处理
"""
import torch
from scipy.ndimage import median_filter


def median_filter_in_torch(labels: torch.Tensor, window_size: int):
    device = labels.device
    labels_np = labels.cpu().numpy()
    filtered_labels_np = median_filter(labels_np, size=window_size)  # 这里的size是滤波器的大小，可以根据需要进行调整
    filtered_labels = torch.from_numpy(filtered_labels_np).to(device)
    return filtered_labels
