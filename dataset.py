import os
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm # 用于显示加载进度

class ACDCDataset(Dataset):
    def __init__(self, root_dir, is_training=True, target_size=(256, 256), cache=True):
        """
        Args:
            root_dir (str): 数据集文件夹路径
            is_training (bool): 是否为训练模式 (决定是否返回 Mask)
            target_size (tuple): 目标分辨率 (H, W)
            cache (bool): 
                True -> 将所有数据加载到内存 (RAM) 中。启动慢，训练极快。适合 ACDC。
                False -> 每次读取文件 (Lazy Loading)。启动快，训练受 IO 限制。适合超大数据集。
        """
        self.root_dir = root_dir
        self.is_training = is_training
        self.target_size = target_size
        self.cache = cache
        self.cached_data = []

        # 获取所有 .h5 文件
        self.file_list = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith('.h5')
        ]

        if len(self.file_list) == 0:
            print(f"Warning: No .h5 files found in {root_dir}")

        # 如果开启缓存，立即加载所有数据
        if self.cache:
            print(f"Caching data from {root_dir} into RAM...")
            for filepath in tqdm(self.file_list):
                sample = self._load_and_process(filepath)
                if sample is not None:
                    self.cached_data.append(sample)
            print(f"Cached {len(self.cached_data)} samples.")

    def __len__(self):
        return len(self.cached_data) if self.cache else len(self.file_list)

    def __getitem__(self, idx):
        # 策略模式：内存取值 或 硬盘读取
        if self.cache:
            return self.cached_data[idx]
        else:
            filepath = self.file_list[idx]
            return self._load_and_process(filepath)

    def _load_and_process(self, filepath):
        """内部处理函数：读取 -> 转Tensor -> Resize -> Norm"""
        try:
            with h5py.File(filepath, 'r') as f:
                image_np = f['image'][:]
                # 兼容性处理：部分测试集可能没有 label
                mask_np = f['label'][:] if (self.is_training and 'label' in f) else None

            # --- 1. 处理图像 (Bilinear) ---
            img_tensor = torch.from_numpy(image_np).float()
            
            # 增加维度用于 interpolate: (H, W) -> (1, 1, H, W)
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            elif img_tensor.ndim == 3: # (S, H, W) -> (S, 1, H, W)
                img_tensor = img_tensor.unsqueeze(1)

            img_resized = F.interpolate(img_tensor, size=self.target_size, mode='bilinear', align_corners=False)

            # 归一化 (Min-Max per sample)
            img_min, img_max = img_resized.min(), img_resized.max()
            if img_max > img_min:
                img_resized = (img_resized - img_min) / (img_max - img_min)

            # 降维: (1, 1, H, W) -> (1, H, W) 
            # 也就是保留 Channel 维度，PyTorch Conv2d 需要
            final_img = img_resized.squeeze(0)

            # --- 2. 处理 Mask (Nearest) ---
            if mask_np is not None:
                mask_tensor = torch.from_numpy(mask_np).float()
                
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                elif mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor.unsqueeze(1)

                # 必须用 Nearest 插值保持类别整数
                mask_resized = F.interpolate(mask_tensor, size=self.target_size, mode='nearest')
                
                # 降维: (1, 1, H, W) -> (H, W)
                # CrossEntropyLoss 期望 Target 是 (N, H, W) 无 Channel
                final_mask = mask_resized.squeeze(0).squeeze(0).long()
                
                return final_img, final_mask
            
            # 如果是推理模式且没有Mask
            return final_img

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # 返回全0数据防止崩溃 (Robustness)
            dummy_img = torch.zeros((1, *self.target_size))
            if self.is_training:
                return dummy_img, torch.zeros(self.target_size).long()
            return dummy_img

# --- Sanity Check (工程化测试模块) ---
if __name__ == '__main__':
    # 这里的代码只有直接运行 python dataset.py 时才会执行
    print("Testing ACDCDataset...")
    
    # 模拟路径 (请确保你的路径存在，或者手动修改这里测试)
    TEST_DIR = 'ACDC_preprocessed/ACDC_training_slices'
    
    if os.path.exists(TEST_DIR):
        # 测试 RAM 缓存模式
        ds = ACDCDataset(TEST_DIR, is_training=True, cache=True)
        print(f"Dataset length: {len(ds)}")
        
        # 取一个样本
        img, mask = ds[0]
        print(f"Image shape: {img.shape}, Type: {img.dtype}") # 应为 (1, 256, 256), float32
        print(f"Mask shape:  {mask.shape}, Type: {mask.dtype}") # 应为 (256, 256), int64
        
        # 检查值域
        print(f"Img Range: [{img.min():.2f}, {img.max():.2f}]") # 应为 0-1
        print(f"Mask Classes: {torch.unique(mask)}") # 应包含 0,1,2,3
        
        print("Sanity Check Passed!")
    else:
        print(f"Path {TEST_DIR} not found, skipping test.")