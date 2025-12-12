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
            root_dir (str): dataset folder path containing multiple .h5 files
            is_training (bool): whether in training mode (determines if Mask is returned)
            target_size (tuple): target resolution (H, W)
            cache (bool): 
                True -> Load all data into memory (RAM). Slow startup, very fast training. Suitable for ACDC.
                False -> Read files on-the-fly (Lazy Loading). Fast startup, training limited by IO. Suitable for very large datasets.
        """
        self.root_dir = root_dir
        self.is_training = is_training
        self.target_size = target_size
        self.cache = cache
        self.cached_data = []

        # get all .h5 files in the directory
        self.file_list = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith('.h5')
        ]

        if len(self.file_list) == 0:
            print(f"Warning: No .h5 files found in {root_dir}")

        # if caching is enabled, load all data into memory
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
        # Cache or disk read
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
                # Compatibility handling: some test sets may not have label
                mask_np = f['label'][:] if (self.is_training and 'label' in f) else None

            # --- 1. Process image (Bilinear) ---
            img_tensor = torch.from_numpy(image_np).float()
            
            # Add dimensions for interpolate: (H, W) -> (1, 1, H, W)
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            elif img_tensor.ndim == 3: # (S, H, W) -> (S, 1, H, W)
                img_tensor = img_tensor.unsqueeze(1)

            img_resized = F.interpolate(img_tensor, size=self.target_size, mode='bilinear', align_corners=False)

            # Normalization (Min-Max per sample)
            img_min, img_max = img_resized.min(), img_resized.max()
            if img_max > img_min:
                img_resized = (img_resized - img_min) / (img_max - img_min)

            # Squeeze dimensions: (1, 1, H, W) -> (1, H, W) 
            # Keep the Channel dimension, required by PyTorch Conv2d
            final_img = img_resized.squeeze(0)

            # --- 2. Process Mask (Nearest) ---
            if mask_np is not None:
                mask_tensor = torch.from_numpy(mask_np).float()
                
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                elif mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor.unsqueeze(1)

                # Must use Nearest interpolation to preserve integer classes
                mask_resized = F.interpolate(mask_tensor, size=self.target_size, mode='nearest')
                
                # Squeeze dimensions: (1, 1, H, W) -> (H, W)
                # CrossEntropyLoss expects Target as (N, H, W) without Channel
                final_mask = mask_resized.squeeze(0).squeeze(0).long()
                
                return final_img, final_mask
            
            # If in inference mode and no Mask is available
            return final_img

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return all-zero data to prevent crashes (Robustness)
            dummy_img = torch.zeros((1, *self.target_size))
            if self.is_training:
                return dummy_img, torch.zeros(self.target_size).long()
            return dummy_img

# --- Sanity Check (Engineering Test Module) ---
if __name__ == '__main__':
    # This code only runs when you execute python dataset.py directly
    print("Testing ACDCDataset...")
    
    # Simulated path (make sure your path exists, or manually modify here for testing)
    TEST_DIR = 'ACDC_preprocessed/ACDC_training_slices'
    
    if os.path.exists(TEST_DIR):
        # Test RAM caching mode
        ds = ACDCDataset(TEST_DIR, is_training=True, cache=True)
        print(f"Dataset length: {len(ds)}")
        
        # Get a sample
        img, mask = ds[0]
        print(f"Image shape: {img.shape}, Type: {img.dtype}") # Should be (1, 256, 256), float32
        print(f"Mask shape:  {mask.shape}, Type: {mask.dtype}") # Should be (256, 256), int64
        
        # Check value range
        print(f"Img Range: [{img.min():.2f}, {img.max():.2f}]") # Should be 0-1
        print(f"Mask Classes: {torch.unique(mask)}") # Should contain 0,1,2,3
        
        print("Sanity Check Passed!")
    else:
        print(f"Path {TEST_DIR} not found, skipping test.")