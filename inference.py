import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2 

# --- import ---
from config import Config
from model_unet import UNet

def load_model():
    """Load the trained model"""
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    
    model = UNet(n_channels=1, n_classes=Config.NUM_CLASSES)
    
    # weights_only=True is the new recommended practice in PyTorch
    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}. Please train first.")
        
    model.to(Config.DEVICE)
    model.eval()
    return model

def preprocess_slice(slice_np):
    """
    Input: (H, W) numpy
    Output: (1, 1, H, W) tensor on device
    """
    # 1.  Tensor
    tensor = torch.from_numpy(slice_np).float()
    
    # 2. add dimensions (H, W) -> (1, 1, H, W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
    # 3. Resize
    resized = F.interpolate(tensor, size=Config.TARGET_SIZE, mode='bilinear', align_corners=False)
    
    # 4. Normalize (0-1)
    img_min, img_max = resized.min(), resized.max()
    if img_max > img_min:
        resized = (resized - img_min) / (img_max - img_min)
        
    return resized.to(Config.DEVICE)

def predict_volume(model, volume_np):
    """
    Predict slice-by-slice for a 3D volume
    Input: (Slices, H, W)
    Output: (Slices, 256, 256)
    """
    num_slices = volume_np.shape[0]
    predictions = []
    
    with torch.no_grad():
        for i in range(num_slices):
            slice_img = volume_np[i]
            input_tensor = preprocess_slice(slice_img)
            
            # inference
            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1) # (1, H, W)
            
            predictions.append(pred_mask.squeeze().cpu().numpy())
            
    return np.array(predictions)

def visualize_prediction(slice_img, pred_mask, gt_mask=None, slice_idx=0):
    """Visualize a single slice prediction"""
    plt.figure(figsize=(12, 4))
    
    # 1. Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(slice_img, cmap='gray')
    plt.title(f"MRI Slice {slice_idx}")
    plt.axis('off')
    
    # 2. Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='viridis', vmin=0, vmax=Config.NUM_CLASSES-1)
    plt.title("Prediction")
    plt.axis('off')
    
    # 3. Ground Truth (if available)
    plt.subplot(1, 3, 3)
    if gt_mask is not None:
        plt.imshow(gt_mask, cmap='viridis', vmin=0, vmax=Config.NUM_CLASSES-1)
        plt.title("Ground Truth")
    else:
        plt.text(0.5, 0.5, "No GT Available", ha='center')
        plt.title("Ground Truth")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 1. Check test data
    if not os.path.exists(Config.TEST_DIR):
        print(f"Error: Test directory {Config.TEST_DIR} not found.")
        exit()
        
    test_files = [f for f in os.listdir(Config.TEST_DIR) if f.endswith('.h5')]
    if not test_files:
        print("No .h5 files found in test folder.")
        exit()

    # 2. Load model
    model = load_model()
    
    # 3. Run single sample test
    sample_file = os.path.join(Config.TEST_DIR, test_files[0]) # Take the first file
    print(f"Processing: {test_files[0]} ...")
    
    with h5py.File(sample_file, 'r') as f:
        # Load data
        vol_img = f['image'][:] # (Slices, H, W)
        vol_gt = f['label'][:] if 'label' in f else None
        
        # 3D Prediction
        vol_pred = predict_volume(model, vol_img)
        
        print(f"Prediction Complete. Output Shape: {vol_pred.shape}")
        
        # Visualize middle slice
        mid_idx = vol_img.shape[0] // 2
        
        
        # For accurate overlay, we simply take the resized version of the original image for display
       

        disp_img = cv2.resize(vol_img[mid_idx], Config.TARGET_SIZE)
        disp_gt = cv2.resize(vol_gt[mid_idx], Config.TARGET_SIZE, interpolation=cv2.INTER_NEAREST) if vol_gt is not None else None


        visualize_prediction(disp_img, vol_pred[mid_idx], disp_gt, slice_idx=mid_idx)