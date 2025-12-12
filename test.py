import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd 
import warnings
import cv2 

# --- import modules ---
from config import Config
from model_unet import UNet

# Ignore some resizing warnings
warnings.filterwarnings("ignore")

def load_model():
    """Load the trained model"""
    print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
    model = UNet(n_channels=1, n_classes=Config.NUM_CLASSES)
    
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file not found! Please train first.")
        
    state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()
    return model

def preprocess_slice(slice_np):
    """Preprocess a single slice (consistent with inference.py)"""
    tensor = torch.from_numpy(slice_np).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    # Resize
    resized = F.interpolate(tensor, size=Config.TARGET_SIZE, mode='bilinear', align_corners=False)
    
    # Norm
    img_min, img_max = resized.min(), resized.max()
    if img_max > img_min:
        resized = (resized - img_min) / (img_max - img_min)
        
    return resized.to(Config.DEVICE)

def calculate_dice_per_case(pred_vol, gt_vol, num_classes):
    """
    Calculate the 3D Dice Score for a single case
    Input: 
        pred_vol: (S, H, W) integers
        gt_vol:   (S, H, W) integers
    Output: 
        list of floats [Dice_RV, Dice_MYO, Dice_LV]
    """
    dice_scores = []
    # Ignore background (class 0), start from 1
    for i in range(1, num_classes):
        pred_i = (pred_vol == i)
        gt_i = (gt_vol == i)
        
        intersection = np.sum(pred_i & gt_i)
        union = np.sum(pred_i) + np.sum(gt_i)
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / (union + 1e-6)
            
        dice_scores.append(dice)
    return dice_scores

def evaluate():
    # 1. Initialize
    model = load_model()
    test_files = [f for f in os.listdir(Config.TEST_DIR) if f.endswith('.h5')]
    
    if not test_files:
        print(f"No files found in {Config.TEST_DIR}")
        return

    # Store results
    results = []
    print(f"Starting evaluation on {len(test_files)} cases...")

    # 2. Iterate over test set
    for filename in tqdm(test_files, desc="Evaluating"):
        filepath = os.path.join(Config.TEST_DIR, filename)
        
        with h5py.File(filepath, 'r') as f:
            if 'label' not in f:
                print(f"Skipping {filename}: No Ground Truth label found.")
                continue
                
            image = f['image'][:] # (S, H, W)
            label = f['label'][:] # (S, H, W) - Original resolution
            
            num_slices = image.shape[0]
            pred_slices = []

            # --- 3D slice-by-slice inference ---
            with torch.no_grad():
                for i in range(num_slices):
                    # Inference
                    input_tensor = preprocess_slice(image[i])
                    logits = model(input_tensor)
                    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy() # (256, 256)
                    pred_slices.append(pred_mask)
            
            pred_vol = np.array(pred_slices) # (S, 256, 256)
            
            # --- Key step: resolution alignment ---
            # Resize GT to 256x256 to match prediction

            
            label_resized = []
            for i in range(num_slices):
                # Use Nearest interpolation to resize Label
                l_res = cv2.resize(label[i], Config.TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                label_resized.append(l_res)
            label_vol = np.array(label_resized)

            # --- Calculate Dice ---
            # d_list = [RV, MYO, LV]
            d_list = calculate_dice_per_case(pred_vol, label_vol, Config.NUM_CLASSES)
            
            # Record data
            results.append({
                'Patient': filename,
                'RV_Dice': d_list[0],
                'MYO_Dice': d_list[1],
                'LV_Dice': d_list[2],
                'Mean_Dice': np.mean(d_list)
            })

    # 3. Results aggregation
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = 'evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print("\n" + "="*40)
    print("FINAL EVALUATION REPORT")
    print("="*40)
    print(df.describe().loc[['mean', 'std', 'min', 'max']])
    print("-" * 40)
    print(f"Overall Mean Dice: {df['Mean_Dice'].mean():.4f}")
    print(f"Detailed results saved to: {csv_path}")
    print("="*40)

if __name__ == '__main__':
    evaluate()
