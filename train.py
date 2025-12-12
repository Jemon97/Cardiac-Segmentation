import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- 导入自定义模块 ---
from config import Config
from model_unet import UNet
from dataset import ACDCDataset

def calculate_dice(preds, targets, num_classes):
    """
    计算验证集的平均 Dice Score (忽略背景类 0)
    preds: (Batch, H, W)
    targets: (Batch, H, W)
    """
    dice_scores = []
    # 从 1 开始遍历 (RV, MYO, LV)
    for i in range(1, num_classes):
        pred_i = (preds == i).float()
        target_i = (targets == i).float()
        
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        
        if union == 0:
            # 如果预测和真值都为空，视为完美匹配
            dice_scores.append(1.0)
        else:
            dice = (2. * intersection) / (union + 1e-6)
            dice_scores.append(dice.item())
            
    return np.mean(dice_scores)

def main():
    Config.print_config()
    
    # --- 1. 数据准备 ---
    if not os.path.exists(Config.TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {Config.TRAIN_DIR}")

    full_dataset = ACDCDataset(
        Config.TRAIN_DIR, 
        is_training=True, 
        target_size=Config.TARGET_SIZE, 
        cache=Config.CACHE_DATA
    )
    
    # 划分 80% 训练, 20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    print(f"Data Loaded: {len(train_ds)} Train samples | {len(val_ds)} Val samples")

    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )

    # --- 2. 模型与优化器 ---
    model = UNet(n_channels=1, n_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器: 当 Dice 指标不再上升时，减小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 混合精度 Scaler (PyTorch 2.4+ 写法)
    scaler = torch.amp.GradScaler('cuda') if Config.USE_AMP else None

    # --- 3. 训练循环 ---
    best_dice = 0.0
    early_stop_counter = 0
    
    print(">>> Start Training <<<")
    
    for epoch in range(Config.EPOCHS):
        # === Training Phase ===
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for imgs, masks in loop:
            imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度前向传播
            if Config.USE_AMP:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通精度前向传播
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # === Validation Phase ===
        model.eval()
        val_loss = 0
        val_dice_list = []
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
                
                # 推理时也可以用 autocast 稍微加速
                with torch.amp.autocast('cuda') if Config.USE_AMP else torch.no_grad():
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # 计算 Dice
                preds = torch.argmax(outputs, dim=1) # (B, C, H, W) -> (B, H, W)
                batch_dice = calculate_dice(preds, masks, Config.NUM_CLASSES)
                val_dice_list.append(batch_dice)
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = np.mean(val_dice_list)
        
        # === Logging & Saving ===
        print(f"\nSummary Epoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Dice:   {avg_val_dice:.4f} (Best: {best_dice:.4f})")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.2e}")
        
        # 更新 Scheduler
        scheduler.step(avg_val_dice)
        
        # 保存最佳模型 & 早停逻辑
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f">>> Model Saved to {Config.MODEL_SAVE_NAME} <<<")
        else:
            early_stop_counter += 1
            print(f"Early Stopping Counter: {early_stop_counter}/{Config.PATIENCE}")
            
            if early_stop_counter >= Config.PATIENCE:
                print("\n!!! Early Stopping Triggered. Training Finished. !!!")
                break
        
        print("-" * 40)

if __name__ == '__main__':
    main()