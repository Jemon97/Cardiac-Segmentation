import os
import torch

class Config:
    # --- 1. 路径设置 (自动获取绝对路径) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据集路径 (请确保文件夹名称与你的实际情况一致)
    TRAIN_DIR = os.path.join(BASE_DIR, 'ACDC_preprocessed', 'ACDC_training_slices')
    TEST_DIR =  os.path.join(BASE_DIR, 'ACDC_preprocessed', 'ACDC_testing_volumes')
    # TEST_DIR =  os.path.join(BASE_DIR, 'TrasferLearning', 'Testing Set', 'b002')
    # 模型保存路径
    MODEL_SAVE_NAME = 'best_unet_acdc.pth'
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, MODEL_SAVE_NAME)

    # --- 2. 数据参数 ---
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 4  # 0:BG, 1:RV, 2:MYO, 3:LV
    
    # 缓存策略: ACDC数据较小，建议 True (载入内存) 以获得最快训练速度
    CACHE_DATA = True 

    # --- 3. 训练超参数 ---
    BATCH_SIZE = 32      # 16GB 显存推荐 32
    LEARNING_RATE = 1e-4 # Adam 默认起点
    EPOCHS = 100         # 最大轮数 (配合早停)
    WEIGHT_DECAY = 1e-5  # 防止过拟合
    PATIENCE = 15        # 早停容忍度 (轮)

    # --- 4. 硬件与加速 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 混合精度训练 (AMP): 显著减少显存占用并加速
    USE_AMP = True
    
    # Windows下为了稳定性建议设为0，Linux/Server建议设为4
    NUM_WORKERS = 0 if os.name == 'nt' else 4

    @classmethod
    def print_config(cls):
        print("\n" + "="*30)
        print(f"CONFIGURATION ({cls.DEVICE})")
        print("="*30)
        print(f"Batch Size:   {cls.BATCH_SIZE}")
        print(f"Learning Rate:{cls.LEARNING_RATE}")
        print(f"Image Size:   {cls.TARGET_SIZE}")
        print(f"Cache Data:   {cls.CACHE_DATA}")
        print(f"AMP Enabled:  {cls.USE_AMP}")
        print(f"Model Path:   {cls.MODEL_SAVE_PATH}")
        print("="*30 + "\n")