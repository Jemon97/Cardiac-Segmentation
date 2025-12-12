import os
import torch

class Config:
    # --- 1. Path  ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset paths (ensure folder names match your actual setup)
    TRAIN_DIR = os.path.join(BASE_DIR, 'ACDC_preprocessed', 'ACDC_training_slices')
    TEST_DIR =  os.path.join(BASE_DIR, 'ACDC_preprocessed', 'ACDC_testing_volumes')
    # TEST_DIR =  os.path.join(BASE_DIR, 'TrasferLearning', 'Testing Set', 'b002')
    # Model save path
    MODEL_SAVE_NAME = 'best_unet_acdc.pth'
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, MODEL_SAVE_NAME)

    # --- 2. Data Parameters ---
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 4  # 0:BG, 1:RV, 2:MYO, 3:LV
    
    # Cache strategy: ACDC data is small, recommend True (load into memory) for fastest training speed
    CACHE_DATA = True 

    # --- 3. Training Hyperparameters ---
    BATCH_SIZE = 32      # Recommended 32 for 16GB VRAM
    LEARNING_RATE = 1e-4 # Adam default starting point
    EPOCHS = 100         # Maximum epochs (with early stopping)
    WEIGHT_DECAY = 1e-5  # Prevent overfitting
    PATIENCE = 15        # Early stopping patience (epochs)

    # --- 4. Hardware and Acceleration ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Mixed Precision Training (AMP): Significantly reduces memory usage and speeds up training
    USE_AMP = True
    
    # Windows 0，Linux/Server 4
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