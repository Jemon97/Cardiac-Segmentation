# Cardiac MRI Segmentation using U-Net (PyTorch)

## Project Overview
This project implements a 2D U-Net architecture for automatic segmentation of cardiac structures from Magnetic Resonance Imaging (MRI) data. It is designed for the **ACDC (Automated Cardiac Diagnosis Challenge)** dataset but can be adapted for other 2D medical imaging segmentation tasks.

The model segments the MRI images into 4 classes:
- **0**: Background
- **1**: Right Ventricle (RV)
- **2**: Myocardium (MYO)
- **3**: Left Ventricle (LV)

## Key Features
- **Modular Architecture**: Clean separation of configuration (`config.py`), model (`model_unet.py`), data (`dataset.py`), and logic.
- **Performance Optimization**: Supports **Automatic Mixed Precision (AMP)** and **RAM Caching** for high-speed training.
- **Robust Training**: Implements **Early Stopping** and **Learning Rate Scheduling** (ReduceLROnPlateau).
- **Comprehensive Evaluation**: Includes 3D volume inference and per-case Dice score calculation.

---

## Directory Structure
```text
My_Project/
├── ACDC_preprocessed/          # Dataset Directory
│   ├── ACDC_training_slices/   # .h5 files (2D slices for training)
│   └── ACDC_testing_volumes/   # .h5 files (3D volumes for testing)
│
├── config.py                   # [Core] Configuration file (Paths & Hyperparameters)
├── model_unet.py               # [Core] U-Net Architecture definition
├── dataset.py                  # [Core] Data loading & Preprocessing
├── train.py                    # Training script
├── test.py                     # Evaluation script (Generates CSV report)
├── inference.py                # Visualization & 3D Inference script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

---
```
## Getting Started

Follow these steps to set up the environment and run the segmentation pipeline.

### 1. Environment Setup
Ensure you have Python 3.8+ and a CUDA-enabled GPU (Recommended).
```bash
pip install -r requirements.txt
```
*Note: It is recommended to install PyTorch strictly according to your CUDA version from pytorch.org.*

### 2. Configuration
The project is controlled by a central configuration file. Open **`config.py`** to adjust hyperparameters. No need to modify the code logic.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `BATCH_SIZE` | `32` | Adjust based on your GPU VRAM (e.g., 32 for 16GB VRAM). |
| `CACHE_DATA` | `True` | Loads all data into RAM. Recommended for ACDC dataset for fastest training. |
| `USE_AMP` | `True` | Enables Automatic Mixed Precision training (Faster & less VRAM). |
| `LEARNING_RATE` | `1e-4` | Initial learning rate for Adam optimizer. |

### 3. Training
Start the training process. The script includes **Early Stopping** and automatically saves the best model weights to `best_unet_acdc.pth` based on validation Dice score.
```bash
python train.py
```
### 4. Evaluation (Test Set)
After training, evaluate the model on the held-out test set. This script calculates the **3D Dice Score** for every patient (volume-wise) and exports a detailed report to `evaluation_results.csv`.
```bash
python test.py
```
### 5. Inference & Visualization
Run the inference script to visualize the segmentation results (Original MRI vs. Prediction vs. Ground Truth) on random samples from the test set.
```
python inference.py
```
---

## Model Architecture Details

The model is a standard **U-Net** (Ronneberger et al., 2015) adapted for 2D cardiac MRI slices.

- **Input**: `256 x 256` Single Channel MRI Slice (Grayscale).
- **Preprocessing**: 
  - Intensity Normalization: Min-Max scaling per slice to `[0, 1]`.
  - Resizing: Bilinear interpolation for images, Nearest-Neighbor for masks.
- **Network Structure**:
  - **Encoder (Contracting Path)**: 4 blocks. Each block consists of two `3x3` Convolutions (with Batch Normalization & ReLU) followed by a `2x2` Max Pooling operation.
  - **Bottleneck**: Two `3x3` Convolutions with 1024 filters.
  - **Decoder (Expansive Path)**: 4 blocks. Each block consists of a `2x2` Transposed Convolution (Upsampling), concatenation with the corresponding skip connection from the encoder, and two `3x3` Convolutions.
- **Output**: `256 x 256` map with 4 channels (one per class), processed via Softmax.

### Training Configuration
- **Loss Function**: Cross Entropy Loss.
- **Optimizer**: Adam (`betas=(0.9, 0.999)`, `eps=1e-8`).
- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=5) monitoring Validation Dice Score.

---

## Author

**Jiameng Diao** PhD Student, Biomedical Engineering  
University of Virginia  
*Research focus: MRI Pulse Sequences & Image Reconstruction*