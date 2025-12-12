# Cardiac MRI Segmentation using U-Net (PyTorch)

## Project Overview
This project implements a 2D U-Net architecture for automatic segmentation of cardiac structures from Magnetic Resonance Imaging (MRI) data. It is designed for the ACDC (Automated Cardiac Diagnosis Challenge) dataset.

The model segments the image into 4 classes:
0. Background
1. Right Ventricle (RV)
2. Myocardium (MYO)
3. Left Ventricle (LV)

## Key Features
- **Modular Design**: Separated configuration, model definition, and data loading.
- **Efficient Training**: Supports Automatic Mixed Precision (AMP) and RAM caching for speed.
- **Robustness**: Implements Early Stopping and Learning Rate Scheduling.
- **Evaluation**: Includes full 3D volume inference and per-case Dice score calculation.

---

## Directory Structure

.
├── ACDC_preprocessed/          # Data Directory
│   ├── ACDC_training_slices/   # .h5 files (2D slices for training)
│   └── ACDC_testing_volumes/   # .h5 files (3D volumes for testing)
├── config.py                   # Configuration file (Hyperparameters & Paths)
├── dataset.py                  # PyTorch Dataset class (Data I/O & Preprocessing)
├── model_unet.py               # U-Net Architecture definition
├── train.py                    # Training script
├── test.py                     # Evaluation script (Metrics & CSV report)
├── inference.py                # Visualization script
├── requirements.txt            # Python dependencies
└── README.md                   # This file

---

## Installation

1. **Prerequisites**:
   - Python 3.8+
   - CUDA-enabled GPU (Recommended)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

