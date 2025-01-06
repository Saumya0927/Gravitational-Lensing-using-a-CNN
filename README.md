# Gravitational-Lensing-using-a-CNN
A deep learning project for detecting gravitational lensing in astronomical images using PyTorch, combining real SLACS survey data with simulated lenses.

## Overview

This project implements a Convolutional Neural Network (CNN) based approach to detect gravitational lensing in astronomical images. It features:
- Custom CNN architecture with residual blocks and attention mechanisms
- Combined real and simulated data pipeline
- Advanced preprocessing for astronomical images
- Comprehensive cross-validation and evaluation framework

## Project Structure

lens_detection_project/
├── data/
│   ├── slacs/          # Real SLACS data
│   ├── simulated/      # Simulated data
│   └── processed/      # Processed dataset
├── notebooks/
│   └── main.ipynb      # Main training notebook
├── scripts/
│   ├── config.py       # Configuration parameters
│   ├── download_data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── dataset.py
│   └── utils.py
└── requirements.txt

## Requirements

```python
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.2
pandas>=1.2.4
astropy>=4.2
galsim>=2.3.5
opencv-python>=4.5.3
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.2
astroquery>=0.4.6
```

## Features
- Advanced CNN Architecture

    - Residual blocks for better gradient flow
    - Squeeze-and-Excitation attention mechanisms
    - Custom data augmentation pipeline
  

- Data Processing

    - FITS file handling
    - Z-scale normalization
    - Custom data collation
    - Comprehensive augmentation strategies


- Training Framework

    - 5-fold cross-validation
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Comprehensive metrics tracking



