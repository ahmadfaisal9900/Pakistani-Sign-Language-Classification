# Pakistani Sign Language Recognition using PkSLMNM Dataset

## Introduction
Pakistani Sign Language (PSL) research has been limited due to the scarcity of annotated datasets and comprehensive studies. This project aims to address this gap by utilizing the PkSLMNM dataset to develop a robust PSL recognition system. Through data augmentation and advanced deep learning techniques, the system extracts temporal and spatial features from sign videos for accurate classification.

The PjSLMM dataset contains videos of 7 basic affective expressions performed by 100 healthy individuals, presented in an easily accessible format of .MP4 that can be used to train and test systems to make robust models for real-time applications using videos.

## Features

### Dataset Preparation
- Data augmentation and frame extraction from PkSLMNM dataset videos.
- Hierarchical directory structure for training, validation, and testing sets.

### Feature Extraction
- **ResNet50** for spatial feature extraction.
- Handles padded sequences for variable-length frames (This script uses 100).
- Includes temporal smoothing for denoising extracted features.

### Classification Model
- **LSTM-based** architecture with hyperparameter optimization using Optuna.
- Supports multi-class PSL gesture classification.

### Performance Metrics
- Training and validation loss/accuracy tracking.
- Test accuracy for final evaluation.

### Hyperparameter Optimization
- Bayesian optimization using Optuna for model fine-tuning.

## Test Accuracy

The table below summarizes the test accuracy of the model before and after hyperparameter tuning using Optuna.

| **Model**                             | **Test Accuracy** |
|---------------------------------------|-------------------|
| **Before Hyperparameter Tuning**      | 82.5%             |
| **After Hyperparameter Tuning**       | 92.7%             |

## Acknowledgments
- PKSLMM Dataset for providing the sign language video dataset.
