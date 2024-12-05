# Pakistani Sign Language Recognition using PkSLMNM Dataset

## Introduction
Pakistani Sign Language (PSL) research has been limited due to the scarcity of annotated datasets and comprehensive studies. This project aims to address this gap by utilizing the PkSLMNM dataset to develop a robust PSL recognition system. Through data augmentation and advanced deep learning techniques, the system extracts temporal and spatial features from sign videos for accurate classification.

## Features

### Dataset Preparation
- Data augmentation and frame extraction from PkSLMNM dataset videos.
- Hierarchical directory structure for training, validation, and testing sets.

### Feature Extraction
- **CNN** for spatial feature extraction.
- **LSTM** for temporal feature learning and sequence modeling.

### Custom Dataset Loader
- Handles padded sequences for variable-length frames.
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

## Installation

To run this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/pk-sign-language-recognition.git
cd pk-sign-language-recognition
pip install -r requirements.txt
