# Pakistani Sign Language Recognition using PkSLMNM Dataset

## Introduction
Pakistani Sign Language (PSL) research has been limited due to the scarcity of annotated datasets and comprehensive studies. This project aims to address this gap by utilizing the PkSLMNM dataset to develop a robust PSL recognition system. Through data augmentation and advanced deep learning techniques, the system extracts temporal and spatial features from sign videos for accurate classification.

The PkSLMM dataset contains videos of 7 basic affective expressions performed by 100 healthy individuals, presented in an easily accessible format of .MP4 that can be used to train and test systems to make robust models for real-time applications using videos.

---

## Vision Transformer (ViT) and Real-Time Inference

### Vision Transformer (ViT)

The Vision Transformer (**ViT**) is a cutting-edge deep learning model that processes images by dividing them into patches and treating them as sequences. ViT effectively captures long-range dependencies and spatial features, making it highly effective for image classification tasks.

### Real-Time Inference

Real-time inference with ViT enables immediate classification of sign language gestures from video input. By processing frames on-the-fly, the system delivers quick predictions, suitable for live communication aids and assistive devices.

### Implementation Steps:

1. **Frame Extraction and Preprocessing**:
    - Extract frames from video input using OpenCV.
    - Resize and format frames to match the input requirements of ViT.

2. **Model Inference**:
    - Load the pre-trained ViT model fine-tuned on the PkSLMNM dataset.
    - Perform inference on each frame to classify sign language gestures in real-time.

3. **Visualization**:
    - Overlay the predicted sign language labels onto the video frames.
    - Display the annotated video stream to the user.


## LSTM Approach and ViT Approach - Features

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
