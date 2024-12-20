import cv2
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np

def get_sign_language_label(class_index):
    # Assuming you have a list of labels in the same order as the class indices
    labels = ['bad', 'best', 'glad', 'sad', 'scared', 'stiff', 'surprise']
    return labels[class_index]

# Load the trained ViT model
model = ViTForImageClassification.from_pretrained(r'E:\Projects\Finished\Sign Language\ViT\PkSLMNM_Model')

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(r'E:\Projects\Finished\Sign Language\ViT\PkSLMNM_Model')

# Define the video capture
cap = cv2.VideoCapture(0)  # You may change the parameter to the appropriate device index or video file

# Set up real-time inference
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    inputs = feature_extractor(images=frame, return_tensors="pt")
    
    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits

    # Post-process the inference results (example: get the predicted class)
    predicted_class = torch.argmax(logits, dim=-1).item()
    sign_language_label = get_sign_language_label(predicted_class)

    
    # Display the results
    cv2.putText(frame, sign_language_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-time Sign Language Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
