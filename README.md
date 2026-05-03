# Brain Tumor Detection from MRI Images using CNN

## Project Overview
This project classifies MRI brain scans into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

A custom Convolutional Neural Network (CNN) is implemented using PyTorch.

## Why CNN?
CNNs automatically learn spatial image features such as edges, textures,
and tumor shapes, making them highly effective for medical image classification.

## Dataset
Brain Tumor MRI Dataset:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Installation
pip install -r requirements.txt
#  To install packages one by one, we use the below code
python3 -m pip install torch torchvision numpy opencv-python pillow scikit-learn matplotlib

## Train Model...The required CNN model is being made
python3 train.py

## "checkpoints" folder contains the final weights of the model.

## Evaluate Model...The CNN model is tested on unseen test data (different from training data)
python3 test.py

## Predict New Images...New images are the unseen test images
python3

from predict import predict_batch
predict_batch(['data/glioma/img01.jpg']) 
# Similarly did for all the classes

## "data" folder contains 10 unseen images of each class

## Hyperparameters
- Batch Size: 16
- Epochs: 10
- Learning Rate: 0.001
- Image Size: 128x128

## Files Included
- model.py : CNN architecture
- dataset.py : Dataset and DataLoader
- train.py : Training loop
- test.py : Evaluation metrics
- predict.py : Batch prediction on unseen images
- interface.py : Grading interface file
- config.py : Hyperparameters

## Results
Final Test Accuracy: 90.38%
Metrics for each class:
              precision    recall  f1-score   support

      glioma       0.92      0.79      0.85       400
  meningioma       0.88      0.91      0.89       400
     notumor       0.85      0.99      0.92       400
   pituitary       0.97      0.93      0.95       400

    accuracy                           0.90      1600
   macro avg       0.91      0.90      0.90      1600
weighted avg       0.91      0.90      0.90      1600

Confusion Matrix: 
 [[316  29  49   6]
 [ 13 363  18   6]
 [  2   3 395   0]
 [ 11  17   0 372]]

 So, our multi-class classification accuracy is pretty strong. This model is reliable and can be used to detect the various kinds of brain tumours using MRI images.

## Future Improvements
- Data augmentation
- Deeper CNN
- Grad-CAM visualization
- Transfer learning comparison - EfficientNet
