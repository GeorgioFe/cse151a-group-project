# CSE 151A Project - Detecting Alzheimer's via MRI Images Using CNNs and QCNNs 

This project aims to develop a multiclass classifier that predicts the presence and stage of Alzheimer's disease using MRI images. Utilizing a comprehensive dataset of approximately 5,000 images categorized into Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented, we will create a traditional CNN multi-class classifier and, if time permits, a quantum-based (QCNN) classifier to compare accuracy and training speed. Additionally, we plan to develop an intuitive user interface using Streamlit to facilitate easy interaction with the model, enhancing accessibility and usability for potential users.

## Dataset

[Link to Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images/data)

The Alzheimer’s MRI Image Dataset contains approximately 6400 MRI images, divided into training and testing sets. These images are categorized into four classes: Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented. The dataset is designed to aid in developing highly accurate models for predicting the stages of Alzheimer’s disease. The primary inspiration behind this dataset is to support advancements in deep learning for Alzheimer’s stage prediction.

## Data Exploration

### 1. Image Size

We checked the image sizes across the dataset (code available in the notebook). We confirmed that all images in both the training and testing sets have a uniform size of 176x208 pixels. This consistency simplifies the preprocessing steps and ensures uniformity in model input dimensions. No additional cropping is necessary since all images are already of the same size. However, normalization will be performed to standardize the pixel values, which is crucial for optimal model performance.

### 2. Image Count Per Class

We assessed the distribution of images across the four classes (Mild Demented, Moderate Demented, Non-Demented, and Very Mild Demented) in both the training and testing sets.
Below are the histograms showing the number of samples per class for the training and testing sets:


![training](https://github.com/user-attachments/assets/e2e828d2-3977-43d4-920e-7e8563dfaef2) ![test](https://github.com/user-attachments/assets/5c0931bd-0e9f-4885-9ad2-326446dfc80c)


## Data Preprocessing
(How will you preprocess your data?)
