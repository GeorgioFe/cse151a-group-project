# CSE 151A Project - Detecting Alzheimer's via MRI Images Using CNNs and QCNNs 

This project aims to develop a multiclass classifier that predicts the presence and stage of Alzheimer's disease using MRI images. Utilizing a comprehensive dataset of approximately 5,000 images categorized into Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented, we will create a traditional CNN multi-class classifier and, if time permits, a quantum-based (QCNN) classifier to compare accuracy and training speed. Additionally, we plan to develop an intuitive user interface using Streamlit to facilitate easy interaction with the model, enhancing accessibility and usability for potential users.

[Link to Colab Notebook](https://colab.research.google.com/drive/1OF6WwmYhwjiLKyQqQpcaLXQcdA_jNSw8?usp=sharing)

## Dataset

[Link to Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images/data)

The Alzheimer’s MRI Image Dataset contains approximately 6400 MRI images, divided into training and testing sets. These images are categorized into four classes: Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented. The dataset is designed to aid in developing highly accurate models for predicting the stages of Alzheimer’s disease. The primary inspiration behind this dataset is to support advancements in deep learning for Alzheimer’s stage prediction.

## Data Exploration

### 1. Image Size

We checked the image sizes across the dataset (code available in the notebook). We confirmed that all images in both the training and testing sets have a uniform size of 176x208 pixels. This consistency simplifies the preprocessing steps and ensures uniformity in model input dimensions. No additional cropping is necessary since all images are already of the same size. However, normalization will be performed to standardize the pixel values, which is crucial for optimal model performance.

### 2. Image Count Per Class

We assessed the distribution of images across the four classes (Mild Demented, Moderate Demented, Non-Demented, and Very Mild Demented) in both the training and testing sets.
Below are the histograms showing the number of samples per class for the training and testing sets:


![training](https://github.com/user-attachments/assets/e2e828d2-3977-43d4-920e-7e8563dfaef2) 
![test](https://github.com/user-attachments/assets/5c0931bd-0e9f-4885-9ad2-326446dfc80c)

As illustrated in the histograms, there is a significant imbalance in the dataset. The "Non Demented" class has the highest number of samples, followed by "Very Mild Demented", "Mild Demented", and lastly, "Moderate Demented". This imbalance poses a challenge as the model might become biased towards the classes with more samples, potentially reducing the accuracy for underrepresented classes.

The challenge with imbalanced datasets is that classification models attempt to categorize data into different buckets. In an imbalanced dataset, one bucket makes up a large portion of the training dataset (the majority class), while the other bucket is underrepresented in the dataset (the minority class). The problem with a model trained on imbalanced data is that the model learns that it can achieve high accuracy by consistently predicting the majority class, even if recognizing the minority class is equally or more important when applying the model to a real-world scenario.

Consider the case of our Alzheimer's MRI image dataset. Most of the images collected fall into the "Non Demented" category, while the "Moderate Demented" patients make up a much smaller portion of the data. During training, the classification model learns that it can achieve high accuracy by predicting "Non Demented" for every MRI image it encounters. That’s a huge problem because what medical professionals really need the model to do is identify those patients in the early or moderate stages of Alzheimer's disease.

More on this will be talked in the Preproccessing step.

### 3. Bluriness Check:

To guarantee that our dataset consists of high-quality images, we checked for image blurriness. Our analysis showed that all images were of consistent quality with no outliers in terms of blurriness (code available in the linked notebook). This ensures that our model is trained on clear and precise images, enhancing its accuracy and reliability.

### 4. Color Distribution

Given that MRI images are typically grayscale, we examined the color distribution for each class to verify uniformity. We plotted the color distribution and confirmed that the grayscale intensity levels were consistent across all classes. This step helps in understanding the inherent differences in image characteristics across different stages of Alzheimer's disease. Plotted below:

![mean_intensity_values](https://github.com/user-attachments/assets/ffbd3acf-ab0e-48cf-bee9-75ef2e930df1)

### 5. Visualizing Sample Images

To get a better visual understanding of the dataset, we plotted examples of images from each class. These visualizations provided insights into the subtle differences and similarities in MRI images for each stage of Alzheimer’s, which is crucial for model training.

![samples](https://github.com/user-attachments/assets/c142f2f3-fa35-4428-ab5e-a72e5f58a8da)

### 6. Summary of Data

The data exploration phase has provided us with a comprehensive understanding of our dataset. By ensuring uniform image sizes, consistent image quality, uniform grayscale intensity levels, and planning for image normalization, we have laid a strong foundation for training our CNN and QCNN models. However, the class imbalance issue needs to be addressed in the preprocessing step. The details of addressing class imbalance, normalization, and other preprocessing steps will be discussed in the preprocessing section.


## Data Preprocessing (Milestone 3)

The preprocessing phase is crucial for preparing our MRI image dataset for training our models. This phase involves several essential steps to ensure that the data is in the best possible format for model training.

### 1. Addressing Class Imbalance

Given the significant class imbalance in our dataset, we are looking into different ways we can handle it. Here are some methods:
- Random Under-Sampling
- Random Over-Sampling
- Tomek Links
- Synthetic Minority Oversampling Technique (SMOTE)
- NearMiss
- Penalize Algorithms (Cost-Sensitive Training)
- Change the Performance Metric

### 2. Image Normalization

Normalization is essential for standardizing the pixel values across all images. We will use the following normalization technique:
- Scaling to [0, 1]: All pixel values will be divided by 255 to scale them to the range [0, 1]. This is because pixel values in an 8-bit image range from 0 to 255, and dividing by 255 scales them to the desired range.
