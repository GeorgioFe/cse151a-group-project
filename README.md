# CSE 151A Project - Detecting Alzheimer's via MRI Images Using CNNs and QCNNs 

This project aims to develop a multiclass classifier that predicts the presence and stage of Alzheimer's disease using MRI images. Utilizing a comprehensive dataset of approximately 5,000 images categorized into Mild Demented, Moderate Demented, Non Demented, and Very Mild Demented, we will create a traditional CNN multi-class classifier and, if time permits, a quantum-based (QCNN) classifier to compare accuracy and training speed. Additionally, we plan to develop an intuitive user interface using Streamlit to facilitate easy interaction with the model, enhancing accessibility and usability for potential users.

[Link to Milestone 2 Colab Notebook - Data Exploration](https://colab.research.google.com/drive/1OF6WwmYhwjiLKyQqQpcaLXQcdA_jNSw8?usp=sharing)

[Link to Milestone 3 Colab Notebook - Preprocessing and First Model](https://colab.research.google.com/drive/1kBraPnGcHOQ8sTkCWeyqBjkl_cgf7Qo2?usp=sharing)

[Link to Milestone 4.1 Colab Notebook - Second Model and Hyperparameter Tuning](https://colab.research.google.com/drive/19JDpgJuJrRr4vNuPY3Ldp5SmB5O1bMZX?usp=sharing)

[Link to Milestone 4.2 Colab Notebook - Bonus Model: Quantum Convolutional Neural Network (QCNN)](https://colab.research.google.com/drive/1cnEas2VgJCUUgy9E-hLxGMKkn4K7zIyg?usp=sharing)

[Final Report](https://drive.google.com/file/d/1YLZouZ1G4iNTnt36JqUXy1S6OTXRAEsW/view?usp=sharing)

## Webapp

Coming Soon!

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


## Data Preprocessing

The preprocessing phase is crucial for preparing our MRI image dataset for training our models. This phase involves several essential steps to ensure that the data is in the best possible format for model training.

### 1. Addressing Class Imbalance
Given the significant class imbalance in our dataset, we will be combining all the cases of Alzheimer's into 1 class. Therefore, our datasets will be sorted into 2 classes, Not Demented and Demented. This will solve the class imbalance issue, as the Not Demented class contains 3200 images spanning both the test and training sets, and the Demented class will have the same amount, 3200 images spanning both the test and training sets. More Specifically, We have 639 Demented cases and 640 Non Demented cases for our test data and 2561 Demented cases and 2560 Non Demented cases for our training data.

### 2. Image Normalization

Normalization is essential for standardizing the pixel values across all images. We will use the following normalization technique:
- Scaling to [0, 1]: All pixel values will be divided by 255 to scale them to the range [0, 1]. This is because pixel values in an 8-bit image range from 0 to 255, and dividing by 255 scales them to the desired range, basically having the same effect as a MinMax normalization.

## First Model

For our first model, we built a convolutional neural network using TensorFlow and Keras to classify MRI scans into demented and non-demented categories. Our model consisted of several convolutional and max-pooling layers, followed by a fully connected layer. We trained the model with a batch size of 32 for 10 epochs, monitoring its performance using a separate validation set. After training, we evaluated the model on both the training and test datasets to measure its accuracy and loss, and visualized the training progress by plotting the training and validation loss and accuracy over the epochs.

<ins>Model Hyperparameters:</ins>

- Hidden Layers Activation Function: ReLu
- Output Layer Activation Function: Sigmoid
- Loss Function: Binary Cross Entropy
- Optimizer: RMSprop
- Learning Rate: 0.0001


The results of our model can be seen on the following graphs:

![metrics](https://github.com/user-attachments/assets/c497cf5e-9e0d-43d6-b768-aaa678fce245)

After 10 epochs:
- Our train loss was around 0.23 and our train accuracy was around 91%.
- Our validation loss was around 0.68 and our validation accuracy was around 65%.
  
Given these numbers as well as the loss/error curve above, we believe that our first model is overfitting to the train data because as we see with the graphs, the training accuracy is increasing linearly over time, and similarly the training error is decreasing linearly over time, however the validation accuracy, although increased in the long run, it did not increase as the same rate and fashion as the training data. As for the error, we see that the difference between training and validation accuracy is noticeable, which is a sign of overfitting.

<ins>1. Where does your model fit in the fitting graph?</ins>

Our model is on the right end of the fitting graph (right of the optimal region) as the discrepancy between train and validation error is noticeable.

<ins>2. What are the next models you are thinking of and why?</ins>

Our next move would be to fix the overfitting issue and tune the hyperparameters of our current model to see if we will achieve an improvement from this base model. We plan to stick with this CNN model because from what we see, it is training pretty good on the training data. We just need to fix the overfitting issue and pick better values for our hyperparameters. Also if time permits, we plan on implementing a quantum-based CNN to see if we can achieve even better results.

<ins>3. What is the conclusion of your 1st model? What can be done to possibly improve it?</ins>

The initial model, while effective in achieving a reasonable classification accuracy, showed signs of overfitting, indicating that it was learning noise and unwanted details from the training examples. This overfitting hinders the model's ability to generalize to new data. To address this, we plan to improve the model by incorporating data augmentation techniques to artificially expand the training dataset and make the model more robust. Additionally, we will introduce dropout layers to prevent the model from becoming too reliant on specific neurons, further reducing overfitting. We also plan to conduct hyperparameter tuning to find the optimal parameters for our model, thereby enhancing its performance and generalization capabilities.

## Second Model

For our second model, we took a couple of steps to ensure the model is not overfitting to the training data as well as changed the structure of our CNN. To solve the overfitting issue, we first augmented our training data, which takes the approach of generating additional training data from our existing examples by augmenting them using random transformations that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better. We then added dropout layers in our CNN, which is another technique to reduce overfitting. When we apply dropout to a layer, it randomly sets the activation function to 0 to some output units from the layer during the training process.

### 1. Changes that we made:

- For data augmentation, we applied RandomZoom, RandomFlip, and RandomRotation.
- For dropout layers, we added 1 dropout layer with a rate of 0.5.
- For the structure of our CNN:
  1. Number of Layers: Our second model has 3 more additional layers compared to our first model.
  2. Preprocesing Layers: Our second model takes care of the preprocessing inside the neural network, whereas our first model did it outside the NN.
  3. Dropout Layer: Our second model includes a dropout layer before the dense layer to help reduce overfitting, which is not present in the first model.

### 2. Outcome and Evaluations:

As we see in the graphs below, the model is training much better than our first attempt. The accuracy of the train data is linearly increasing as well as the accuracy of the validation data. Similarly, the error on the train data is linearly decreasing, along with the validation error. This shows that the approach we took to handle the overfitting issue actually worked. We are now in the optimal region on the fitting graph. Additionally, by changing the structure of our CNN, we are now achieving better train and validation accuracy after 100 epochs. Around 92%.

![download (1)](https://github.com/user-attachments/assets/2443020e-2dbe-42d2-806d-853623e63f0e)

As for our test data, we can see our classification report and confusion matrix below:

![Screenshot 2024-07-31 162006](https://github.com/user-attachments/assets/0f2bef2a-475e-4b97-b184-d85df0286e4f)

![download (2)](https://github.com/user-attachments/assets/461f2676-cf6e-4158-8fa2-b94562f53f12)

As we see we are around 75% accuracy on our test data. A 50% improvement from our base model which had an accuracy of 50% on the test data.
Another thing to note is the recall. Recall measures the proportion of actual positive cases (patients with Alzheimer's) that are correctly identified by the model. In the context of Alzheimer's detection, it is crucial to minimize the number of false negatives, meaning we want to catch as many true cases of Alzheimer's as possible. Missing a diagnosis could delay treatment and support for patients who need it. In our case, the recall for AllDemented is 0.82, which is much better than our first model but can still be improved.

With our third and final model, we will try to increase the test accuracy as well as the recall with the AllDemented class via hyperparameter tuning.

## Final Model - Hyperparameter tuning

We decided to tune 4 hyperparameters:
1. The Learning Rate
2. The Optimizer
3. The Activation Function for the Convolution Layers
4. The Activation Function for the Dense Layers

We achieved this via random search and the outcomes were the following:
1. The optimal learning rate is 0.0001.
2. The optimal optimizer is adam.
3. The optimal activation functions are tanh for conv layers and relu for dense layers.

We then used those hyperparameters to retrain the model for 100 epochs.

As we see in the graphs below, the model is training in a controlled manner and no overfitting is occurring, which is a good sign. However, we did not achieve a significant improvement from our second model, so around 92% accuracy on the train and validation data.

![download (3)](https://github.com/user-attachments/assets/8d8b569b-97c0-4525-936d-eae16e727982)

Here is our classification report and confusion matrix for our final model:

![Screenshot 2024-07-31 162006](https://github.com/user-attachments/assets/63303ba1-e92a-48da-b80b-c6f4ff21bf70)

![download (4)](https://github.com/user-attachments/assets/10dc893a-78e7-421e-ab3a-ec3488216ba7)

As we see, in terms of accuracy, our final model performed worse than the second model. It has an accuracy of 63%. However, if we look at the recall on the AllDemented class, we see that it's at 0.98 which is a big improvement (about 20% improvement) from our previous model. This indicates that only 2% of the time the model is going to miss Alzheimer's detection. The only downside to this is we believe our model is learning to predict AllDemented more often than NonDemented which explains why the low accuracy compared to the recall, however in a task like this it is better to predict false positives than false negatives.

## Bonus Model - Quantum Neural Network

We know the matrix computation for classic computer is expensive, so we tried to find the feasibility of training the neural network as a quantum approach. 

<ins>Feature Maps:</ins>
Based on our research, the industry latest quantum computer can handle around 500 qubits at a time. However, when training neural network, we need to treat each pixel as a feature for the input. We have over 30,000 pixels for each picture in our dataset, it is impossible to convert all the features into quantum states. We need to find a way to reduce the quantity of features we have. The most straight forward method is downscaling. It can reduce the number of features to the maximum number that we can handle. However, this approach could loss a lot of information.

<ins>Improvement Idea:</ins>
In order to increase the accuracy of our model and reduce the impact of the information loss, we have an idea to improve the model. Theoretically, we can cut an image into smaller parts with the same size that could handled by a quantum computer, then we train these small parts first and check the accuracy to see if the patch of the image is useful for us. Then we use the accuracy and the mean value of that small patch as new features for next round training. After several rounds of iteration, we can train a large image with multiple smaller parts. There is no rigorous prove for this idea, and this approach could also cause some issue after data augmentation. But it seems doable with the dataset we chose.

<ins>Data Preprocessing:</ins>
Basically, the data preprocessing of the quantum part is very similar to what we do for the traditional machine learning. We just ensure our data are normalized, and downscaled all the images to make sure we are able to convert each feature into a limited number of superstates. We also separated the whole dataset into the training set, validation set and a test set. 

<ins>Create Quantum Circuit:</ins>
We need to design the quantum circuit based on our model in quantum computation. It is a sequence of quantum gates that represents the transformation on the quantum states. In our model, we downscaled the image into 256 pixels. So we need to initialize a 16 x 16 grid of qubits which represents 256 qubits. In order to tranlate the image data into quantum states, we applied a X gate, which means NOT, to each state where the pixel value is non-zero.

<ins>Quantum Neural Network:</ins>
We used cirq and tensorflow_quantum libraries to help us build the QNN. The basic idea is that we apply a Hardamard gate to each quantum states to convert them into super positions. Then we add some layers like ZZ or XX. After that, we add Hardamard gates one more time, and measure the results that a quantum computer calculated. Then we do the same as a traditional quantum computer does. We compile our model with our ideal loss function and optimizer, and fit our data with specific batch size and epochs. But for this time, we do not need to care about the matrix calculation and activation functions.
=======
