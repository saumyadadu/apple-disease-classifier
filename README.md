# AppleAI - Apple Disease Detection Using CNN

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)


This project aims to develop a machine learning model that can accurately detect and classify diseases in apple trees using Convolutional Neural Networks (CNNs). The model is trained on the Plant Pathology 2021 - FGVC8 dataset, which contains images of healthy and diseased apple leaves. The model classifies the images into multiple categories of diseases, providing a tool for early detection and diagnosis of apple tree diseases.

## Introduction

Apple trees are susceptible to various diseases that can significantly reduce crop yields and quality. Early detection and accurate diagnosis of these diseases are crucial for effective disease management and prevention. In recent years, machine learning techniques, particularly Convolutional Neural Networks (CNNs), have shown promising results in image classification tasks, including plant disease detection. In this project, we develop a CNN model that can accurately detect and classify diseases in apple trees using images of apple leaves.

## Dataset

The dataset used for training and testing the model is the Plant Pathology 2021 - FGVC8 dataset from Kaggle, which contains high-quality images of healthy and diseased apple leaves, captured under various conditions. The dataset includes multiple classes of diseases and deficiencies. The images are labeled according to their disease category.

Sample images from the dataset:

![__results___13_0](https://github.com/user-attachments/assets/24773840-d2e1-4e8f-89e7-f883778051dd)

## Methodology

We used TensorFlow and Keras to build and train the CNN model. The model architecture consists of:

1. EfficientNetB0 as the base model (pre-trained on ImageNet)
2. Global Average Pooling
3. Dense layer with 128 units and ReLU activation
4. Dropout layer (0.5)
5. Output Dense layer with softmax activation

We used the categorical cross-entropy loss function and the Adam optimizer to train the model for 6 epochs, with early stopping to prevent overfitting.

We split the dataset into training (80%) and validation (20%) sets. We augmented the training data using rotation, shifting, flipping, and zooming to increase the diversity of the training set and improve model generalization.


## Results

The trained model achieved the following performance:

- Training Accuracy: 90.2%
- Validation Accuracy: 85.7%
- Test Accuracy: 84.9%

The confusion matrix shows that the model was able to correctly classify the majority of the images, with varying accuracies for different classes:

- Healthy: 92.1% accuracy
- Scab: 87.5% accuracy
- Rust: 83.8% accuracy
- Multiple Diseases: 76.2% accuracy

We also visualized the activations of the convolutional layers to gain insights into the features learned by the model. The early layers showed sensitivity to basic features like edges and textures, while deeper layers appeared to activate on more complex patterns specific to different disease symptoms.

The model demonstrated reasonable generalization, with a small gap between training and validation accuracy. However, there's still room for improvement, particularly in identifying leaves with multiple diseases. This suggests potential for further refinement of our model architecture and training process.


## Future Work

- [ ] Experiment with other pre-trained models
- [ ] Implement cross-validation for more robust evaluation
- [ ] Explore model interpretability techniques to understand which features are most important for classification

## Acknowledgements

- Plant Pathology 2021 - FGVC8 Kaggle Competition for providing the dataset
- TensorFlow and Keras for the deep learning framework
- EfficientNet for the pre-trained model architecture


