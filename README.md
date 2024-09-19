# Apple Disease Classifier

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Deep learning model for apple disease detection using EfficientNetB0 and TensorFlow. Classifies 12 apple disease categories from images to aid in early diagnosis and crop management.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Performance](#performance)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project implements a state-of-the-art deep learning model to detect and classify diseases in apple trees. It utilizes the EfficientNetB0 architecture with transfer learning, built using TensorFlow and Keras.

## Dataset

The model is trained on the [Plant Pathology 2021 - FGVC8](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8) dataset from Kaggle. This dataset contains high-quality images of apple leaves with various diseases and deficiencies.

### Sample Images

<table>
  <tr>
    <td><img src="images/healthy_leaf.jpg" alt="Healthy Leaf" width="200"/></td>
    <td><img src="images/scab.jpg" alt="Scab" width="200"/></td>
  </tr>
  <tr>
    <td align="center"><em>Healthy Apple Leaf</em></td>
    <td align="center"><em>Apple Leaf with Scab</em></td>
  </tr>
  <tr>
    <td><img src="images/rust.jpg" alt="Rust" width="200"/></td>
    <td><img src="images/multiple_diseases.jpg" alt="Multiple Diseases" width="200"/></td>
  </tr>
  <tr>
    <td align="center"><em>Apple Leaf with Rust</em></td>
    <td align="center"><em>Apple Leaf with Multiple Diseases</em></td>
  </tr>
</table>

## Features

- Multi-class image classification for apple diseases
- Data augmentation for improved model generalization
- Transfer learning using EfficientNetB0 pre-trained model
- Model evaluation with accuracy, precision, and recall metrics
- Single image classification functionality

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- OpenCV
- TensorFlow Hub

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/apple-disease-classifier.git
   cd apple-disease-classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset:
   - Download the dataset from the Kaggle competition
   - Organize the data as follows:
     ```
     /content/
     ├── train_images/
     ├── test_images/
     ├── train.csv
     └── sample_submission.csv
     ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. The script will:
   - Load and preprocess the data
   - Build and train the model
   - Evaluate the model on the validation set
   - Generate predictions for the test set
   - Create a submission file
   - Plot training history

4. To classify a single image:
   ```python
   upload_and_classify_image()
   ```

## Model Architecture

```
EfficientNetB0 (pre-trained on ImageNet)
│
├── Flatten
│
├── Dense (128 units, ReLU activation)
│
├── Dropout (0.5)
│
└── Dense (12 units, Softmax activation)
```

## Training

- **Batch size:** 32
- **Epochs:** 6 (with early stopping)
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss function:** Categorical Crossentropy

## Performance

The model's performance is evaluated using:
- Validation Accuracy
- Validation Precision
- Validation Recall

Actual values will be displayed after training.

## Future Improvements

- [ ] Experiment with other pre-trained models
- [ ] Implement cross-validation
- [ ] Add more data augmentation techniques
- [ ] Fine-tune hyperparameters
- [ ] Develop a web interface for easy image upload and classification

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [Plant Pathology 2021 - FGVC8 Kaggle Competition](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8)
