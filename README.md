# Pneumonia-Classification-Pytorch

# X-Ray Image Classification

![Project Logo](link/to/your/logo.png)

## Overview

This project aims to perform X-Ray image classification using multiple pretrained models such as VGG16, DenseNet, AlexNet, and ResNet. The models are trained on a custom dataset, and their performance is compared using metrics like accuracy. MLflow is used for experiment tracking and model versioning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Training](#training)
- [Experiment Tracking](#experiment-tracking)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download pretrained models:

   ```bash
   # You may need to download models using torchvision
   ```

## Usage

1. Start the application:

   ```bash
   python app.py
   ```

2. Access the application in your web browser at [http://localhost:8000](http://localhost:8000).

## Dataset

Describe your dataset, its structure, and how to obtain it. If it's a public dataset, provide a link.

## Models

- **VGG16**: [Link to VGG16](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16)
- **DenseNet**: [Link to DenseNet](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.densenet121)
- **AlexNet**: [Link to AlexNet](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.alexnet)
- **ResNet**: [Link to ResNet](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet18)

# Pneumonia Classification Project

## Overview
This project focuses on classifying X-Ray images to detect pneumonia. It uses multiple pre-trained models such as VGG16, DenseNet, AlexNet, and ResNet to train and compare their performance. The project includes features for uploading images, selecting sample images, and viewing prediction results.

## Features
- **Image Upload:** Users can upload X-Ray images for classification.
- **Sample Image Selection:** Choose from pre-loaded sample images for quick classification.
- **Model Comparison:** Trained models include VGG16, DenseNet, AlexNet, and ResNet.
- **Result Display:** View predictions, probabilities, and uploaded images in the result page.
- **Aesthetic Design:** The interface is designed for a pleasant user experience.

## Training and Comparison
The models were trained using a diverse dataset, including training, testing, and validation sets. The performance of each model was compared based on metrics such as accuracy, precision, recall, and F1-score. Graphs showing training and validation loss and accuracy are available.

## Usage
1. **Upload Image:** Use the file upload form to submit your own X-Ray image for classification.
2. **Select Sample Image:** Choose from the provided sample images for a quick prediction.
3. **Result Page:** View the classification result, uploaded image, prediction, and probabilities.

## Model Comparison
- **VGG16:** [Details about VGG16 model and performance]
- **DenseNet:** [Details about DenseNet model and performance]
- **AlexNet:** [Details about AlexNet model and performance]
- **ResNet:** [Details about ResNet model and performance]

## Results
Graphs and visualizations comparing the performance of each model are available in the project.

### Sample Results
![Training Loss Graph](/path/to/training_loss.png)
![Validation Accuracy Graph](/path/to/validation_accuracy.png)

## How to Run
1. Install the required dependencies: [List of dependencies]
2. Run the application: `python app.py`
3. Access the application in your web browser: [http://localhost:8000/](http://localhost:8000/)

## Acknowledgments
- Mention any libraries, frameworks, or datasets used.
- Credit any pre-trained models or code snippets used in the project.
