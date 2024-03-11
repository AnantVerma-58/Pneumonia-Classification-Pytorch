# Pneumonia Classification from X-Ray Images



## Overview

This project aims to perform X-Ray image classification using multiple pretrained models such as VGG16, DenseNet, AlexNet, and ResNet. The models are trained on a custom dataset, and their performance is compared using metrics like accuracy. MLflow is used for experiment tracking and model versioning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Deployment](#deployment)
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AnantVerma-58/Pneumonia-Classification-Pytorch.git
   cd Pneumonia-Classification-Pytorch
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:

   ```bash
   uvicorn fastapiapp:app
   ```

2. Access the application in your web browser at [http://localhost:8000](http://localhost:8000).

## About the data
The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.
http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## Dataset
The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.
[dataset on kagg\le](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Models

- **VGG16**: [Link to VGG16](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16)
- **DenseNet**: [Link to DenseNet201](https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet201.html#torchvision.models.densenet201)
- **AlexNet**: [Link to AlexNet](https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.alexnet)
- **ResNet**: [Link to ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)

## Results

| Training Accuracy                                         |                       Training Loss                 |
| --------------------------------------------------------- | --------------------------------------------------- |
| ![training accuracy](/results/train_accuracy.png)         |          ![train loss](/results/train_loss.png)     |
| Validation Accuracy                                       |                     Validation Loss                 |
| ![training accuracy](/results/validation_accuracy.png)    |    ![validation loss](/results/validation_loss.png) |
| Test Accuracy                                             |                          Test Loss                  |
| ![training accuracy](/results/test_accuracy.png)          |       ![test loss](/results/test_loss.png)          |

Legends

$${\color{orange}---- VGG16}$$

$${\color{pink}---- Alexnet}$$

$${\color{yellow}---- Densenet}$$

$${\color{green}---- Resnet}$$

VGG16 outperformed AlexNet, DenseNet, and ResNet in training improvement, as evidenced by consistently superior validation and test results. Its deeper architecture and increased model complexity contributed to enhanced learning capabilities, demonstrating superior performance across various evaluation metrics.

## Results of Training VGG16

Learning Rates

![Learning Rates](/results/f69beaaf-7fee-4e1c-954a-7d23ed705083.png)

| Training Accuracy                                         |                       Training Loss                 |
| --------------------------------------------------------- | --------------------------------------------------- |
| ![Training Accuracy](/results/vgg16_train_accuracy.png)   |          ![Training Loss](/results/vgg16_train_loss.png)     |
| Validation Accuracy                                       |                     Validation Loss                 |
| ![Validation Accuracy](/results/vgg16_validation_accuracy.png)    |    ![Validation Loss](/results/vgg16_validation_loss.png) |
| Test Accuracy                                             |                          Test Loss                  |
| ![Test Accuracy](/results/vgg16_test_accuracy.png)        |       ![Test Loss](/results/vgg16_test_loss.png)          |

We employed various learning rate schedules during the training of VGG16 and observed their impact on model performance. Among the decay schedules experimented, including step, cosine, exponential, polynomial, natural exponential, and staircase decay, the polynomial decay schedule emerged as the most effective. The determination was made by analyzing training progress through graphical representations, where the polynomial decay schedule demonstrated superior convergence and optimization compared to other schedules. This approach provided valuable insights into selecting the optimal learning rate schedule for maximizing training efficiency and model performance.

## Deployment
The application was containerized using Docker, allowing for a consistent and reproducible deployment environment. The Docker container, encapsulating the entire application and its dependencies, was then uploaded to Azure for hosting. This containerized approach ensures seamless deployment across different environments and facilitates efficient scaling and management on the Azure platform. The docker instance can be created and uploaded on cloud platform like azure to be made acessible for public use.

![](https://avatars.githubusercontent.com/u/5429470?s=200&v=4)

![](https://avatars.githubusercontent.com/u/6844498?s=200&v=4)

## Usage
1. **Upload Image:** Use the file upload form to submit your own X-Ray image for classification.
2. **Select Sample Image:** Choose from the provided sample images for a quick prediction.
3. **Result Page:** View the classification result, uploaded image, prediction, and probabilities.

## How to Run
1. Install the required dependencies: [List of dependencies]
2. Run the application: `uvicorn fastapiapp:app`
3. Access the application in your web browser: [http://localhost:8000/](http://localhost:8000/)

