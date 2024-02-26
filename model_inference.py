import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from data_transformation import image_transform
# Assuming 'model' is your VGG16 model
def inference(image_path:str=None):
    vgg16_model = torch.jit.load('vgg16_final_model_jit.pt').to('cpu')
    vgg16_model.eval()

    # Load and preprocess the image
    image_path = image_path  # Replace with the actual path to your image
    img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    transform = image_transform()
    img = transform(img)
    img = img.unsqueeze(0)  # Add a batch dimension

    # Perform the prediction
    with torch.no_grad():
        output_logits = vgg16_model(img)

    # Apply softmax externally if needed
    probabilities = F.softmax(output_logits, dim=1)
    # print("probability : ",probabilities)
    # Get the predicted class index
    predicted_class = torch.argmax(probabilities).item()
    probabilities = probabilities.squeeze(dim=0).tolist()
    # print(f"Predicted class: {predicted_class}, Probability: {probabilities[0][predicted_class].item()}")
    class_labels = ["Normal","Pneumonia"]
    # print(f"Predicted class: {predicted_class}, Class Name: {class_labels[predicted_class]}, Probability: {probabilities}")
    return {"Predicted Class":predicted_class, "Class Label":class_labels[predicted_class], "probabilities":probabilities}



# paths = [i for i in range(1,6)]
# for nums in paths:
#     print(inference(image_path=f"normal0{nums}.jpeg"))
# print("\n")

# for nums in paths:
#     inference(image_path=f"notnormal0{nums}.jpeg")