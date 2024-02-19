import torchvision.transforms as transforms


image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])
# def image_transform():
#     from torchvision.models import VGG16_Weights
#     weights = VGG16_Weights.DEFAULT
#     auto_transforms = weights.transforms()
#     return auto_transforms