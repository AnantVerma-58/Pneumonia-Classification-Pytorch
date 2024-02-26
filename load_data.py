from torchvision.datasets import ImageFolder
from data_transformation import image_transform
from torch.utils.data import DataLoader

def dataloader(batch_size=32, transform = image_transform):
    train_data = ImageFolder("chest_xray/train", transform = transform)
    test_data = ImageFolder("chest_xray/test", transform = transform)
    validation_data = ImageFolder("chest_xray/val", transform = transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader