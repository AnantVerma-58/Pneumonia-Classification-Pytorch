from torchvision.models import VGG16_Weights, vgg16
import utils
from data_transformation import image_transform
from load_data import dataloader
import torch.nn as nn
from torchinfo import summary
from steps import train_step, validation_step, test_step
from helper_functions import accuracy_fn



weights = VGG16_Weights.DEFAULT

train_dataloader, validation_dataloader, test_dataloader = dataloader(batch_size=32, transform=image_transform())

model = vgg16(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=25088,
              out_features=2,
              bias=True),
    nn.Sigmoid()
)


loss_fn = utils.loss()

optimizer = utils.optimizer(params=model.parameters(), lr=0.01)

device = utils.device()

epochs = utils.epochs

from timeit import default_timer as timer

start = timer()
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    validation_step(data_loader=validation_dataloader,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,optimizer = optimizer
    )
    test_step(data_loader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,optimizer = optimizer
    )

end = timer()
total_time = end - start
print(f"Train time on {device}: {total_time:.3f} seconds")