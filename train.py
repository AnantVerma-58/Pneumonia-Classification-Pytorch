from data_transformation import image_transform
from steps import train_step, validation_step, test_step
from load_data import dataloader
from torchvision.models import vgg16, VGG16_Weights
import torch
from torch import nn, optim



train_dataloader, validation_dataloader, test_dataloader = dataloader(batch_size=32, transform= image_transform)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = vgg16(weights=VGG16_Weights)


for param in model.features.parameters():
  param.requires_grad = False


model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=25088,
              out_features=2,
              bias=True)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = model.parameters(),
                      lr = 0.1)



def print_train_time(start: float, end: float, device: torch.device = None):

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time
     



torch.manual_seed(42)
from tqdm.auto import tqdm
from helper_functions import accuracy_fn
from timeit import default_timer as timer
train_time_start_model = timer()

# Train and test model
epochs = 10
for epoch in tqdm(range(epochs)):
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
    test_step(data_loader=validation_dataloader,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device,optimizer = optimizer
    )

train_time_end_model = timer()
total_train_time_model = print_train_time(start=train_time_start_model,
                                           end=train_time_end_model,
                                           device=device)
     