import torch

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
  train_loss=0
  train_acc=0
  model.to(device)
  for batch, (X, y) in enumerate(data_loader):
    X,y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y_true=y,
                             y_pred=y_pred.argmax(dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
  test_loss=0
  test_acc=0
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device), y.to(device)

      test_pred = model(X)

      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y_true = y, y_pred = test_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def validation_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
  val_loss=0
  val_acc=0
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device), y.to(device)

      val_pred = model(X)

      val_loss += loss_fn(val_pred, y)
      val_acc += accuracy_fn(y_true = y, y_pred = val_pred.argmax(dim=1))

    val_loss /= len(data_loader)
    val_acc /= len(data_loader)
    print(f"Validation loss: {val_loss:.5f} | Validation accuracy: {val_acc:.2f}%\n")