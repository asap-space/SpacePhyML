import torch
import spacephyml as spm
from spacephyml.datasets import mms
from spacephyml.models.arcs import PCReduced

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss

        pred = model(x.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if __VERBOSE and batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, epoch, device, csv_filename = "test_accuracy.csv"):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    acc = 100*correct
    print(f"Test Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    write_to_csvfile(csv_filename, {
                     "Accuracy" : [acc],
                     "Avg loss" : [test_loss]}, epoch)
    return acc

def train_model(model, train_set, test_set, epochs = 2, batch_size=64,
                learning_rate = __DEFAULT_LEARNING_RATE, device="cpu", eps = None,
                csv_filename = "test_accuracy.csv", bailout = 10):

    dataloader_train = DataLoader(train_set, batch_size = batch_size,
                                  shuffle=True,num_workers=10)
    dataloader_test = DataLoader(test_set, batch_size = batch_size,
                                 num_workers=10)

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer_kwargs = {'lr': learning_rate}
    if eps:
        optimizer_kwargs['eps'] = eps

    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    highest_acc = 0
    epochs_without_acc_increase = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader_train, model, loss_fn, optimizer, batch_size, device)
        current_acc = test_loop(dataloader_test, model, loss_fn, t+1, device, csv_filename)

        if current_acc > highest_acc:
            highest_acc = current_acc
            epochs_without_acc_increase = 0
        elif epochs_without_acc_increase > bailout:
            print(f"Bailout at {t} epochs! Accuracy have not incresed for " +
                    f"{epochs_without_acc_increase} epochs.")
            return model, t
        else:
            epochs_without_acc_increase += 1

    print("Done!")

    return model, epochs

def main(seed):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")

    torch.manual_seed(seed)

    dataset_train = mms.MMS1IonDistLabeled('SCNov2017')
    dataset_test = mms.MMS1IonDistLabeled('SCDec2017')
    model = PCReduced()

    base_filename = f"train_{seed}.pth"

    train_model_kwargs = {
        'epochs': 200,
        'learning_rate': 1e-5,
        'device': device,
        'csv_filename': f"test_accuracy_{base_filename}" }

    model, epochs = train_model(model, dataset_train, dataset_test, **train_model_kwargs)

    torch.save(model.classifier.state_dict(), f"model_{base_filename}")

if __name__ == "__main__":
    for seed in [42,84,168,336]:
        main(seed)
