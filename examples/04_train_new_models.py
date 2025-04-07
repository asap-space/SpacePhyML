"""
Train a new PCReduced model using the PCNov2017 dataset.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from spacephyml.datasets.mms import MMS1IonDistLabeled
from spacephyml.models.arcs.mms import PCReduced_arc, PCBaseline_arc

_VERBOSE = True
_EPOCHS = 5
_LEARNING_RATE = 1e-5
_BATCH_SIZE = 32


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

        if _VERBOSE and batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_model(model, dataset, batch_size = _BATCH_SIZE, device="cpu"):

    dataloader_train = DataLoader(dataset, batch_size = _BATCH_SIZE, shuffle=True)

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = _LEARNING_RATE)

    for t in range(_EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader_train, model, loss_fn, optimizer, batch_size, device)

    print("Done!")

    return model


def main(seed):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    torch.manual_seed(seed)

    print(f"Using {device} device")

    dataset = MMS1IonDistLabeled('SCNov2017')

    model = PCBaseline_arc()

    model = train_model(model, dataset, device = device)

    torch.save(model.classifier.state_dict(), f"./model_PCBaseline_s{seed}.ptk")

if __name__ == "__main__":
    seed = 42
    main(seed)
