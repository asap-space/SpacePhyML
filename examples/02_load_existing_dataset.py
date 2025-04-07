"""
Load an existing dataset and model, then classify
the data using the model.
"""
import torch
from torch.utils.data import DataLoader

from spacephyml.datasets.mms import MMS1IonDistLabeled
from spacephyml.models.mms import PCReduced

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

dataset = MMS1IonDistLabeled('SCNov2017')
model = PCReduced('s42').to(device)

labels = {'human': [], 'classifier': []}
correct = 0
with torch.no_grad():
    for x,l in DataLoader(dataset, batch_size=32):
        lc = model(x.to(device)).to('cpu')
        lc = torch.argmax(lc, axis = 1)
        correct += torch.sum(lc==l)
        labels['human'].extend(l)
        labels['classifier'].extend(lc)
    print(f'Accuracy: {correct/len(labels['human'])}')
