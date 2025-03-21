"""
Load an existing dataset and model, then classify
the data using the model.
"""
import torch
from torch.utils.data import DataLoader

from spacephyml.datasets.mms import MMS1IonDistLabeled
from spacephyml.models import PCReduced

dataset = MMS1IonDistLabeled('SCDec2017')
model = PCReduced(seed='42')

labels = {'human': [], 'classifier': []}
with torch.no_grad():
    for x,l, _, in DataLoader(dataset, batch_size=32):
        lc = model(x)
        labels['human'].extend(l)
        labels['classifier'].extend(lc)
