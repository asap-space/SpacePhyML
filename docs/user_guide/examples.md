# Examples

An incomplete example on how to use the MMSDataset class. Here we assume that the dataset
`my_dataset.csv`, have been created using the [dataset creator](./dataset_creator.md)

````
import torch
from torch.utils.data import DataLoader
from spacephyml.datasets import MMSDataset

dataset = MMSDataset('my_dataset.csv')

for x,y, _, _ in DataLoader(dataset, batch_size=32, shuffle = True):
    train(network,x,y)

````

## Classifying MMS dayside space plasma regions

````
import torch
from torch.utils.data import DataLoader
from spacephyml.datasets.mms import MMS1IonDistLabeled
from spacephyml.models.mms import PCReduced


dataset = MMS1IonDistLabeled('SCDec2017')

model = PCReduced()

for x,l, _, _ in DataLoader(dataset, batch_size=32, shuffle = True):
    lc = model(x)

````


