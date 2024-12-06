# Examples

An incomplete example on how to use the MMSDataset class. Here we assume that the dataset
`my_dataset.csv`, have been created using the [dataset creator](./dataset_creator.md)

````
import torch
from torch.utils.data import Dataset, DataLoader
from spacephyml.datasets import MMSDataset

dataset = MMSDataset('my_dataset.csv')

for x,y, _, _ in DataLoader(dataset, batch_size=32, shuffle = True):
    train(network,x,y)

````
