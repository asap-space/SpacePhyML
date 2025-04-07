# Quick Start

## Installation
To use SpacePhyML you can install the package using pip.

```
pip install git+https://github.com/Jonah-E/SpacePhyML.git
```

## Usage

Create a dataset through the commandline tool using the dataset generator.

```
spacephyml create my_dataset.csv
```

Load the dataset in your python script, alternativly you can use one of the already exisiting [datasets](user_guide/datasets_mms.md).

```
from spacephyml.datasets.general.mms import ExternalMMSData

dataset = ExternalMMSData('my_dataset.csv')
```

Load a model and classify the data.

```
from spacephyml.models.mms import PCReduced
model = PCReduced('s42')

labels = {'human': [], 'classifier': [], 'epoch': []}

with torch.no_grad():
    for x, l, e in DataLoader(dataset, batch_size=32):
        lc = model(x)
        lc = torch.argmax(lc, axis = 1)
        labels['human'].extend(l)
        labels['classifier'].extend(lc)
        labels['epoch'].extend(e)
```
